"""
train.py - Script for training Llama 3 (8B) models using the Unsloth library.

This module contains functionality for fine-tuning Llama 3 models on custom datasets.
It includes model initialization, loading of pre-processed data, training loop implementation,
and checkpoint saving.

Functions:
    initialize_model: Initialize the base Llama 3 model
    setup_lora: Configure LoRA adapters for efficient fine-tuning
    setup_trainer: Set up the SFTTrainer with appropriate configuration
    train_model: Main function for training the model using configurations
    save_model: Helper function for saving model checkpoints
"""
import os
import yaml
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import time
from src.data_loader import prepare_training_dataset

def initialize_model(config):
    """
    Initialize the base Llama model and tokenizer.
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        Tuple: (model, tokenizer)
    """
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)
    dtype = None  # Auto-detection
    
    # HF token for gated models
    token = config["model"].get("hf_token", None)
    
    print(f"Initializing model {model_name} with sequence length {max_seq_length}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token
        )
        
        # Add chat template for the tokenizer if not already present
        if tokenizer.chat_template is None:
            # Default chat template for Llama models - this works for Llama 3.1
            tokenizer.chat_template = '''{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}
'''
            print(f"Added default Llama 3.1 chat template to tokenizer")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def setup_lora(model, config):
    """
    Configure LoRA adapters for efficient fine-tuning.
    
    Args:
        model: The base model to apply LoRA to
        config: Configuration dictionary with model settings
        
    Returns:
        model: The model with LoRA adapters configured
    """
    lora_config = config["model"]
    lora_rank = lora_config.get("lora_rank", 16)
    lora_alpha = lora_config.get("lora_alpha", 16)
    lora_dropout = lora_config.get("lora_dropout", 0)
    use_rslora = lora_config.get("use_rslora", False)
    
    # Target modules for LoRA adaptation - typically attention layers and MLP
    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Use gradient checkpointing for memory efficiency
    use_gradient_checkpointing = lora_config.get("use_gradient_checkpointing", True)
    
    # Set random seed for reproducibility
    random_state = config.get("inference", {}).get("seed", 3407)
    
    print(f"Setting up LoRA with rank {lora_rank}, alpha {lora_alpha}, dropout {lora_dropout}")
    if use_rslora:
        print("Using Rank-Stabilized LoRA (RSLoRA)")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",  # "none" is optimized in Unsloth
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=None   # LoftQ for quantization-aware training
    )
    
    # Print trainable parameters information
    model.print_trainable_parameters()
    
    return model

def setup_trainer(model, tokenizer, datasets, config):
    """
    Set up the SFTTrainer with appropriate configuration.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer for the model
        datasets: Dictionary containing 'train' and 'eval' datasets
        config: Training configuration
        
    Returns:
        SFTTrainer: Configured trainer
    """
    train_config = config["training"]
    max_seq_length = config["model"]["max_length"]
    
    # Extract datasets
    train_dataset = datasets.get('train')
    eval_dataset = datasets.get('eval')
    
    if train_dataset is None:
        raise ValueError("Training dataset is required")
    
    # Ensure eval_dataset is properly handled
    if eval_dataset is None:
        print("Warning: No evaluation dataset provided. Training will proceed without evaluation.")
    
    # Get training parameters from config
    batch_size = train_config.get("batch_size", 2)
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 4)
    warmup_steps = train_config.get("warmup_steps", 100)
    learning_rate = train_config.get("learning_rate", 2e-4)
    epochs = train_config.get("epochs", 3)
    max_steps = train_config.get("max_steps", None)
    save_steps = train_config.get("save_steps", 500)
    logging_steps = train_config.get("logging_steps", 10)
    optimizer = train_config.get("optimizer", "adamw_8bit")
    lr_scheduler = train_config.get("lr_scheduler", "linear")
    weight_decay = train_config.get("weight_decay", 0.01)
    
    # Output directories
    output_dir = config["output"]["checkpoint_dir"]
    log_dir = config["output"]["log_dir"]
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        logging_dir=log_dir,
        output_dir=output_dir,
        save_steps=save_steps,
        optim=optimizer,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        seed=3407,
        report_to="tensorboard",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=save_steps if eval_dataset is not None else None,
    )
    
    # Determine if packing should be used (can speed up training for short sequences)
    packing = train_config.get("packing", False)
    
    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=os.cpu_count() // 2,  # Use half of available CPUs
        packing=packing,
        args=training_args,
    )
    
    return trainer

def save_model(model, tokenizer, output_dir, save_method="lora"):
    """
    Save the trained model using the specified method.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        output_dir: Directory to save the model to
        save_method: Method to use for saving the model. Options:
            - 'lora': Save only the LoRA adapters (smallest)
            - 'merged_16bit': Save fully merged model in float16
            - 'merged_4bit': Save fully merged model in 4-bit quantization
    
    Returns:
        None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model to {output_dir} using method: {save_method}")
        
        if save_method == "lora":
            # Save LoRA adapters only (smallest)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Successfully saved LoRA adapters to {output_dir}")
        
        elif save_method == "merged_16bit":
            # Save fully merged model in float16
            model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
            print(f"Successfully saved merged 16-bit model to {output_dir}")
            
        elif save_method == "merged_4bit":
            # Save fully merged model in 4-bit quantization
            model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
            print(f"Successfully saved merged 4-bit model to {output_dir}")
        
        else:
            raise ValueError(f"Unknown save method: {save_method}. Supported methods: lora, merged_16bit, merged_4bit")
            
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def train_model(config_path):
    """
    Main function for training the model using configurations.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple: (model, tokenizer, trainer_stats)
    """
    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {str(e)}")
    
    # 1: Initialize model
    model, tokenizer = initialize_model(config)
    
    # 2: Load and prepare dataset
    data_path = config.get("data", {}).get("train_path", "../data/trainset_with_claims.jsonl")
    train_dataset = prepare_training_dataset(tokenizer, data_path)
    
    # Check if there's a separate evaluation dataset
    eval_path = config.get("data", {}).get("eval_path", None)
    eval_dataset = prepare_training_dataset(tokenizer, eval_path) if eval_path else None
    
    datasets = {'train': train_dataset, 'eval': eval_dataset}

    # If eval_dataset is not provided, split train_dataset
    if eval_dataset is None and config.get("training", {}).get("use_validation", True):
        print("No evaluation dataset provided. Splitting training dataset...")
        train_size = config.get("training", {}).get("train_split", 0.8)
        
        try:
            # Use datasets' built-in split functionality instead of sklearn's train_test_split
            split_datasets = train_dataset.train_test_split(
                test_size=(1 - train_size),
                seed=config.get("inference", {}).get("seed", 36)
            )
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"]
            
            datasets = {'train': train_dataset, 'eval': eval_dataset}
            print(f"Split dataset: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
        except (TypeError, AttributeError) as e:
            print(f"Error when splitting dataset: {e}")
            print("This may happen if your dataset doesn't support Hugging Face's train_test_split method.")
            print("Proceeding with original dataset (no validation set)")
            datasets = {'train': train_dataset, 'eval': None}
    else:
        datasets = {'train': train_dataset, 'eval': eval_dataset}

    # 3: Setup LoRA
    model = setup_lora(model, config)
    
    # 4: Setup trainer
    trainer = setup_trainer(model, tokenizer, datasets, config)
    
    # Log memory usage before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")
    
    # 5: Train model
    print("Starting training...")
    start_time = time.time()
    trainer_stats = trainer.train()
    end_time = time.time()
    
    # Log training time and memory usage
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak memory for training = {used_memory_for_training} GB.")
    
    # 6: Save model
    final_model_dir = config["output"]["final_model_dir"]
    save_method = config.get("output", {}).get("save_method", "lora")
    save_model(model, tokenizer, final_model_dir, save_method)
    
    # Prepare model for inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer, trainer_stats

if __name__ == "__main__":
    train_model("configs/config.yaml") 