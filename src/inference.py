"""
inference.py - Script for running inference with fine-tuned Llama 3 (8B) models.

This module provides functionality for loading trained Llama 3 models and running
inference on new inputs. It handles model loading, input preprocessing, and output
generation with various inference parameters.

Functions:
    load_model: Load a fine-tuned model from disk
    format_prompt: Format input prompts for the model
    generate_text: Generate text using the model
    run_inference: Main function for running inference with a trained model
    stream_text: Generate text with streaming output
"""
import os
import yaml
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, GenerationConfig

def load_model(model_path, config):
    # Get model parameters
    max_seq_length = config["model"]["max_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)
    dtype = None  # Auto-detection
    
    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
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
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def format_prompt(instruction, input_text="", output_text="", prompt_format="chat"):
    if prompt_format.lower() == "chat":
        # Use the chat template format consistent with training
        system_msg = {
            "role": "system",
            "content": "You are an intelligent annotation assistant specializing in detecting propaganda."
        }
        
        # Combine instruction and input_text if both are provided
        if instruction and input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction or input_text
            
        user_msg = {
            "role": "user",
            "content": user_content
        }
        
        messages = [system_msg, user_msg]
        
        # Use the tokenizer's chat template
        from transformers import AutoTokenizer
        # We need to import the tokenizer here since we don't have it available yet
        try:
            temp_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
            # Use the same chat template we defined elsewhere
            if temp_tokenizer.chat_template is None:
                temp_tokenizer.chat_template = '''{% for message in messages %}
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
            prompt = temp_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback if tokenizer import fails
            prompt = f"<|system|>\n{system_msg['content']}\n<|user|>\n{user_msg['content']}\n<|assistant|>\n"
    
    elif prompt_format.lower() == "alpaca":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    else:
        # Default to a simple prompt format
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
    
    return prompt

def generate_text(model, tokenizer, prompt, generation_config=None):
    # Tokenize the prompt
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Use provided generation config or default parameters
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "use_cache": True,
        }
    
    # Generate text
    outputs = model.generate(
        **inputs,
        **generation_config,
    )
    
    # Decode the outputs
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_text[0]

def stream_text(model, tokenizer, prompt, generation_config=None):
    # Tokenize the prompt
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Use provided generation config or default parameters
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }
    
    # Create a text streamer
    text_streamer = TextStreamer(tokenizer)
    
    # Generate text with streaming
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        **generation_config,
    )

def run_inference(model_path, input_text, instruction=None, config_path=None, stream=False):
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "max_length": 2048,
                "load_in_4bit": True,
            },
            "inference": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1,
            }
        }
    
    # Load model
    model, tokenizer = load_model(model_path, config)
    
    # Format the prompt
    if instruction is None:
        # If no instruction is provided, use input_text as the instruction
        instruction = input_text
        input_text = ""
    
    # Get the prompt template format from config, default to "chat"
    prompt_format = config.get("inference", {}).get("prompt_template", "chat")
    prompt = format_prompt(instruction, input_text, prompt_format=prompt_format)
    
    # Configure generation parameters
    inference_config = config.get("inference", {})
    generation_config = {
        "max_new_tokens": inference_config.get("max_new_tokens", 512),
        "temperature": inference_config.get("temperature", 0.7),
        "top_p": inference_config.get("top_p", 0.9),
        "top_k": inference_config.get("top_k", 50),
        "repetition_penalty": inference_config.get("repetition_penalty", 1.1),
        "do_sample": inference_config.get("do_sample", True),
        "use_cache": True,
    }
    
    # Generate text
    if stream:
        # Stream the output
        return stream_text(model, tokenizer, prompt, generation_config)
    else:
        # Return the generated text
        return generate_text(model, tokenizer, prompt, generation_config)

if __name__ == "__main__":
    # Example usage
    result = run_inference(
        model_path="models/final_model",
        instruction="Explain the concept of machine learning",
        input_text="",
        config_path="configs/config.yaml",
        stream=True
    )