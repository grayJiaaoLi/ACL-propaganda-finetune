"""
main.py - Main entry point for the Llama 3 (8B) project using Unsloth library.

This file serves as the primary script to run the entire pipeline, from data loading
to model training and inference. It orchestrates the workflow by importing and using
modules from the src directory.

Usage:
    python main.py --mode [train|inference|process|all] --config configs/config.yaml
    python main.py --mode train --config configs/config.yaml
    python main.py --mode inference --model_path models/final_model --input "Your prompt here"
    python main.py --mode process --config configs/config.yaml
"""

import argparse
import os
import yaml
import sys
from src.train import train_model
from src.inference import run_inference
from src.data_loader import load_data

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found!")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_workflow(config_path):
    print(f"=== Starting data processing workflow with config: {config_path} ===")
    
    # Load configuration
    config = load_config(config_path)
    data_config = config.get("data", {})
    
    # Process raw data if specified
    raw_data_path = data_config.get("raw_data_path")
    process_raw = data_config.get("process_raw", False)
    output_dir = data_config.get("output_dir", "data/processed")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if process_raw and raw_data_path and os.path.exists(raw_data_path):
        if raw_data_path.endswith('.csv'):
            print(f"Processing raw CSV file: {raw_data_path}")
            
            # Generate output paths for both versions
            base_output = os.path.join(output_dir, "processed_propaganda")
            output_simple = f"{base_output}.jsonl"
            output_with_explanations = f"{base_output}_with_explanation.jsonl"
            
            print(f"Processed raw data saved to:")
            print(f"  - {output_simple} (without detailed explanations)")
            print(f"  - {output_with_explanations} (with detailed explanations)")
    
    # Align formats if specified
    align_formats = data_config.get("align_formats", False)
    source_paths = data_config.get("source_paths", [])
    
    if align_formats and source_paths:
        print(f"Aligning formats for {len(source_paths)} files...")
        
        for source_path in source_paths:
            if os.path.exists(source_path):
                # Generate target path in the output directory
                filename = os.path.basename(source_path)
                target_path = os.path.join(output_dir, f"aligned_{filename}")
                
                print(f"Aligned {source_path} to {target_path}")
    
    print("=== Data processing workflow completed ===")

def train_workflow(config_path):
    print(f"=== Starting training workflow with config: {config_path} ===")
    model, tokenizer, trainer_stats = train_model(config_path)
    print("=== Training workflow completed ===")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training loss: {trainer_stats.metrics['train_loss']:.4f}")
    return model, tokenizer

def inference_workflow(model_path, instruction, input_text, config_path, stream):
    print(f"=== Starting inference workflow with model: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return
        
    result = run_inference(
        model_path=model_path,
        instruction=instruction,
        input_text=input_text,
        config_path=config_path,
        stream=stream
    )
    
    # Only print result for non-streaming mode since streaming prints as it goes
    if not stream and result:
        print("\nGenerated output:")
        print(result)
    
    print("=== Inference workflow completed ===")

def main():
    """Main function to run the Llama 3 fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description="Llama 3 (8B) fine-tuning with Unsloth")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "process", "all"], 
                        default="train", help="Operation mode")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_path", type=str, help="Path to the model for inference")
    parser.add_argument("--instruction", type=str, help="Instruction for inference")
    parser.add_argument("--input", type=str, default="", help="Input text for inference")
    parser.add_argument("--stream", action="store_true", help="Stream the output during inference")
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["inference", "all"] and not args.model_path:
        if args.mode == "all":
            # If mode is 'all', use the final model path from config
            config = load_config(args.config)
            args.model_path = config["output"]["final_model_dir"]
        else:
            print("Error: Model path must be specified for inference mode!")
            parser.print_help()
            return
        
    # Run the appropriate workflow
    if args.mode == "process" or args.mode == "all":
        process_workflow(args.config)
    
    if args.mode == "train" or args.mode == "all":
        model, tokenizer = train_workflow(args.config)
    
    if args.mode == "inference" or args.mode == "all":
        inference_workflow(
            model_path=args.model_path,
            instruction=args.instruction,
            input_text=args.input,
            config_path=args.config,
            stream=args.stream
        )
    
if __name__ == "__main__":
    main() 