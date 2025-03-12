# Llama 3 (8B) Fine-Tuning with Unsloth

## Project Overview

This project provides a complete framework for fine-tuning Llama 3 (8B) models using the Unsloth library. Unsloth enables faster and more memory-efficient fine-tuning of large language models. The project structure follows best practices for ML projects and provides a modular codebase for easy customization.

Key features:
- Parameter-efficient fine-tuning with LoRA
- Support for different datasets including Alpaca format
- Optimized training with 4-bit quantization
- Flexible configuration system
- Various model export formats (LoRA adapters, merged 16-bit, merged 4-bit, GGUF)
- Interactive inference with streaming output

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ACL_Propaganda_Finetune
   ```

2. **Set up the environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Hugging Face Access**:
   To access gated models (e.g., Llama 3), you need to set up a Hugging Face token:
   ```bash
   # Option 1: Environment variable
   export HF_TOKEN=your_huggingface_token
   
   # Option 2: Update config.yaml
   # Uncomment and set the hf_token field in the config.yaml file
   ```

## Project Structure

```
ACL_Propaganda_Finetune/
│── data/                     # Store datasets and preprocessed data
│   │── raw_data/                     # Raw datasets before processing
│   │── trainset/                # Processed ready to fine-tune datasets
│── notebooks/                 # Jupyter notebooks for experiments
│   │── Llama3_1_(8B)_Alpaca.ipynb # Original notebook for reference
│── src/                       # Source code for training and inference
│   │── train.py                  # Training script
│   │── inference.py              # Inference script
│   │── data_loader.py            # Data preparation and loading utilities
│── models/                    # Directory for saved models
│   │── checkpoints/              # Training checkpoints
│   │── final_model/              # Final trained models
│── configs/                   # Configuration files for model training
│   │── config.yaml               # YAML file with hyperparameters
│── logs/                      # Training and evaluation logs
│── requirements.txt           # List of dependencies for the project
│── main.py                    # Main entry point for the pipeline
│── README.md                  # Project documentation and instructions
│── TODO.md                    # Task list and progress tracking
│── .gitignore                 # Ignore unnecessary files (e.g., models, logs)
```

## Usage
