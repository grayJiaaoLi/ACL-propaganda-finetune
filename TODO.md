# Llama 3 (8B) Fine-tuning Project TODO List

## 1. Environment Setup
  - [ ] Load or Set up virtual environment
  - [ ] Verify all dependencies in requirements.txt are installed
  - [ ] Ensure GPU access with CUDA compatibility, and check multi-GPU accessbility
  - [ ] Configure Hugging Face token for accessing gated models(Later)

## 2. Data Preparation
  - [x] Implement the data_loader.py module based on notebook examples
  - [x] Ensure EOS tokens are properly added to formatted data
  - [ ] Split a validation dataset for evaluating fine-tuning progress from the selected trainset

## 3. Model Initialization
  - [x] Complete model initialization in train.py
  - [x] Implement FastLanguageModel initialization with proper parameters
  - [x] Configure max_seq_length, dtype, and 4-bit quantization settings
  - [x] Add LoRA adapter configuration with specified target modules
  - [x] Implement gradient checkpointing for efficient training

## 4. Training Pipeline
  - [x] Implement the full training_model function in train.py
  - [x] Set up SFTTrainer with proper configuration
  - [x] Configure Training Arguments with appropriate hyperparameters
  - [x] Implement checkpointing and model saving logic
  - [x] Add basic training metrics logging

## 5. Inference Implementation
  - [x] Complete the inference.py module for running predictions
  - [x] Implement model loading functionality
  - [x] Create prompt formatting similar to notebook examples
  - [x] Add support for different generation parameters
  - [x] Implement text streaming for realtime generation
  - [x] Support batch inference

## 6. Configuration System
  - [x] Update config.yaml with all necessary parameters
  - [x] Ensure all training parameters are configurable
  - [x] Add data preprocessing configuration options
  - [x] Include inference parameters
  - [x] Add model saving options

## 7. Main Script Integration
  - [x] Complete main.py to organise the full pipeline
  - [x] Implement command-line argument parsing
  - [x] Create workflow for training mode
  - [x] Create workflow for inference mode
  - [x] Add configuration loading and validation

## 8. Evaluation
  - [ ] Add evaluation metrics for model performance
  - [ ] Add loss tracking and store the training process data into logs

## 9. Testing
  - [ ] Validate model initialization
  - [ ] Test inference on sample prompts
  - [ ] Ensure configuration loading works properly

## Next Steps
1. Create a simple example script that demonstrates the full pipeline
2. Test the implementation on a small dataset to verify functionality
