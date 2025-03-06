# ACL Propaganda: Fine-Tuning LLMs with Unsloth

## Project Overview
This project is focused on fine-tuning LLMs using the Unsloth library. 

## Folder Structure
```
Project_Folder
│── data/                     # Store datasets and preprocessed data
│   │── prepare_data.md        # Guide user to prepare data
│── notebooks/                 # Jupyter notebooks from Unsloth for experiments
│   │── llama3_training.ipynb 
│── src/                       # Source code for training and inference
│   │── train.py                  # Training script
│   │── inference.py              # Inference script
│   │── data_loader.py            # Data preparation and loading utilities
│── models/                    # Saved models and checkpoints
│   │── checkpoints/              # Intermediate model checkpoints
│   │── final_model/              # Final trained model
│── configs/                   # Configuration files for model training
│   │── config.yaml               # YAML file with hyperparameters
│── helper_scripts/                   # Helper scripts
│   │── setup_env.sh              # Set up dependencies
│── logs/                      # Training and evaluation logs
│── requirements.txt           # List of dependencies for the project
│── README.md                  # Project documentation and instructions
│── .gitignore                 # Ignore unnecessary files (e.g., models, logs)
```

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Project_Folder
   ```

2. **Set up the environment**:
   ```bash
   bash scripts/setup_env.sh
   ```

## Usage
- **Training**: Run the training script with the desired configuration.
  ```bash
  python main.py --mode train --config configs/config.yaml
  ```

- **Inference**: Use the inference script to generate text.
  ```bash
  python main.py --mode inference --config configs/config.yaml
  ```