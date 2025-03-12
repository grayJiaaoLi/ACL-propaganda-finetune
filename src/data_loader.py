"""
data_loader.py - Utilities for loading, preprocessing, and annotating propaganda detection data for Llama 3 training.

This module handles data loading, preprocessing, and formatting for Llama 3 fine-tuning
on propaganda detection tasks with a focus on the trainset_with_claims.jsonl dataset.
"""
from datasets import load_dataset
import json
import os
from typing import Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field

#  Schema Definitions

class FineLabelVerdict(str, Enum):
    """Fine-grained categorization of propaganda techniques."""

    # Emotional Appeals
    LOADED_LANGUAGE = "loaded_language"
    NAME_CALLING = "name_calling"
    APPEAL_TO_FEAR_PREJUDICE = "appeal_to_fear_prejudice"
    FLAG_WAVING = "flag-waving"
    SLOGANS = "slogans"

    # Simplification and Distortion
    REPETITION = "repetition"
    EXAGGERATION = "exaggeration"
    CAUSAL_OVERSIMPLIFICATION = "causal_oversimplification"
    BLACK_AND_WHITE_FALLACY = "black-and-white_fallacy"
    THOUGHT_TERMINATING_CLICHES = "thought-terminating_cliches"

    # Manipulating Trust and Authority
    DOUBT = "doubt"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    WHATABOUTISM = "whataboutism"
    STRAW_MAN = "straw_man"
    RED_HERRING = "red_herring"
    BANDWAGON = "bandwagon"
    REDUCTIO_AD_HITLERUM = "reductio_ad_hitlerum"

class PropagandaSpan(BaseModel):
    """An identified propaganda span within the original text with an explanation."""
    span: str = Field(..., description="The exact propaganda span extracted from the original text.")
    explanation: str = Field(..., description="The explanation why this span is considered propaganda.")
    local_label: FineLabelVerdict = Field(..., description="The appropriate label assigned towards the detected label.")

class OutputSchema(BaseModel):
    """Schema for structured LLM output after propaganda detection and normalization."""
    propaganda_spans: List[PropagandaSpan] = Field(..., description="List of identified propaganda spans.")
    global_label: FineLabelVerdict = Field(..., description="The label for the dominant propaganda technique in the statement.")


def prompt(case: str) -> List[dict]:
    """
    Constructs the prompt for LLM fine-tuning with structured JSON output.

    Args:
        case: The user-provided propaganda claim text.
    
    Returns:
        A structured prompt including system instructions.
    """

    system = {
        "role": "system",
        "content": f"""You are an intelligent annotation assistant specializing in detecting propaganda in political claims, particularly those related to the Russia-Ukraine conflict. Your task is to analyze, explain, and pre-annotate the presented text based on a set of potential propaganda classifications. You MUST return the output in valid JSON following the defined schema.

1. **Identify specific words or text spans that indicate propaganda techniques.**

2. **For each identified span, provide an explanation why it should be considered propaganda.**

3. **For each span, determine the specific propaganda technique from the following list**:
    - Loaded language: Words or phrases with strong emotional connotations that influence opinions.
    - Name calling: Labeling targets to evoke negative emotions like fear or hatred.
    - Appeal to fear/prejudice: Building support by evoking anxiety or exploiting biases.
    - Flag-waving: Appealing to national or group identity to justify actions.
    - Slogans: Concise, striking phrases that incorporate stereotyping to influence beliefs.
    - Repetition: Continuous repetition of messages to increase acceptance.
    - Exaggeration/minimization: Overstating significance or downplaying importance.
    - Causal oversimplification: Attributing issues to a single cause despite complexity.
    - Black-and-white fallacy: Presenting only two opposing options as the only choices.
    - Thought-terminating cliches: Short phrases that suppress critical thinking.
    - Doubt: Raising uncertainty about credibility to undermine trust.
    - Appeal to authority: Claiming something is true based solely on authority support.
    - Whataboutism: Deflecting criticism by pointing to others' similar issues.
    - Straw man: Misrepresenting an opponent's position to make it easier to attack.
    - Red herring: Diverting attention with irrelevant information.
    - Bandwagon: Convincing by claiming "everyone else is doing it."
    - Reductio ad hitlerum: Discrediting by associating with despised groups/individuals.

4. **Finally, determine the global label that best represents the dominant propaganda technique in the entire claim.**

**Output Format:**
```json
{json.dumps(OutputSchema.model_json_schema(), indent=4)}
```
"""
    }

    user = {
        "role": "user",
        "content": f"Analyze the following claim for propaganda techniques: \"{case}\""
    }

    return [system, user]


def format_propaganda_with_labels(examples: Dict[str, List[Any]], tokenizer):
    """
    Format the propaganda dataset with labels for supervised fine-tuning.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer for formatting
        
    Returns:
        Dictionary with formatted text
    """
    texts = []
    # Flag to track if warning has been shown
    chat_template_warning_shown = False
    
    for claim, spans, global_label in zip(
        examples["claim"],
        examples["propaganda_spans"], 
        examples["global_label"]
    ):
        # Create system and user messages
        system_msg = {
            "role": "system",
            "content": "You are an intelligent annotation assistant specializing in detecting propaganda."
        }
        
        user_msg = {
            "role": "user",
            "content": claim
        }
        
        # Create assistant response in the expected format
        output_schema = OutputSchema(
            propaganda_spans=[
                PropagandaSpan(
                    span=span["span"],
                    explanation=span["explanation"],
                    local_label=span["local_label"]
                ) for span in spans
            ],
            global_label=global_label
        )
        
        assistant_msg = {
            "role": "assistant",
            "content": json.dumps(output_schema.model_dump(), indent=2)
        }
        
        try:
            # Try to format as chat using the model's chat template
            formatted_text = tokenizer.apply_chat_template(
                [system_msg, user_msg, assistant_msg], 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            # If chat template is not available, fall back to a simple format
            if not chat_template_warning_shown:
                print(f"Warning: Could not apply chat template ({str(e)})")
                print("Using fallback format. This warning will only be shown once.")
                chat_template_warning_shown = True
                
            formatted_text = f"<s>[SYSTEM] {system_msg['content']}\n\n"
            formatted_text += f"[USER] {user_msg['content']}\n\n"
            formatted_text += f"[ASSISTANT] {assistant_msg['content']}</s>"
        
        # Add EOS token if not already present
        if not formatted_text.endswith(tokenizer.eos_token):
            formatted_text += tokenizer.eos_token
    
        texts.append(formatted_text)
    
    return {"text": texts}

def load_propaganda_dataset(data_path="../data/trainset_with_claims.jsonl"):
    """
    Load a propaganda dataset from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file (default: ../data/trainset_with_claims.jsonl)
        
    Returns:
        A HuggingFace Dataset object
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset file not found at {data_path}")
    
    # Load the dataset using the json format with lines=True for JSONL
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Validate the dataset structure
    required_columns = ["id", "claim", "propaganda_spans", "global_label"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
        
    print(f"Successfully loaded propaganda dataset with {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
    
    return dataset

def prepare_training_dataset(tokenizer, data_path="../data/trainset_with_claims.jsonl"):
    """
    Load and prepare the propaganda dataset for training.
    
    Args:
        tokenizer: Tokenizer to use for formatting
        data_path: Path to the JSONL dataset file
        supervised: Whether to prepare for supervised fine-tuning (with labels) or not
    
    Returns:
        Formatted dataset ready for training
    """
    # Load the raw dataset
    raw_dataset = load_propaganda_dataset(data_path)
    
    # Select the appropriate formatting function
    format_func = format_propaganda_with_labels
    
    # Format the dataset for training
    formatted_dataset = raw_dataset.map(
        lambda examples: format_func(examples, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    
    return formatted_dataset

if __name__ == "__main__":
    print("Loading propaganda detection dataset...")
    try:
        from transformers import AutoTokenizer
        
        # Try to load Llama-3 tokenizer, fall back to another if not available
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
        except:
            print("Could not load Llama-3 tokenizer. Using a different one for demonstration.")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            
        # Load and process the dataset
        raw_dataset = load_propaganda_dataset()
        
        # Display a sample
        if len(raw_dataset) > 0:
            example = raw_dataset[0]
            print("\nExample from dataset:")
            print(f"ID: {example['id']}")
            print(f"Claim: {example['claim']}")
            print("Propaganda spans:")
            for span in example['propaganda_spans']:
                print(f"  - Span: '{span['span']}'")
                print(f"    Explanation: {span['explanation']}")
                print(f"    Label: {span['local_label']}")
            print(f"Global label: {example['global_label']}")
            
            # Show formatted examples
            print("\nFormatting for supervised fine-tuning:")
            supervised_formatted = format_propaganda_with_labels(
                {"claim": [example["claim"]], 
                 "propaganda_spans": [example["propaganda_spans"]], 
                 "global_label": [example["global_label"]]}, 
                tokenizer
            )
            print(supervised_formatted["text"][0][:200] + "...")

            # Test preparing the entire dataset
            print("\nPreparing entire dataset for training...")
            try:
                formatted_dataset = prepare_training_dataset(tokenizer)
                print(f"Successfully prepared dataset with {len(formatted_dataset)} examples")
                print(f"Sample output: {formatted_dataset[0]['text'][:100]}...")
            except Exception as e:
                print(f"Error preparing dataset: {str(e)}")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        
    print("Data loader demonstration complete.")