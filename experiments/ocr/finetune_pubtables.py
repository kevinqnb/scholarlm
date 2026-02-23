import os
import json
import random
from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# Seeds
random.seed(342)
np.random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed_all(342)

####################################################################################################
# Configuration

MODEL_NAME = "allenai/olmOCR-2-7B-1025-FP8"
DATASET_DIR = "data/ocr/finetune_dataset"
OUTPUT_DIR = "data/ocr/finetune_output"
MERGED_OUTPUT_DIR = "data/ocr/finetune_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA hyperparameters
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training hyperparameters
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # effective batch size = 16
LEARNING_RATE = 1.5e-4
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 16384
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250

####################################################################################################
# Load dataset

print("Loading dataset...")
dataset = load_from_disk(DATASET_DIR)
train_dataset = dataset['train']
val_dataset = dataset['validation']

print(f"Train: {len(train_dataset)} examples")
print(f"Val: {len(val_dataset)} examples")

# Load metadata
with open(os.path.join(DATASET_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)
print(f"Table examples: {metadata['train_table_count']}, "
      f"Non-table examples: {metadata['train_non_table_count']}")

####################################################################################################
# Load model and processor

print("Loading model and processor...")

# Quantization config for memory-efficient training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

####################################################################################################
# Configure LoRA
#
# Target modules for Qwen2-VL (olmOCR's base architecture).
# We target attention projections and MLP layers in the language model,
# but freeze the vision encoder to preserve OCR capability.

print("Configuring LoRA...")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    modules_to_save=None,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

####################################################################################################
# Data collator: parse the stored messages JSON and format for the model

def formatting_func(example):
    """
    Parse the stored JSON messages into the chat format expected by the processor.
    Returns the formatted text string.
    """
    messages = json.loads(example['messages'])
    # Use the processor's chat template to format
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


####################################################################################################
# Training configuration

print("Setting up trainer...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,  # we use formatting_func instead
    report_to="none",  # set to "wandb" if you want W&B logging
    seed=342,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    formatting_func=formatting_func,
    peft_config=None,  # already applied above
)

####################################################################################################
# Train

print("Starting training...")
train_result = trainer.train()

# Log metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Evaluate
print("Running final evaluation...")
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Save the LoRA adapter
print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

####################################################################################################
# Merge LoRA weights into the base model for vLLM inference

print(f"Merging LoRA adapter into base model at {MERGED_OUTPUT_DIR}...")
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

from peft import PeftModel

# Reload base model in full precision for merging
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load and merge the LoRA adapter
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(MERGED_OUTPUT_DIR)
processor.save_pretrained(MERGED_OUTPUT_DIR)

print(f"Merged model saved to {MERGED_OUTPUT_DIR}/")
print("Done! You can now use the merged model with vLLM:")
print(f'  vlm = LLM("{MERGED_OUTPUT_DIR}")')