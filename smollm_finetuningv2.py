'''
Make sure to sign into the huggingface-hub-cli with your hf hub token before running!
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends_train.mps.is_available() else "cpu"
)

print("---- Checking for GPU ----")
print(f"Using device: {device}")
print(torch.cuda.is_available())

# import smollm2-360m and it's tokenizer
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

original_tokenizer = AutoTokenizer.from_pretrained(model_name)
#print("Original chat template:")
#print(original_tokenizer.chat_template)

# Also check special tokens
#print(f"BOS token: {original_tokenizer.bos_token}, EOS token: {original_tokenizer.eos_token}, UNK token: {original_tokenizer.unk_token}, PAD token: {original_tokenizer.pad_token}")
# If it has specific user/assistant tokens, check those too if they are part of the template logic

# runs with cpu checks and does a quick inference to test model readiness
#prompt = "Explain mental health ?"
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
#print(pipe(prompt, max_new_tokens=200, return_full_text=False))

# load the mental health dataset
# The dataset "Amod/mental_health_counseling_conversations" has a 'train' split.
# We load it and then we will split it manually.
raw_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

# tokenize and clean the mental health training dataset
def format_example(data):
    prompt = data["Context"].strip()
    response = data["Response"].strip()
    system_prompt = "You are an extremely empathetic and helpful AI assistant named SmolHealth designed to listen to the user and provide insight."

    formatted = tokenizer.apply_chat_template(
        [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt},
         {"role": "assistant", "content": response}
         ],
        tokenize=False,
        add_generation_prompt=False # Important for training
    )
    return formatted

#Check for correct auto TRL tokenization
#print(format_example(raw_dataset[0]))

# define training arguments
use_bf16 = torch.cuda.is_bf16_supported()

print(f"Using bf16: {use_bf16}")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=100,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="smollm2-mentalhealth-360m-fp16-v2", # IMPORTANT: model save directory
)

# Split the tokenized dataset into training and testing sets
# 10% of data reserved for testing
dataset_splits = raw_dataset.train_test_split(test_size=0.1, seed=training_args.seed)

ds_train_final = dataset_splits['train']
ds_test_final = dataset_splits['test']

# set up trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train_final,
    processing_class=tokenizer,
    eval_dataset=ds_test_final,
    formatting_func=format_example,
    args=training_args,
)

# train 💪
trainer.train()

# retest model after training!