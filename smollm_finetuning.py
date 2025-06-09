'''
Make sure to sign into the huggingface-hub-cli with your hf hub token before running!

(This was used for the v1 version of the model)
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
print("Original chat template:")
print(original_tokenizer.chat_template)

# Also check special tokens
print(f"BOS token: {original_tokenizer.bos_token}, EOS token: {original_tokenizer.eos_token}, UNK token: {original_tokenizer.unk_token}, PAD token: {original_tokenizer.pad_token}")
# If it has specific user/assistant tokens, check those too if they are part of the template logic

'''
# set chat format to none
tokenizer.chat_template = None

# reset the chat format
from trl.models.utils import setup_chat_format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)'''

# runs with cpu checks and does a quick inference to test model readiness
prompt = "Explain mental health ?"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
print(pipe(prompt, max_new_tokens=200, return_full_text=False))

exit()

# load the mental health dataset
# The dataset "Amod/mental_health_counseling_conversations" has a 'train' split.
# We load it and then we will split it manually.
raw_dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

# Ensure raw dataset loads correctly by printing 10 pairs of model responses
print(raw_dataset["Response"][0:9])

# tokenize and clean the mental health training dataset
def tokenize_function(examples):
    prompts = [p.strip() for p in examples["Context"]]
    responses = [r.strip() for r in examples["Response"]]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False
        )
        for p, r in zip(prompts, responses)
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

# Tokenize the entire loaded dataset
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)

# define training arguments
use_bf16 = torch.cuda.is_bf16_supported()

print(f"Using bf16: {use_bf16}")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="smollm2-mentalhealth-360m-fp16",
)

# Split the tokenized dataset into training and testing sets
# 10% of data reserved for testing
dataset_splits = tokenized_dataset.train_test_split(test_size=0.1, seed=training_args.seed)

ds_train_final = dataset_splits['train']
ds_test_final = dataset_splits['test']

# set up trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds_train_final,
    eval_dataset=ds_test_final,
    args=training_args,
)

# train 💪
trainer.train()

# save the model
save_directory = "smollm2-mentalhealth-360m-fp16"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# retest model after training!