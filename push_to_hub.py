from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# (Assuming your model and tokenizer objects are loaded and fine-tuned)
# (And you are logged in via huggingface-cli login)

# Define your repository ID on the Hub
# It's typically YourHuggingFaceUsername/YourModelName
# For example, if your username is "dzure":
hub_model_id = "" # REPLACE WITH YOUR MODEL ID
commit_message_str = "Upload fine-tuned SmolLM2-360M-V2 for mental health counseling" # REPLACE WITH YOUR COMMIT MESSAGE

model = AutoModelForCausalLM.from_pretrained("") # REPLACE WITH LOCAL PATH TO BEST MODEL CHECKPOINT
tokenizer = AutoTokenizer.from_pretrained("") # REPLACE WITH LOCAL PATH TO THE SAME TOKENIZER CHECKPOINT

try:
    print(f"Pushing model to Hub: {hub_model_id}")
    model.push_to_hub(
        repo_id=hub_model_id,
        commit_message=commit_message_str,
    )
    print("Model pushed successfully!")

    print(f"Pushing tokenizer to Hub: {hub_model_id}")
    tokenizer.push_to_hub(
        repo_id=hub_model_id,
        commit_message=commit_message_str, # Can be the same or different
    )
    print("Tokenizer pushed successfully!")

except Exception as e:
    print(f"An error occurred during push_to_hub: {e}")
    print("Ensure you are logged in (huggingface-cli login) and have 'write' permissions for the repo.")