from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_path = "" # REPLACE MODEL PATH!
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad token is set if tokenizer doesn't have one (pipeline might need it)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create a pipeline
# We will format the text *before* sending it to the pipeline's generator call
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Model loaded and ready for interaction.")

# Define a more specific system prompt for your fine-tuned model
system_prompt_content = "You are an extremely empathetic and helpful AI assistant named SmolHealth designed to listen to the user and provide insight."

while True:
    print("\nType 'quit' to leave the conversation.")
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    # 1. Construct the messages list with system and user prompts
    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_input}
    ]

    # 2. Apply the chat template
    # add_generation_prompt=True is crucial to add the cue for the assistant to start responding
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        print("Ensure your tokenizer has a chat_template attribute properly configured.")
        continue # Skip this turn if formatting fails

    # 3. Generate a response using the fully formatted prompt
    # Pass generation parameters directly here for more control
    response = generator(
        formatted_prompt,
        max_new_tokens=1024,          # Increased slightly
        num_return_sequences=1,
        return_full_text=False,      # Get only the newly generated text
        do_sample=True,              # Use sampling
        temperature=0.8,             # Adjust for creativity vs. focus
        top_p=0.9,                   # Nucleus sampling
        # repetition_penalty=1.1,    # Optionally try to reduce parroting further
    )

    print("Model:", response[0]['generated_text'].strip())

print("Exiting.")
