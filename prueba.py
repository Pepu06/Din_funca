from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer with trust_remote_code=True
model_name = "THUDM/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Function to complete user text
def complete_text(input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
input_text = "The sun shines on a day "
generated_text = complete_text(input_text)
print(generated_text)
