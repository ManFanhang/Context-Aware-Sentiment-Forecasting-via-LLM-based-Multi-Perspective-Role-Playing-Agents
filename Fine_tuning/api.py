from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
import os

app = Flask(__name__)

# Define the LoRA configuration
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Define the model and LoRA weight path
mode_path = os.path.expanduser('~/LLM-Research/Meta-Llama-3-8B-Instruct')
lora_path = os.path.expanduser('~/LLaMA3/psy')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)
tokenizer.pad_token = tokenizer.eos_token  # Setting a pad token

# Load the base model
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

# Load LoRA weights to the base model
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('text', '')

    if not user_input:
        return jsonify({"response": "No input text provided."})

    # Generate model input
    model_inputs = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')

    # Generate a reply
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Remove the ID from the input section
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006) #The ports set must be consistent
