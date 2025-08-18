from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bharatgenai/AgriParam"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="cpu"
)


def call_llm(context,user_input):



    prompt = f"<context> {context} <user> {user_input} <assistant>"
    # prompt = f"<context> {user_context} <user> {user_input} <assistant>"
    # prompt = f"<user> {user_input1} <assistant> {user_input2} <user> {user_input3} <assistant>..."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ### play with the hyperparameters a little .
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.6,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)