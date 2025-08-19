from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def call_llm(context,user_input):
    context = str(context)
    prompt = f"""<context> {context} Query: {user_input}"""
    messages = [
        {"role": "system", "content": """You are a crop disease expert assisting farmers in issues related to crop diseases . I am giving you a <context> that is just some information realated to the query . So first make use of the knowledge in the context , then use your own knowledge to give additional information. Clear Instructions : You are a text normalizer. Convert all numbers, dates, percentages, fractions, and symbols into plain spoken English. 
    Do not use any digits or special characters. 
    Example:
    - "0.2%" → "zero point two percent"
    - "Rs. 500" → "five hundred rupees"
    - "2025-08-18" → "eighteenth of August twenty twenty five"
    """},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
    return response