import torch
from IndicTransToolkit import IndicProcessor # NOW IMPLEMENTED IN CYTHON !!
## BUG: If the above does not work, try:
# from IndicTransToolkit.IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cpu"

ip_en = IndicProcessor(inference=True)
en_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
trans_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True).to(device)


lang_codes = {
    "english": "eng_Latn",   
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "kannada": "kan_Knda"
}


def translate_to_en(text,spoken_lang):
    print(text)
    print("#################")
    sentences = [
        text
    ]
    
    batch = ip_en.preprocess_batch(sentences, src_lang=lang_codes[spoken_lang], tgt_lang=lang_codes["english"], visualize=False) # set it to visualize=True to print a progress bar
    batch = en_tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = trans_model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256,use_cache=False).to(device)
    
    outputs = en_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    outputs = ip_en.postprocess_batch(outputs, lang_codes["english"])
    return outputs[0]


ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
indic_trans_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True).to(device)

def translate_to_indic(text,trgt_lang):
    print(text)
    sentences = [
        text
    ]
    
    batch = ip.preprocess_batch(sentences, src_lang=lang_codes["english"], tgt_lang=lang_codes[trgt_lang], visualize=False) # set it to visualize=True to print a progress bar
    batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = indic_trans_model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256,use_cache=False)
    
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    outputs = ip.postprocess_batch(outputs, lang_codes[trgt_lang])
    return outputs[0]