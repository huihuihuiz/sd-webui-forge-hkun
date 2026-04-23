# modules/translator_api.py

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import os

# 全局变量，初始化为 None
model = None
tokenizer = None

# 模型缓存目录（和原插件保持一致）
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# 语言选项字典（简略版）
language_options = {
    "Chinese": "zh_CN",
    "English": "en_XX",
    # 添加其他语言...
}

def load_translator():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("[Translator] Lazy-loading MBart model...")
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", cache_dir=cache_dir
        )
        tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", cache_dir=cache_dir
        )
        print("[Translator] Model loaded.")

def translate_text_api(text: str, source_lang_name: str, target_lang_name: str = "English") -> str:
    load_translator()

    src_lang = language_options.get(source_lang_name)
    tgt_lang = language_options.get(target_lang_name, "en_XX")
    
    if not src_lang:
        raise ValueError(f"Unsupported source language: {source_lang_name}")
    
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
