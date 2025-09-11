from underthesea import text_normalize

def viet_preprocess(text):
    text = text.strip()
    text = text_normalize(text)
    return text

