from utils.normalize_text import normalize_chinese_line, normalize_vietnamese, normalize_vietnamese_line

class Data:
    def __init__(self, chinese, phonetic, translation = None, page = -1):
        if not isinstance(chinese, list):
            chinese = normalize_chinese_line(chinese)
            chinese = list(chinese)
        if not isinstance(phonetic, list):
            phonetic = normalize_vietnamese_line(phonetic)
            phonetic = phonetic.split(' ')
        
        self.chinese = chinese
        self.phonetic = phonetic
        self.translation = translation
        self.page = page

    def __str__(self):
        return f"{''.join(self.chinese)}\n{' '.join(self.phonetic)}\n{self.translation}"