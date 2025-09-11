from utils.autofixer import AutoFixer
from utils.normalize_text import normalize_vietnamese, normalize_vietnamese_line, normalize_chinese_line
from utils.preprocess import viet_preprocess
from utils.words import WordFrequency, PhoneticToOriginal, calculate_word_edit_distance, generate_variants, generate_similar
from utils.manual_fixer import ManualFixer
from utils.constant import *
from utils.llm_fixer import LLMFixer, TonguStrategy, GPTStrategy, GeminiStrategy, GeminiSeleniumStrategy
from utils.vietnamese_similar import process_rule
from utils.data import Data