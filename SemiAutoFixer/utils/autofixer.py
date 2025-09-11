import pandas as pd
import ast
import tabulate
from tqdm import trange
import torch
import math
from copy import deepcopy
from transformers import BertTokenizer,BertForMaskedLM
from utils.constant import VOWEL
from utils.llm_fixer import GPTStrategy, GeminiSeleniumStrategy, LLMFixer
from utils.manual_fixer import ManualFixer
from utils.normalize_text import normalize_chinese_line, normalize_vietnamese, normalize_vietnamese_line
from utils.vietnamese_similar import process_rule
from utils.words import PhoneticToOriginal, calculate_word_edit_distance, generate_similar, generate_variants
from utils.data import Data

DATA_PATH = './utils/data/'

class AutoFixer:
    def __init__(self):
        self.manual_fix_cnt = 0
        self.phonetic_chinese_dict = {}
        self.chinese_phonetic_dict = {}
        self.llm_fixer = LLMFixer(GeminiSeleniumStrategy(), self.chinese_phonetic_dict)
        print('Loading dictionary...')
        with open(f'{DATA_PATH}single_characters.tsv', 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for i in trange(len(lines)):
                line = lines[i]
                chinese, phones = line.split('\t')
                phones = [phone.strip() for phone in phones.split(',')]
                self.chinese_phonetic_dict[chinese] = phones

                for phone in phones:
                    if phone not in self.phonetic_chinese_dict:
                        self.phonetic_chinese_dict[phone] = []
                    self.phonetic_chinese_dict[phone].append(chinese)

        print('Loading similar Chinese dictionary...')
        self.sinonom_similar_dict = {}
        df = pd.read_excel(f'{DATA_PATH}SinoNom_similar_Dic.xlsx')
        for i in trange(len(df)):
            row = df.iloc[i]
            char = row['Input Character']
            similar_chars = row['Top 20 Similar Characters']
            similar_chars = ast.literal_eval(similar_chars)
            self.sinonom_similar_dict[char] = similar_chars
            self.sinonom_similar_dict[char].append(char)

        print('Loading similar phonetic dictionary...')
        self.phonetic_similar_dict = {}
        for i in trange(len(self.phonetic_chinese_dict.keys())):
            phonetic1 = list(self.phonetic_chinese_dict.keys())[i]
            if phonetic1 not in self.phonetic_similar_dict:
                self.phonetic_similar_dict[phonetic1] = []

            for phonetic2 in generate_variants(phonetic1, self.phonetic_chinese_dict):
                self.phonetic_similar_dict[phonetic1].append((phonetic2, calculate_word_edit_distance(phonetic1, phonetic2)))
            

        for phone in self.phonetic_similar_dict:
            self.phonetic_similar_dict[phone] = list(set(self.phonetic_similar_dict[phone]))
            self.phonetic_similar_dict[phone].sort(key=lambda x: x[1])                

        print('Loading morpho syllable list...')
        self.morpho_syllable_list = []
        with open(f'{DATA_PATH}MorphoSyllable_List.txt', 'r', encoding = 'utf-16') as f:
            lines = f.readlines()
            for i in trange(len(lines)):
                line = lines[i]
                self.morpho_syllable_list.append(line.strip())

        print('Loading similar phonetic dictionary...')
        self.phonetic_similar_dict = {}
        with open(f'{DATA_PATH}vietnamese_similar_dict.txt', 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for i in trange(len(lines)):
                line = lines[i]
                phonetic, similar_phonetics = line.split('\t')
                similar_phonetics = similar_phonetics.split(',')
                self.phonetic_similar_dict[phonetic] = deepcopy(similar_phonetics)

                for similar_phonetic in similar_phonetics:
                    if similar_phonetic not in self.phonetic_similar_dict:
                        self.phonetic_similar_dict[similar_phonetic] = []
                    self.phonetic_similar_dict[similar_phonetic].append(phonetic)

        print('Processing similar phonetic rules...')
        with open(f'{DATA_PATH}vietnamese_similar_rules.csv', 'r', encoding = 'utf-8') as f:
            lines = f.readlines()[1:]
            for i in trange(len(lines)):
                line = lines[i].split('\t')
                for j in range(len(line)):
                    line[j] = line[j].strip()
                    if len(line[j]) == 0:
                        line[j] = []
                    else:
                        line[j] = line[j].split(', ')

                # print(line)
                result_dict = process_rule(line[0], line[1], line[2], line[3])
                for key in result_dict:
                    if key not in self.phonetic_similar_dict:
                        self.phonetic_similar_dict[key] = []
                    self.phonetic_similar_dict[key].extend(result_dict[key])

        input('Press Enter after you have set up everything...')
    
    def test(self, word):
        # return generate_similar(word, self.phonetic_similar_dict, self.phonetic_chinese_dict)
        return generate_variants(word, self.phonetic_chinese_dict)
    
    def align_helper(self, connect_list, short_length, long_length) -> Data:
        # dp[i][j] means the maximum number of connections between 
        # the first i characters on shorter line and the first j characters on longer line
        # considering the ith character on shorter line and the jth character on longer line are connected
        dp = [[0 for _ in range(long_length)] for _ in range(short_length)]
        trace = {}
        for i in range(short_length):
            for j in range(long_length):
                trace[(i, j)] = (-1, -1)

        for a in connect_list[0]:
            dp[0][a] = 1

        for i in range(long_length):
            if i > 0:
                if dp[0][i] < dp[0][i - 1]:
                    dp[0][i] = dp[0][i - 1]
                    trace[(0, i)] = (0, i - 1)

        for i in range(len(connect_list)):
            if 0 in connect_list[i]:
                dp[i][0] = 1

        for i in range(1, short_length):
            for j in range(1, long_length):
                status = (j in connect_list[i])
                for k in range(j - 1, -1, -1):
                    if dp[i][j] < dp[i - 1][k] + status:
                        dp[i][j] = dp[i - 1][k] + status
                        trace[(i, j)] = (i - 1, k)

                for k in range(i - 1, -1, -1):
                    if dp[i][j] < dp[k][j - 1] + status:
                        dp[i][j] = dp[k][j - 1] + status
                        trace[(i, j)] = (k, j - 1)

            
            # print("Dp:")
            # for j in range(short_length):
            #     print(j, dp[j])
            # print("Trace")
            # for j in range(short_length):
            #     print(j, end=' ')
            #     for k in range(long_length):
            #         print(trace[(j, k)], end=' ')
            #     print()
            # input()        

        max_value = 0
        max_index = (-1, -1)

        for i in range(short_length):
            for j in range(long_length):
                if dp[i][j] > max_value:
                    max_value = dp[i][j]
                    max_index = (i, j)
                elif dp[i][j] == max_value and i == j:
                    max_index = (i, j)

        # print(tabulate.tabulate(dp, headers='keys', tablefmt='fancy_grid'))
        # print(tabulate.tabulate(trace, headers='keys', tablefmt='fancy_grid'))
        i, j = max_index
        result = []
        while i >= 0 and j >= 0:
            if connect_list[i] is not None and j in connect_list[i]:
                result.append((i, j))
            i, j = trace[(i, j)]

        result.reverse()

        # result = result[::-1]
        # print('Result:', result)
        return result

    def align(self, data: Data) -> Data:
        print('Aligning...' + str(data))
        chinese_line = data.chinese
        phonetic_line = data.phonetic
        
        phonetic_options = [[] for _ in range(len(phonetic_line))]
        
        # Connect missed-OCR words
        i = 0
        while i < len(phonetic_line) - 1:
            normalized_1 = normalize_vietnamese(phonetic_line[i])
            normalized_2 = normalize_vietnamese(phonetic_line[i + 1])
            vowel_exist = False
            for char in VOWEL:
                if char in normalized_1 or char in normalized_2:
                    vowel_exist = True
                    break

            if not vowel_exist:
                print('Connecting words:', phonetic_line[i], phonetic_line[i + 1])
                new_word = phonetic_line[i] + phonetic_line[i + 1]

                possible_phonetics = []
                for j in range(max(0, i - 2), min(i + 2, len(chinese_line))):
                    possible_phonetics.extend(self.chinese_phonetic_dict.get(chinese_line[j]))

                possible_phonetics = list(set(possible_phonetics))

                for phone in possible_phonetics:
                    edit_distance = calculate_word_edit_distance(phone, new_word)
                    if edit_distance <= 3:
                        phonetic_options[i].append(phone)

                for phonetic_char in self.phonetic_chinese_dict.keys():
                    edit_distance = calculate_word_edit_distance(phonetic_char, new_word)
                    if edit_distance <= 2:
                        phonetic_options[i].append(phonetic_char)

                phonetic_line.pop(i + 1)
                phonetic_line[i] = new_word

            i += 1

        link_chinese_phonetic = [[] for _ in range(min(len(chinese_line), len(phonetic_line)))]
        phonetic_options = []
        print('Phonetic:', phonetic_line)
        print('Chinese:', chinese_line)
        for i in range(len(phonetic_line)):
            phonetic_options.append([])
            indexes = [
                min(i, len(chinese_line)-1), 
                max(0, min(i - 1, len(chinese_line) - 1)),
                min(i + 1, len(chinese_line) - 1), 
                max(0, min(i - 2, len(chinese_line) - 1)),
                min(i + 2, len(chinese_line) - 1)
            ]
            indexes = list(set(indexes))
            for j in indexes:
                status, phonetic_option = self.check_phonetic_chinese(phonetic_line[i], chinese_line[j])
                # print(f'{phonetic_line[i]} - {chinese_line[j]}: {status}')
                if len(phonetic_options[i]) != 1:
                    if len(phonetic_option) == 1 and status:
                        phonetic_options[i] = phonetic_option
                    else:
                        phonetic_options[i].extend(phonetic_option)
                if status:
                    if len(phonetic_line) <= len(chinese_line):
                        link_chinese_phonetic[i].append(j)
                    else:
                        link_chinese_phonetic[j].append(i)

        print('Link:', link_chinese_phonetic)

        # (Phonetic, Chinese)
        print('Chinese before align helper:', chinese_line)
        print('Link Chinese Phonetic:', link_chinese_phonetic)
        link_result = self.align_helper(link_chinese_phonetic, min(len(chinese_line), len(phonetic_line)), max(len(chinese_line), len(phonetic_line)))
        print('Link result:', link_result)
        if len(phonetic_line) > len(chinese_line):
            for i in range(len(link_result)):
                link_result[i] = (link_result[i][1], link_result[i][0])
                print(phonetic_line[link_result[i][0]], chinese_line[link_result[i][1]])

        prev_chinese_index = 0
        prev_phonetic_index = 0

        new_chinese_line = []
        new_phonetic_line = []
        new_phonetic_options = []

        print('Chinese:', chinese_line)
        print('Phonetic:', phonetic_line)

        for i in range(len(link_result)):
            phonetic_index, chinese_index = link_result[i]
            offset = min(chinese_index - prev_chinese_index, phonetic_index - prev_phonetic_index)

            for j in range(offset):
                new_chinese_line.append(chinese_line[prev_chinese_index + j])
                new_phonetic_line.append(phonetic_line[prev_phonetic_index + j])
                new_phonetic_options.append(phonetic_options[prev_phonetic_index + j])

            prev_chinese_index += offset
            prev_phonetic_index += offset
            
            for j in range(max(0, (phonetic_index - prev_phonetic_index) - (chinese_index - prev_chinese_index))):
                new_chinese_line.append('#')
                new_phonetic_line.append(phonetic_line[prev_phonetic_index + j])
                new_phonetic_options.append(phonetic_options[prev_phonetic_index + j])

            for j in range(max(0, (chinese_index - prev_chinese_index) - (phonetic_index - prev_phonetic_index))):
                new_phonetic_line.append('#')
                new_phonetic_options.append(self.chinese_phonetic_dict.get(chinese_line[prev_chinese_index + j]))
                new_chinese_line.append(chinese_line[prev_chinese_index + j])

            new_chinese_line.append(chinese_line[chinese_index])
            new_phonetic_line.append(phonetic_line[phonetic_index])
            new_phonetic_options.append(phonetic_options[phonetic_index])

            prev_chinese_index = chinese_index + 1
            prev_phonetic_index = phonetic_index + 1
            print('Link result:', link_result[i])
            print('New Chinese:', new_chinese_line)
            print('New Phonetic:', new_phonetic_line)
            # print('New Options:', new_phonetic_options)

        for i in range(len(phonetic_line) - prev_phonetic_index):
            new_phonetic_line.append(phonetic_line[prev_phonetic_index + i])
            new_phonetic_options.append(phonetic_options[prev_phonetic_index + i])

        for i in range(len(chinese_line) - prev_chinese_index):
            new_chinese_line.append(chinese_line[prev_chinese_index + i])

        while len(new_chinese_line) < len(new_phonetic_line):
            new_chinese_line.append('#')

        while len(new_phonetic_line) < len(new_chinese_line):
            new_phonetic_line.append('#')
            new_phonetic_options.append([])

        print('Final Chinese:', new_chinese_line)
        print('Final Phonetic:', new_phonetic_line)
        print('Final Options:', new_phonetic_options)
        for i in range(len(new_phonetic_options)):
            status, opt = self.check_phonetic_chinese(new_phonetic_line[i], new_chinese_line[i])
            if (status):
                print(new_phonetic_line[i], new_chinese_line[i], ' combined')
                print(opt)
                new_phonetic_options[i] = deepcopy(opt)
            print('New phonetic options: ', new_phonetic_options[i])

        return new_phonetic_line, new_phonetic_options, new_chinese_line
    
    def check_phonetic_chinese(self, phonetic: str, chinese: str):
        phonetic_variants = generate_variants(phonetic, self.phonetic_chinese_dict)
        print(f'var: {phonetic_variants}')
        phonetic_options = []

        # Exact match
        if  (self.chinese_phonetic_dict.get(chinese) is not None) \
            and (phonetic in self.chinese_phonetic_dict.get(chinese)):
            phonetic_options.append(phonetic)

            # for variant in phonetic_variants:
            #     if variant in self.chinese_phonetic_dict.get(chinese):
            #         phonetic_options.append(variant)
            return True, phonetic_options

        if self.chinese_phonetic_dict.get(chinese) is None:
            phonetic_options.append(phonetic)
            return False, phonetic_options

        # Similar characters check
        similar_chars = [chinese]
        if self.sinonom_similar_dict.get(chinese) is not None:
            similar_chars = self.sinonom_similar_dict.get(chinese)
        chinese_chars = [chinese]
        if self.phonetic_chinese_dict.get(phonetic) is not None:
            chinese_chars = deepcopy(self.phonetic_chinese_dict.get(phonetic))
            phonetic_options.append(phonetic)

        for variant in phonetic_variants:
            # print('    +', variant)
            if variant not in phonetic_options:
                chinese_chars.extend(self.phonetic_chinese_dict.get(variant))
                phonetic_options.append(variant)

        # Remove duplicates
        chinese_chars = list(set(chinese_chars))

        if similar_chars is not None and chinese_chars is not None:
            candidates = list(set(similar_chars) & set(chinese_chars))

            if chinese in candidates:
                candidates = [chinese]
            candidates_phonetics = []
            for candidate in candidates:
                for phonetic_char in self.chinese_phonetic_dict.get(candidate):
                    candidates_phonetics.append(phonetic_char)
            if phonetic == 'phế' or phonetic == 'bặc':
                print(f'=====\n{chinese} - {phonetic}')
                print('Candidates:', candidates)
                print('Variants:', phonetic_variants)
                print('Candidates phonetics:', candidates_phonetics)
                print('Phonetic options:', phonetic_options)
            phonetic_options = list(set(phonetic_options) & set(candidates_phonetics))
            if len(candidates) > 0:
                return True, phonetic_options
            
        return False, phonetic_options

    def check_chinese_phonetic(self, phonetic: str, chinese: str):
        # Similar characters check
        similar_chars = []
        if self.sinonom_similar_dict.get(chinese) is not None:
            similar_chars = self.sinonom_similar_dict.get(chinese)

        chinese_chars = []
        if self.phonetic_chinese_dict.get(phonetic) is not None:
            chinese_chars = deepcopy(self.phonetic_chinese_dict.get(phonetic))

        if chinese_chars is None:
            return None

        # Remove duplicates
        chinese_chars = list(set(chinese_chars))

        if similar_chars is not None and chinese_chars is not None:
            candidates = list(set(similar_chars) & set(chinese_chars))
            print('Candidates:', candidates)
            if len(candidates) > 0:
                return candidates
            
        return None

    def fix(self, data: Data) -> Data:
        
        phonetic_line, phonetic_options, chinese_line = self.align(data)
        
        # Manual Fix
        need_manual_fix = False
        for i in range(len(phonetic_line)):
            if phonetic_options[i] is None:
                phonetic_options[i] = []

            if len(phonetic_options[i]) != 0:
                phonetic_options[i] = list(set(phonetic_options[i]))

            if len(phonetic_options[i]) != 1:
                old_state = need_manual_fix
                need_manual_fix = True

                if len(phonetic_options[i]) == 0:
                    if self.phonetic_chinese_dict.get(phonetic_line[i]) is not None:
                        need_manual_fix = old_state
                    else:
                        phonetic_options[i].append(phonetic_line[i])
                        phonetic_options[i].extend(self.chinese_phonetic_dict.get(chinese_line[i]))
            else:
                phonetic_line[i] = phonetic_options[i][0]

            print(f'{phonetic_options[i]}')

        if need_manual_fix:
            self.manual_fix_cnt += 1
            print('Manual Fixing: ', ' '.join(phonetic_line))
            print(' '.join(chinese_line))
            print(data.translation)
            phonetic_line = ManualFixer(phonetic_line, phonetic_options, data.page).run()

            i = 0
            while i < len(phonetic_line):
                if len(phonetic_line[i]) == 0:
                    phonetic_line.pop(i)
                    chinese_line.pop(i)
                    phonetic_options.pop(i)
                else:
                    i += 1

        print('Phonetic after manual fixing:', phonetic_line)

        # Phonetic is fixed!
        while len(chinese_line) < len(phonetic_line):
            chinese_line.append('#')
        chinese_line = chinese_line[:len(phonetic_line)]

        # for i in range(len(phonetic_line)):
        #     chinese_line[i] = self.check_chinese_phonetic(phonetic_line[i], chinese_line[i])

        # LLM Fix Chinese
        need_llm_fix = False
        chinese_options = [ [] for _ in range(len(phonetic_line)) ]
        for i in range(len(phonetic_line)):
            print(f'{phonetic_line[i]} - {chinese_line[i]}')
            possible_chinese = deepcopy(self.phonetic_chinese_dict.get(phonetic_line[i]))
            if possible_chinese is not None and chinese_line[i] in possible_chinese:
                chinese_options[i].append(chinese_line[i])
            else:
                tmp_chinese_opt = self.check_chinese_phonetic(phonetic_line[i], chinese_line[i])
                if tmp_chinese_opt is not None:
                    possible_chinese = deepcopy(tmp_chinese_opt)

                if possible_chinese is None:
                    user_choice = input(f'Cannot find Chinese for {phonetic_line[i]} - {chinese_line[i]}. Exit? (y/n)')
                    if user_choice in ['y', 'Y']:
                        raise Exception(f'Cannot find Chinese for {phonetic_line[i]} - {chinese_line[i]}')
                    else:
                        possible_chinese = [chinese_line[i]]
                chinese_options[i].extend(possible_chinese)
                if len(possible_chinese) > 1:
                    need_llm_fix = True
            chinese_options[i] = list(set(chinese_options[i]))

        print('Chinese options:', chinese_options)

        if need_llm_fix:
            print('LLM Fixing...')
            chinese_line = self.llm_fixer.fix(phonetic_line, chinese_options, data)

        return Data(chinese_line, phonetic_line, data.translation)

    def initial_check(self, data: Data) -> Data:
        
        phonetic_line, phonetic_options, chinese_line = self.align(data)
        
        # Manual Fix
        cnt = 0
        need_manual_fix = False
        for i in range(len(phonetic_line)):
            if phonetic_options[i] is None:
                phonetic_options[i] = []

            if len(phonetic_options[i]) != 0:
                phonetic_options[i] = list(set(phonetic_options[i]))

            if len(phonetic_options[i]) != 1:
                old_state = need_manual_fix
                cnt += 1
                need_manual_fix = True

                if len(phonetic_options[i]) == 0:
                    if self.phonetic_chinese_dict.get(phonetic_line[i]) is not None:
                        need_manual_fix = old_state
                    else:
                        phonetic_options[i].append(phonetic_line[i])
                        try:
                            phonetic_options[i].extend(self.chinese_phonetic_dict.get(chinese_line[i]))
                        except:
                            print(f'{chinese_line[i]} not found in dictionary')
            else:
                phonetic_line[i] = phonetic_options[i][0]

            print(f'{phonetic_options[i]}')

        if need_manual_fix:
            if cnt > len(phonetic_line) // 2:
                return False
        return True

    def fix_pair(self, line1: Data, line2: Data) -> tuple:
        overall_len = 0
        max_cnt = 0
        line_len_cnt = {}
        for line_length in [len(line1.chinese), len(line1.phonetic), len(line2.chinese), len(line2.phonetic)]:
            if line_length not in line_len_cnt:
                line_len_cnt[line_length] = 0
            line_len_cnt[line_length] += 1

            if line_len_cnt[line_length] > max_cnt:
                max_cnt = line_len_cnt[line_length]
                overall_len = line_length

        print('Overall length:', overall_len)

    def fix_all(self, data_list: list) -> list:
        return [self.fix(data) for data in data_list]