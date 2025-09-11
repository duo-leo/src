from utils.normalize_text import denormalize_vietnamese, normalize_vietnamese

class WordFrequency:
    def __init__(self, chinese_word, freq):
        self.han = chinese_word
        self.freq = freq

    def increase(self):
        self.freq += 1

    def __hash__(self) -> int:
        return hash(self.han)

    def __lt__(self, other):
        return self.freq < other.freq
    
    def __str__(self) -> str:
        return (self.freq, self.han)
    
class PhoneticToOriginal:
    def __init__(self):
        self.chinese_freq_dict = {}

    def add(self, orignal_word):
        if orignal_word in self.chinese_freq_dict:
            self.chinese_freq_dict[orignal_word].increase()
        else:
            self.chinese_freq_dict[orignal_word] = WordFrequency(orignal_word, 1)

    def __lt__(self, other):
        return self.freq < other.freq
    
    def __hash__(self) -> int:
        return hash(self.chinese_freq_dict)
    
    def __str__(self) -> str:
        words = []
        for chinese_word in self.chinese_freq_dict:
            words.append(self.chinese_freq_dict[chinese_word])
        words.sort(reverse=True)
        result = []
        for word in words:
            result.append(str(word[0]) + ' ' + str(word[1]))
        return ' '.join(result)

def calculate_word_edit_distance(word1, word2) -> int:
    word1 = normalize_vietnamese(word1)
    word2 = normalize_vietnamese(word2)
    
    n = len(word1)
    m = len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1

    return dp[n][m]

def generate_similar(word, similar_dict, dict) -> list:
    word = word.lower()
    similar_words = []

    for key in similar_dict.keys():
        if key in word:
            for similar_char in similar_dict[key]:
                new_word = word.replace(key, similar_char)
                if new_word in dict:
                    similar_words.append(new_word)

    similar_words = list(set(similar_words))
    print(similar_words)
    return similar_words

def generate_variants(word, dict) -> list:
    word = word.lower()
    word = normalize_vietnamese(word)
    variants = []

    tonemarks = [0, 1, 2, 3, 4, 5]
    char_variants = {
        'a': ['a', 'aw', 'aa'],
        'e': ['e', 'ee'],
        'd': ['d', 'dd'],
        'o': ['o', 'oo', 'ow'],
        'u': ['u', 'uw'],
        'i': ['i', 'y'],
        'y': ['y', 'i']
    }

    for char in char_variants:
        for variant in char_variants[char]:
            word = word.replace(variant, char)

    word = ''.join([i for i in word if not i.isdigit()])


    # for char_variant in char_variants:
    #     if char_variant in word:
    #         for variant in char_variants[char_variant]:
    #             for i in tonemarks:
    #                 if i == 0:
    #                     new_word = denormalize_vietnamese(word.replace(char_variant, variant))
    #                 else:
    #                     new_word = denormalize_vietnamese(word.replace(char_variant, variant) + str(i))
    #                 print(new_word)
    #                 if new_word in dict:
    #                     variants.append(new_word)

    # word_variant, index
    variant_stack = [(word, 0)]
    while len(variant_stack) > 0:
        current_word = variant_stack.pop()
        word_variant = current_word[0]
        index = current_word[1]

        if index >= len(word_variant):
            continue
        if word_variant[index] not in char_variants:
            variant_stack.append((word_variant, index + 1))
            continue

        for variant in char_variants[word_variant[index]]:
            for i in tonemarks:
                if i == 0:
                    new_word = denormalize_vietnamese(word_variant[:index] + variant + word_variant[index + 1:])
                else:
                    new_word = denormalize_vietnamese(word_variant[:index] + variant + str(i) + word_variant[index + 1:])
                if new_word in dict:
                    variants.append(new_word)
                variant_stack.append((new_word, index + 1))

    variants = list(set(variants))

    return variants