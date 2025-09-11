from copy import deepcopy
from utils.normalize_text import denormalize_vietnamese

def process_rule(vowel, tone_mark, new_vowel, new_tone_mark):
    result_dict = {}
    vowel_list = ['a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y']
    if len(vowel) == 0:
        vowel = deepcopy(vowel_list)
    
    if len(new_vowel) == 0:
        new_vowel = deepcopy(vowel)

    if len(tone_mark) == 0:
        tone_mark = [0]

    if len(new_tone_mark) == 0:
        new_tone_mark = [0]

    for i in range(len(vowel)):
        for j in range(len(tone_mark)):
            original = denormalize_vietnamese(vowel[i] + str(tone_mark[j]))
            for k in range(len(new_tone_mark)):
                processed = denormalize_vietnamese(new_vowel[i] + str(new_tone_mark[k]))
                # print(original, processed)
                if result_dict.get(original) is None:
                    result_dict[original] = [processed]
                else:
                    result_dict[original].append(processed)

                if result_dict.get(processed) is None:
                    result_dict[processed] = [original]
                else:
                    result_dict[processed].append(original)

    return result_dict

