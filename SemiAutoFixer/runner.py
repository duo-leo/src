from utils import AutoFixer, Data, generate_variants
from tqdm import trange
import os
from utils.llm_fixer import GeminiStrategy, GeminiSeleniumStrategy
import pandas as pd

DIR_PATH = 'WORKING_FOLDER/NewProject/'
EXCEL_PATH = ''#'WORKING_FOLDER/quan3.xlsx'
INITIAL_CHECK = False

autofixer = AutoFixer()
# autofixer.llm_fixer.strategy = GeminiSeleniumStrategy()
data = []

if len(EXCEL_PATH) > 0:
    df = pd.read_excel(EXCEL_PATH)
    df.columns = ['Chinese', 'Sino', 'Modern']
    chinese = df['Chinese'].tolist()
    phonetic = df['Sino'].tolist()
    meaning = df['Modern'].tolist()
    page_index = [0] * len(chinese)
else:
    try:
        with open(f'{DIR_PATH}original_chinese.txt', 'r', encoding = 'utf-8') as f:
            chinese = f.readlines()
    except:
        raise FileNotFoundError('original_chinese.txt not found')

    try:
        with open(f'{DIR_PATH}original_sinoviet.txt', 'r', encoding = 'utf-8') as f:
            phonetic = f.readlines()
    except:
        raise FileNotFoundError('original_sinoviet.txt not found')

    try:
        with open(f'{DIR_PATH}page-index.txt', 'r', encoding = 'utf-8') as f:
            page_index = f.readlines()
    except:
        print('page-index.txt not found, using default page index')
        page_index = [0] * len(chinese)

    try:
        with open(f'{DIR_PATH}meaning.txt', 'r', encoding = 'utf-8') as f:
            meaning = f.readlines()
    except:
        print('meaning.txt not found, using empty meaning')
        meaning = [''] * len(chinese)

print('Loading data...')
for i in trange(0, len(chinese)):
    data.append(Data(chinese[i], phonetic[i], meaning[i], page_index[i]))

if INITIAL_CHECK:
    print('Performing initial check...')
    for i in trange(0, len(data)):
        if not autofixer.initial_check(data[i]):
            if not os.path.exists(f'{DIR_PATH}error.txt'):
                with open(f'{DIR_PATH}error.txt', 'w', encoding = 'utf-8') as f:
                    pass
            with open(f'{DIR_PATH}error.txt', 'a', encoding = 'utf-8') as f:
                f.write(str(data[i]) + '\n')

    if not os.path.exists(f'{DIR_PATH}error.txt'):
        print('No error found')
    else:
        raise Exception('Error found, please check error.txt for more information')
else:
    if not os.path.exists(f'{DIR_PATH}chinese.txt'):
        with open(f'{DIR_PATH}chinese.txt', 'w', encoding = 'utf-8') as f:
            pass

    if not os.path.exists(f'{DIR_PATH}phonetic.txt'):
        with open(f'{DIR_PATH}phonetic.txt', 'w', encoding = 'utf-8') as f:
            pass

    start_index = 0
    with open(f'{DIR_PATH}chinese.txt', 'r', encoding = 'utf-8') as f:
        start_index = len(f.readlines())

    for i in trange(start_index, len(data)):
        data[i] = autofixer.fix(data[i])

        with open(f'{DIR_PATH}chinese.txt', 'a', encoding = 'utf-8') as f:
            f.write(' '.join(data[i].chinese) + '\n')

        with open(f'{DIR_PATH}phonetic.txt', 'a', encoding = 'utf-8') as f:
            f.write(' '.join(data[i].phonetic) + '\n')

        current_cnt = 0
        if not os.path.exists(f'{DIR_PATH}log.txt'):
            with open(f'{DIR_PATH}log.txt', 'w', encoding = 'utf-8') as f:
                pass
            current_cnt = 0
        else:
            with open(f'{DIR_PATH}log.txt', 'r', encoding = 'utf-8') as f:
                current_cnt = int(f.readline())

        with open(f'{DIR_PATH}log.txt', 'w', encoding = 'utf-8') as f:
            f.write(str(current_cnt + autofixer.manual_fix_cnt))
        
        autofixer.manual_fix_cnt = 0