from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import pyperclip
import regex as re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import google.generativeai as genai
from utils.data import Data

def check(chinese_line, phonetic_line, chinese_phonetic_dict) -> bool:
    if len(chinese_line) != len(phonetic_line):
        return False
    
    for i, chinese in enumerate(chinese_line):
        if chinese in chinese_phonetic_dict:
            if phonetic_line[i] not in chinese_phonetic_dict[chinese]:
                print(f'Phonetic mismatch: {i} - {chinese} - {phonetic_line[i]}')
                return False
        else:
            return False
    return True

class LLMFixer():
    def __init__(self, strategy: LLMFixerStrategy, chinese_phonetic_dict) -> None:
        self._strategy = strategy
        self._strategy_using = strategy
        self.chinese_phonetic_dict = chinese_phonetic_dict

    @property
    def strategy(self) -> LLMFixerStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: LLMFixerStrategy) -> None:
        self._strategy = strategy
        self._strategy_using = strategy

    def fix(self, phonetic_line, chinese_options, data: Data):
        translation = data.translation
        print(self._strategy.getPrompt(phonetic_line, chinese_options, translation))
        retry_cnt = 0
        self._strategy = self._strategy_using
        while (True):
            response = self._strategy.fix(phonetic_line, chinese_options, translation)
            print('Response:', response)
            response = response.strip()
            response = re.sub(r'[^\w]', '', response)

            result = ''
            for i in range(len(chinese_options)):
                opt = chinese_options[i]
                if len(opt) == 1:
                    result += opt[0]
                else:
                    if i < len(response):
                        result += response[i]
                    else:
                        result += '#'
            print('Parsed response: ', result)
            
            if (check(result, phonetic_line, self.chinese_phonetic_dict)):
                return result
            
            retry_cnt += 1
            if retry_cnt > 4:
                print('Retry count exceeded, switching to another strategy')
                if True:# isinstance(self._strategy, GeminiStrategy):
                    print(f'Current line: {data.page} - {data.chinese} - {data.phonetic} - {data.translation}')
                    chinese = input('Cannot fix the line. Please input the correct Chinese sentence: ')
                    return chinese
                self._strategy = GeminiStrategy()
                retry_cnt = 0

class LLMFixerStrategy(ABC):
    @abstractmethod
    def fix(self, phonetic_line, chinese_options, translation):
        pass

    def getPrompt(self, phonetic_line, chinese_options, translation):
        options = ''
        original = ''
        cnt = 1
        for i, option in enumerate(chinese_options):
            if len(option) == 1:
                original += f'{option[0]} '
            else:
                original += f'({cnt}) '
                options += f'\n({cnt}): {option}'
                cnt += 1
        prompt = f'''Help me construct a Chinese sentence, I have its Vietnamese phonetics translation, Vietnamese meaning translation and the potential candidates for each position like this:

Masked Sentence: {original}

1. Candidates for Each Word Position: (Given as a list of lists below){options}
2. Vietnamese Phonetic Translation: {' '.join(phonetic_line)}
3. Vietnamese Meaning Translation: {translation}

Requirements:

You can only use the provided candidates for each word position.
The final sentence should be meaningful in Chinese and match the Vietnamese phonetic translation as closely as possible.
Use the Vietnamese meaning translation as context to select the most semantically appropriate characters when multiple candidates share the same phonetic pronunciation.
The length of the constructed sentence must be {len(phonetic_line)} characters.
Only output the final fully constructed sentence.
'''
        prompt = prompt.encode('utf-16', 'surrogatepass').decode('utf-16')
        return prompt

class TonguStrategy(LLMFixerStrategy):
    def __init__(self, model_path="SCUT-DLVCLab/TongGu-7B-Instruct", offload_folder="D:/tongu-offload"):
        self.model_path = model_path
        self.offload_folder = offload_folder
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map='auto',
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            offload_folder=self.offload_folder
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

    def fix(self, phonetic_line, chinese_options, translation):
        system_message = "你是通古，由华南理工大学DLVCLab训练而来的古文大模型。你具备丰富的古文知识，为用户提供有用、准确的回答。"

        option_prompt = ""
        for i, option in enumerate(chinese_options):
            option_prompt += f"第{i + 1}个字: {option}\n"

        user_query = f'''请帮我根据以下信息生成一个符合要求的中文句子：
每个位置的候选汉字：
{option_prompt}
越南语含义翻译：{translation}
要求：

根据越南意思翻译，选择最合适的汉字以匹配句子的语义和上下文。
若候选汉字中存在多个合理选项，请优先选择常用或经典的表达方式。
请直接输出生成的句子。
'''
        print(user_query)
        prompt = f"{system_message}\n<用户> {user_query}\n<通古> "
        inputs = self.tokenizer(prompt, return_tensors='pt')
        generate_ids = self.model.generate(
            inputs.input_ids.cuda(), 
            max_new_tokens=128
        )
        generate_text = self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0][len(prompt):]
        print(generate_text)
        return generate_text

class GPTStrategy(LLMFixerStrategy):
    def __init__(self) -> None:
        pass

    def fix(self, phonetic_line, chinese_options, translation):
        self.options = uc.ChromeOptions()
        self.options.headless = False
        self.driver = uc.Chrome(options=self.options)
        time.sleep(1)

        tryTimes = 0
        while True:
            try:
                self.driver.get('https://chatgpt.com/')
                time.sleep(1)

                input = self.driver.find_element(By.XPATH, '//*[@id="prompt-textarea"]')
                prompt = self.getPrompt(phonetic_line, chinese_options, translation)
                print(prompt)
                pyperclip.copy(prompt)
                input.send_keys(Keys.CONTROL, 'v')

                # for part in prompt.split('\n'):
                #     input.send_keys(part)
                #     ActionChains(self.driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()

                time.sleep(1)
                input.send_keys(Keys.ENTER)
                # find button with aria-label="Send prompt"

                time.sleep(2)

                while True:
                    try:
                        # find element with class result-streaming
                        response = self.driver.find_elements(By.CLASS_NAME, 'streaming-animation')
                        if len(response) == 0:
                            break
                        time.sleep(0.5)
                    except Exception as e:
                        print(e)
                        break
                time.sleep(1)

                # find element with data-message-author-role="assistant"
                try:                
                    response = self.driver.find_element(By.XPATH, '//div[@data-message-author-role="assistant"]').text
                except Exception as e:
                    time.sleep(2)
                    response =  self.driver.find_elements(By.XPATH, '//div[@data-message-author-role="assistant"]')[0].text
                print(response)
                self.driver.quit()
                return response
            except Exception as e:
                print(e)
                time.sleep(2)
                tryTimes += 1
                if tryTimes > 1:
                    tryTimes = 0
                    print('tryTimes > 10, exit')
                    self.driver.quit()
                    self.options = uc.ChromeOptions()
                    self.options.headless = False
                    self.driver = uc.Chrome(options=self.options)

class GeminiStrategy(LLMFixerStrategy):
    def __init__(self) -> None:
        pass

    def fix(self, phonetic_line, chinese_options, translation):
        prompt = self.getPrompt(phonetic_line, chinese_options, translation)
        # print(prompt)
        response = None
        try:
            genai.configure(api_key="AIzaSyD1Zpt3VTMTiyxGSUjOr8zAo4dGDDbCTjQ")

            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            model = genai.GenerativeModel(
                model_name="gemini-exp-1206",
                generation_config=generation_config,
            )

            chat_session = model.start_chat(
                history=[
                ]
            )
            response = chat_session.send_message(prompt)
        except Exception as e:
            print(e)
            return ''

        return response.text
    
class GeminiSeleniumStrategy(LLMFixerStrategy):
    def __init__(self):
        super().__init__()
        options = uc.ChromeOptions()
        options.headless = False
        self.driver = uc.Chrome(options=options)
        self.driver.get('https://aistudio.google.com/prompts/new_chat')

    def fix(self, phonetic_line, chinese_options, translation):
        prompt = self.getPrompt(phonetic_line, chinese_options, translation)
        try:
            self.driver.get('https://aistudio.google.com/prompts/new_chat')
            # focus this window
            self.driver.switch_to.window(self.driver.current_window_handle)
            time.sleep(3)
            # get the input box by find_element aria-label="User text input"
            input = self.driver.find_element(By.CSS_SELECTOR, 'textarea[aria-label="Type something"]')
            button = self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Run"]')

            
            pyperclip.copy(prompt)

            input.send_keys(Keys.CONTROL, 'v')

            # for part in prompt.split('\n'):
            #     input.send_keys(part)
            #     ActionChains(self.driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()

            button.click()

            time.sleep(3)

            response = ''
            prev_len = 0
            check1 = True
            check2 = True
            while True:
                try:
                    try:
                        loading_indicator = self.driver.find_element(By.CLASS_NAME, 'generating-indicator')
                    except:
                        if check1 == True:
                            check1 = False
                            print('Loading indicator found')
                            raise Exception('Loading indicator found')
                    
                    # check if class name stoppable is present in button
                    if 'stoppable' not in button.get_attribute('class'):
                        if check2 == True:
                            check2 = False
                            print('Button is not stoppable')
                            time.sleep(1)
                            raise Exception('Button is not stoppable')
                except:
                    if check1 == False and check2 == False:
                        try:
                            print(len(self.driver.find_elements(By.CSS_SELECTOR, 'ms-prompt-chunk')))
                            response = self.driver.find_elements(By.CSS_SELECTOR, 'ms-prompt-chunk')[2].text
                        except:
                            try:
                                response = self.driver.find_elements(By.CSS_SELECTOR, 'ms-prompt-chunk')[1].text
                            except:
                                response = self.driver.find_elements(By.CSS_SELECTOR, 'ms-prompt-chunk')[0].text
                        if len(response) == prev_len:
                            break
                        prev_len = len(response)
                        time.sleep(2)
                        check1 = True
                        check2 = True

            return response
        except Exception as e:
            print(e)
            return ''

# if __name__ == "__main__":
#     context = LLMFixer(GPTStrategy())
#     print("Client: Strategy is set to normal sorting.")
#     context.fix(["a", "b", "c", "d", "e"])
#     print()

# class LLMFixer:
#     def __init__(self, model_path="SCUT-DLVCLab/TongGu-7B-Instruct", offload_folder="D:/tongu-offload"):
#         self.model_path = model_path
#         self.offload_folder = offload_folder
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path, 
#             device_map='auto',
#             torch_dtype=torch.bfloat16, 
#             trust_remote_code=True, 
#             offload_folder=self.offload_folder
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.model.to(self.device)

#     def fix(self, phonetic_line, chinese_options, translation):
#         system_message = "你是通古，由华南理工大学DLVCLab训练而来的古文大模型。你具备丰富的古文知识，为用户提供有用、准确的回答。"

#         option_prompt = ""
#         for i, option in enumerate(chinese_options):
#             option_prompt += f"第{i + 1}个字: {option}\n"

#         user_query = f'''请帮我根据以下信息生成一个符合要求的中文句子：
# 每个位置的候选汉字：
# {option_prompt}
# 越南语含义翻译：{translation}
# 要求：

# 根据越南意思翻译，选择最合适的汉字以匹配句子的语义和上下文。
# 若候选汉字中存在多个合理选项，请优先选择常用或经典的表达方式。
# 请直接输出生成的句子。
# '''
#         print(user_query)
#         prompt = f"{system_message}\n<用户> {user_query}\n<通古> "
#         inputs = self.tokenizer(prompt, return_tensors='pt')
#         generate_ids = self.model.generate(
#             inputs.input_ids.cuda(), 
#             max_new_tokens=128
#         )
#         generate_text = self.tokenizer.batch_decode(
#             generate_ids, 
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0][len(prompt):]
#         print(generate_text)
#         return generate_text