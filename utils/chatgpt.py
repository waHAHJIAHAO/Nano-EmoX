import openai

global_index = 0
candidate_keys = ["sk-xxx"]
openai.api_key = candidate_keys[global_index]


def func_get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print('发生错误：', e)
        global global_index
        global_index = (global_index + 1) % len(candidate_keys)
        print(f'========== key index: {global_index} ==========')
        openai.api_key = candidate_keys[global_index]
        return ''


def get_completion(prompt, model, maxtry=5):
    response = ''
    try_number = 0
    while len(response) == 0:
        try_number += 1
        if try_number == maxtry:
            print(f'fail for {maxtry} times')
            break
        response = func_get_completion(prompt, model)
    return response


def func_postprocess_chatgpt(response):
    response = response.strip()
    if response.startswith("输入"):   response = response[len("输入"):]
    if response.startswith("输出"):   response = response[len("输出"):]
    if response.startswith("翻译"):   response = response[len("翻译"):]
    if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    if response.startswith("input"): response = response[len("input"):]
    if response.startswith("Input"): response = response[len("Input"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    if response.startswith("："): response = response[len("："):]
    response = response.strip()
    response = response.replace('\n', '')
    response = response.strip()
    return response


def get_translate_eng2chi(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    text = text.replace('\n', '')
    prompt = f"""
              请将以下输入翻译为中文：

              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    return response


def get_translate_chi2eng(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    text = text.replace('\n', '')
    prompt = f"""
              请将以下输入翻译为英文：

              输入：我爱你

              输出：I love you

              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    return response
