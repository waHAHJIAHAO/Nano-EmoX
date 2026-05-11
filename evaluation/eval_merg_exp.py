from collections import defaultdict
import numpy as np
import re
from typing import List, Union, Dict, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score
from collections import defaultdict, Counter

emotion_projection_map_empathe_mead = {
    "neutral": "neutral", 
    "happy": "happy", 
    "surprised": "surprised", 
    "angry": "angry", 
    "fear": "fear", 
    "sad": "sad", 
    "disgusted": "disgusted", 
    "contempt": "contempt",

    # happy
    "joyful": "happy",
    "prepared": "happy",
    "content": "happy",
    "caring": "happy",
    "trusting": "happy",
    "faithful": "happy",
    "confident": "happy",
    "hopeful": "happy",
    "grateful": "happy",
    "proud": "happy",
    "excited": "happy",
    "anticipating": "happy",

    # sad
    "lonely": "sad",
    "guilty": "sad",
    "anxious": "sad",
    "nostalgic": "sad",
    "embarrassed": "sad",
    "disappointed": "sad",
    "sentimental": "sad",
    "ashamed": "sad",
    "devastated": "sad",

    # surprised
    "impressed": "surprised",

    # angry
    "furious": "angry",
    "annoyed": "angry",

    # fear
    "afraid": "fear",
    "terrified": "fear",
    "apprehensive": "fear",

    # disgusted
    "embarrassed": "disgusted",

    # contempt
    "jealous": "contempt",
}

def evaluate_dist_n(data, n):
    import numpy as np
    from collections import Counter

    def _distinct_n(data, n):

        def get_ngrams(resp, n):
            tokens = resp.split()
            return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]

        dist_results = []
        for sent in data:
            ngrams = get_ngrams(sent.strip().lower(), n)
            counter = Counter()
            counter.update(ngrams)

            ngram_counter = counter

            if sum(ngram_counter.values()) == 0:
                # print("Warning: encountered a response with no {}-grams".format(n))
                # print(sent.strip().lower())
                # print("ngram_counter: ", ngram_counter)
                continue
                
            dist = len(ngram_counter) / sum(ngram_counter.values())
            dist_results.append(dist)
        
        if not dist_results:
            return 0.0
        return np.average(dist_results)
    
    return _distinct_n(data, n)

def calculate_hitrate(name2gt: Dict[str, str],
                      name2pred: Dict[str, str]) -> float:
    """
    计算hitrate，即预测情感与真实情感一致的比例
    
    参数:
    - name2gt: 样本名称到真实情感标签的映射
    - name2pred: 样本名称到预测情感标签的映射
    
    返回:
    - hitrate (0-100之间的浮点数)
    """
    if not name2gt or not name2pred:
        return 0.00
    
    hit_count = 0
    total_count = 0
    
    for name in name2gt:
        if name in name2pred:
            total_count += 1
            if emotion_projection_map_empathe_mead[name2gt[name]] == emotion_projection_map_empathe_mead[name2pred[name]]:
                hit_count += 1
    
    if total_count == 0:
        return 0.00
    
    return (hit_count / total_count) * 100

def extract_emotion_from_response(response_text: str, emotion_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    从生成的回复中提取情感标签
    
    参数:
    - response_text: 生成的回复文本
    - emotion_mapping: 情感映射字典，用于标准化情感标签
    
    返回:
    - 提取的情感标签（标准化后）
    """
    if not response_text:
        return "unknown"
    
    # 常见情感词汇列表（可根据需要扩展）
    emotion_keywords = {
    "angry": ["angry", "anger", "furious", "mad", "irate", "enraged"],
    "annoyed": ["annoyed", "annoyance", "irritated", "irritation", "vexed", "bothered"],
    "caring": ["caring", "compassionate", "empathetic", "kind", "sympathetic", "concern"],
    "content": ["content", "contentment", "satisfied", "pleased", "peaceful", "fulfilled"],
    "disappointed": ["disappointed", "disappointment", "letdown", "discouraged", "disheartened"],
    "excited": ["excited", "excitement", "enthusiastic", "thrilled", "eager", "animated"],
    "faithful": ["faithful", "loyal", "devoted", "dedicated", "steadfast", "fidelity"],
    "frustrated": ["frustrated", "frustration", "exasperated", "aggravated", "irked"],
    "grateful": ["grateful", "gratitude", "thankful", "appreciative", "obliged"],
    "guilty": ["guilty", "guilt", "remorseful", "ashamed", "contrite", "regretful"],
    "hopeful": ["hopeful", "hope", "optimistic", "encouraged", "buoyant", "expectant","anticipating"],
    "joyful": ["joyful", "joy", "happy", "delighted", "cheerful", "gleeful"],
    "lonely": ["lonely", "loneliness", "lonesome", "isolated", "forlorn", "desolate"],
    "loneliness": ["lonely", "loneliness", "lonesome", "isolated", "forlorn", "desolate"],    
    "nostalgic": ["nostalgic", "nostalgia", "sentimental", "wistful", "yearning"],
    "overwhelmed": ["overwhelmed", "overwhelm", "swamped", "burdened", "overloaded"],
    "sad": ["sad", "sadness", "sorrowful", "melancholy", "grief", "depressed"],
    "sadness": ["sad", "sadness", "sorrowful", "melancholy", "grief", "depressed"],    
    "surprised": ["surprised", "surprise", "amazed", "astonished", "shocked", "stunned"],
    "terrified": ["terrified", "terror", "frightened", "afraid", "scared", "petrified"],
    "trusting": ["trusting", "trust", "confident", "relying", "believing", "assured","prepared"],
    "worried": ["worried", "worry", "anxious", "nervous", "concerned", "uneasy","apprehensive"],
    "anxious": ["anxious", "anxiety", "worried", "nervous", "concerned", "uneasy"]}
    
    # 处理包含<think>标签的情况
    if "<think>" in response_text and "</think>" in response_text:
        # 提取<think>标签中的内容
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            think_content = response_text[think_start+7:think_end].lower()
            
            # 在<think>内容中查找"the emotion of the speaker is"模式
            emotion_pattern = r"the emotion of the speaker is\s+(\w+)"
            match = re.search(emotion_pattern, think_content)
            if match:
                extracted_emotion = match.group(1).strip()
                # 标准化情感标签
                if emotion_mapping and extracted_emotion in emotion_mapping:
                    return emotion_mapping[extracted_emotion]
                return extracted_emotion
    
    # 如果没有<think>标签或没有找到情感，则在整个文本中搜索情感关键词
    response_lower = response_text.lower()
    
    # 查找情感关键词
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in response_lower:
                # 如果有映射字典，进行标准化
                if emotion_mapping and emotion in emotion_mapping:
                    return emotion_mapping[emotion]
                return emotion
    
    # 如果没有找到明确的情感词，返回unknown
    return "unknown"

def calculate_emotion_accuracy(name2gt: Dict[str, str], 
                             name2pred: Dict[str, str]) -> float:
    """
    计算情感准确率 (Emotion Accuracy)
    
    参数:
    - name2gt: 样本名称到真实情感标签的映射
    - name2pred: 样本名称到预测情感标签的映射
    
    返回:
    - 情感准确率 (0-100之间的浮点数)
    """
    if not name2gt or not name2pred:
        return 0.00
    
    correct_emotions = 0
    total_emotions = 0
    
    for name in name2gt:
        if name in name2pred:
            total_emotions += 1
            if name2gt[name] == name2pred[name]:
                correct_emotions += 1
    
    if total_emotions == 0:
        return 0.00
    
    return (correct_emotions / total_emotions) * 100

def evaluate_avamerg_metrics(name2gt: Dict[str, str], 
                           name2response: Dict[str, str],
                           emotion_mapping: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """
    计算AvaMERG数据集的完整评估指标
    
    参数:
    - name2gt: 样本名称到真实情感标签的映射
    - name2response: 样本名称到生成回复的映射
    - emotion_mapping: 情感映射字典
    
    返回:
    - 包含Acc、Hitrate、Dist-1、Dist-2的评估结果字典
    """
    # 1. 从生成的回复中提取情感标签,构造模型预测输出
    name2pred = {}
    for name, response in name2response.items():
        pred_emotion = extract_emotion_from_response(response, emotion_mapping)
        name2pred[name] = pred_emotion
    
    # 2. 计算情感准确率 和命中率
    emotion_acc = calculate_emotion_accuracy(name2gt, name2pred)
    emotion_hitrate = calculate_hitrate(name2gt, name2pred)
    
    # 3. 收集所有生成的回复文本
    all_responses = list(name2response.values())
    
    # 4. 计算Dist-1和Dist-2指标
    # 预处理文本，移除<think>标签内容
    cleaned_responses = []
    for text in all_responses:
        if isinstance(text, str):
            # 移除<think>...</think>标签及其内容
            cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            cleaned_text = cleaned_text.strip()
            # print(cleaned_text)
            if cleaned_text:  # 只添加非空文本
                cleaned_responses.append(cleaned_text)
        else:
            cleaned_responses.append(text)

    dist1 = evaluate_dist_n(cleaned_responses, 1) * 100
    dist2 = evaluate_dist_n(cleaned_responses, 2) * 100
    
    # 5. 组合所有评估结果
    results = {
        "Emotion_Accuracy": round(emotion_acc, 2),
        "Hitrate": round(emotion_hitrate, 2),
        "Dist-1": round(dist1, 2),
        "Dist-2": round(dist2, 2)
    }
    
    return results

def calculate_intent_accuracy(name2gt: Dict[str, str], 
                             name2pred: Dict[str, str]) -> float:
    """
    计算意图准确率 (Intent Accuracy)
    
    参数:
    - name2gt: 样本名称到真实意图标签的映射
    - name2pred: 样本名称到预测意图标签的映射
    
    返回:
    - 意图准确率 (0-100之间的浮点数)
    """
    if not name2gt or not name2pred:
        return 0.00
    
    # 收集所有有效的预测结果
    y_true = []
    y_pred = []
    
    for name in name2gt:
        if name in name2pred:
            y_true.append(name2gt[name])
            y_pred.append(name2pred[name])
    
    if len(y_true) == 0:
        return 0.00
    
    # 使用sklearn计算准确率
    accuracy = accuracy_score(y_true, y_pred) * 100
    return accuracy


def calculate_intent_weighted_metrics(name2gt: Dict[str, str], 
                                     name2pred: Dict[str, str]) -> Dict[str, float]:
    """
    计算意图识别的加权F1和加权精确率
    
    参数:
    - name2gt: 样本名称到真实意图标签的映射
    - name2pred: 样本名称到预测意图标签的映射
    
    返回:
    - 包含加权F1和加权精确率的字典
    """
    if not name2gt or not name2pred:
        return {"weighted_f1": 0.00, "weighted_precision": 0.00}
    
    # 收集所有有效的预测结果
    y_true = []
    y_pred = []
    
    for name in name2gt:
        if name in name2pred:
            y_true.append(name2gt[name])
            y_pred.append(name2pred[name])
    
    if len(y_true) == 0:
        return {"weighted_f1": 0.00, "weighted_precision": 0.00}
    
    # 计算加权F1和加权精确率
    weighted_f1 = f1_score(y_true, y_pred, average='weighted') * 100
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    return {
        "weighted_f1": round(weighted_f1, 2),
        "weighted_precision": round(weighted_precision, 2)
    }


def evaluate_mintrec_metrics(name2gt: Dict[str, str], 
                            name2response: Dict[str, str],
                            intent_mapping: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """
    计算MIntRec数据集的完整评估指标
    
    参数:
    - name2gt: 样本名称到真实意图标签的映射
    - name2response: 样本名称到模型预测意图标签的映射（直接输出意图标签，如'complain', 'praise'等）
    - intent_mapping: 意图映射字典（可选，用于标准化标签）
    
    返回:
    - 包含ACC、Weighted F1、Weighted Precision的评估结果字典
    """
    name2pred = name2response.copy()
    
    # 2. 如果提供了映射字典，应用映射
    if intent_mapping:
        for name in name2pred:
            if name2pred[name] in intent_mapping:
                name2pred[name] = intent_mapping[name2pred[name]]
    
    # 3. 计算意图准确率
    intent_acc = calculate_intent_accuracy(name2gt, name2pred)
    
    # 4. 计算加权F1和加权精确率
    weighted_metrics = calculate_intent_weighted_metrics(name2gt, name2pred)
    
    # 5. 组合所有评估结果
    results = {
        "Intent_Accuracy": round(intent_acc, 2),
        "Weighted_F1": weighted_metrics["weighted_f1"],
        "Weighted_Precision": weighted_metrics["weighted_precision"]
    }
    
    return results


# 示例用法
if __name__ == "__main__":  
    
    # 测试数据（模拟你的实际数据格式）
    test_name2gt = {
        "dia03997utt0_60": "anxious",
        "dia03997utt2_60": "anxious", 
        "dia03999utt0_60": "sad",
        "dia03999utt2_60": "sad"
    }
    
    test_name2response = {
        "dia03997utt0_60": '<think>Firstly, the event scenario of this conversation is: discussing fatigue\nSecondly, the emotion of the speaker is anxious\nThirdly, the emotion cause is feeling of declining health\nFourthly, the goal to response is to express concern and validate feelings</think>\n That sounds really tough. It\'s understandable to feel concerned about your energy levels as you age.',
        "dia03997utt2_60": "<think>Firstly, the event scenario of this conversation is: Expressing concern about losing independence due to fatigue.\nSecondly, the emotion of the speaker is anxious\nThirdly, the emotion cause is Fear of losing autonomy.\nFourthly, the goal to response is To validate feelings and encourage sharing more.</think>\n It's completely valid to feel that way. Independence is so important, and it's tough when it feels out of reach.",
        "dia03999utt0_60": "<think>Firstly, the event scenario of this conversation is: Adjusting to living alone\nSecondly, the emotion of the speaker is furious\nThirdly, the emotion cause is Loss of companionship\nFourthly, the goal to response is To express understanding of the speaker's feelings</think>\n That sounds incredibly tough. Losing a partner can leave such a big void in your life.",
        "dia03999utt2_60": "<think>Firstly, the event scenario of this conversation is: Reflecting on past memories\nSecondly, the emotion of the speaker is nostalgic\nThirdly, the emotion cause is Nostalgia for happier times\nFourthly, the goal to response is Encourage sharing positive memories</think>\n It's natural to long for those moments. Have you thought about what you'd like to remember about him?"
    }
    
    # 计算修正后的评估指标
    test_results = evaluate_avamerg_metrics(test_name2gt, test_name2response)
    print("AvaMERG评估结果:", test_results)
    
    # 测试MIntRec意图识别评估
    print("\n=== MIntRec意图识别评估测试 ===")
    
    # MIntRec意图识别评估测试
    mintrec_name2gt = {
        "sample1": "complain",
        "sample2": "praise", 
        "sample3": "thank",
        "sample4": "greet"
    }

    # 模型直接输出意图标签的情况
    mintrec_name2pred = {
        "sample1": "complain",
        "sample2": "praise",
        "sample3": "thank",
        "sample4": "greet"
    }

    mintrec_results = evaluate_mintrec_metrics(mintrec_name2gt, mintrec_name2pred)
    print("MIntRec评估结果:", mintrec_results)
    
    # 测试部分错误的情况
    print("\n=== MIntRec部分错误测试 ===")
    mintrec_name2pred_partial = {
        "sample1": "complain",  # 正确
        "sample2": "criticize", # 错误
        "sample3": "thank",     # 正确
        "sample4": "leave"      # 错误
    }
    
    mintrec_results_partial = evaluate_mintrec_metrics(mintrec_name2gt, mintrec_name2pred_partial)
    print("MIntRec部分错误评估结果:", mintrec_results_partial)