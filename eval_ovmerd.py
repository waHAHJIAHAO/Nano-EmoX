import re
import os
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

###########################################
# common function
###########################################
from openai import OpenAI
from typing import Dict, Any

def get_completion(prompt: str, model: str = "gpt-3.5-turbo-0125") -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f'API调用失败: {str(e)}')
        return ''

def func_gain_name2value(csv_path, value='reasons'):
    name2value = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        name = row['names']
        reason = row[value]
        if pd.isna(reason): reason=""
        name2value[name] = reason
    return name2value


###########################################
# evaluate clue overlap
###########################################
def func_clue_scoring(gt_reason, pred_reason):
    prompt = f"""
                下面给出人物的《真实描述》以及《预测描述》。请按照以下步骤计算《预测描述》的得分，得分范围为1-10。最终仅输出预测分数的数值大小，并给出原因。

                1.请总结《真实描述》中有关人物的情感状态描述

                2.请总结《预测描述》中有关人物的情感状态描述

                3.计算《预测描述》与《真实描述》之间的重叠度，重叠度越高，返回的分数越高。

                4.输出格式为: '预测分数'：预测分数；'原因'：原因、
                
                输入：

                《真实描述》：{gt_reason}

                《预测描述》：{pred_reason}

                输出：

                """
    response = get_completion(prompt)
    return response

def chatgpt_clue_scoring_main(data_root, save_path, debug=False):

    ## read ground-truth label
    gt_path = os.path.join(data_root, 'gt-eng.csv')
    name2gt = func_gain_name2value(gt_path, value='reasons')

    ## read predicted labels - 自动扫描所有CSV文件
    name2preds = {}
    # 自动扫描data_root目录下所有CSV文件（除了gt-eng.csv）
    all_csv_files = glob.glob(os.path.join(data_root, '*.csv'))
    pred_paths = [f for f in all_csv_files if not f.endswith('gt-eng.csv')]
    
    # 如果没有找到CSV文件，使用您指定的模型列表
    if not pred_paths:
        pred_paths = [
            os.path.join(data_root, 'nano_emox.csv'),
            # 在这里添加您想要评估的其他模型CSV文件
            # os.path.join(data_root, 'your_model.csv'),
        ]

    for pred_path in pred_paths:
        predname = os.path.basename(pred_path)[:-4]
        name2reason = func_gain_name2value(pred_path, value='chi_reasons')
        common_names = set(name2reason).intersection(name2gt)
        for ii, name in enumerate(common_names):
            if debug and ii == 2:
                break
            if name not in name2preds:
                name2preds[name] = []
            name2preds[name].append((predname, name2reason[name]))
    
    # save chatgpt_score
    name2score = {}
    for name in name2preds:
        print (f'====== {name} ======')
        gt = name2gt[name]
        for (predname, predreason) in tqdm.tqdm(name2preds[name]):
            score = func_clue_scoring(gt, predreason)
            print(score)
            # 提取分数，添加错误处理
            if score is not None:
                match = re.search(r"[\d][.]*[\d]*", score)
                if match:
                    start, end = match.span()
                    score = float(score[start:end])
                else:
                    print(f"Warning: Could not extract score from: {score}")
                    score = 0.0  # 默认分数
            else:
                score = 0.0
            if name not in name2score: name2score[name] = []
            print (score, predreason)
            name2score[name].append((predname, score))
    # 使用pickle保存字典数据，避免numpy类型检查问题
    import pickle
    with open(save_path.replace('.npz', '.pkl'), 'wb') as f:
        pickle.dump({'name2score': name2score}, f)


###########################################
# evaluate emotion overlap
###########################################
def get_summarized_emotions(reason):
    prompt = f"""
              请根据以下视频描述，推测视频中人物最有可能的情感状态，仅输出情感词：
              
              视频描述：他虽然看起来很高兴，但是实际上很焦虑
              
              输出结果：焦虑

              视频描述：{reason}

              输出结果：
              """
    response = get_completion(prompt)
    return response

def func_label_scoring(gt_reason, pred_reason):
    prompt = f"""
                下面给出人物的《真实情感》以及《预测情感》。请计算《预测情感》与《真实情感》之间的重叠度，重叠度越高，返回的分数越高。得分范围为1-10。最终仅输出预测分数的数值大小，并给出原因。
                
                输出格式为: '预测分数'：预测分数；'原因'：原因
                
                输入：

                《真实情感》：{gt_reason}

                《预测情感》：{pred_reason}

                输出：

                """
    response = get_completion(prompt)
    return response

def chatgpt_label_scoring_main(data_root, save_path, debug=False):

    ## read ground-truth label
    gt_path = os.path.join(data_root, 'gt-eng.csv')
    name2gt = func_gain_name2value(gt_path, value='reasons')

    ## read predicted labels - 自动扫描所有CSV文件
    name2preds = {}
    # 自动扫描data_root目录下所有CSV文件（除了gt-eng.csv）
    all_csv_files = glob.glob(os.path.join(data_root, '*.csv'))
    pred_paths = [f for f in all_csv_files if not f.endswith('gt-eng.csv')]
    
    # 如果没有找到CSV文件，使用您指定的模型列表
    if not pred_paths:
        pred_paths = [
            os.path.join(data_root, 'nano_emox.csv'),
            # 在这里添加您想要评估的其他模型CSV文件
            # os.path.join(data_root, 'your_model.csv'),
        ]

    for pred_path in pred_paths:    
        predname = os.path.basename(pred_path)[:-4]
        name2reason = func_gain_name2value(pred_path, value='chi_reasons')
        common_names = set(name2reason).intersection(name2gt)
        for ii, name in enumerate(common_names):
            if debug and ii == 2:
                break
            if name not in name2preds:
                name2preds[name] = []
            name2preds[name].append((predname, name2reason[name]))
            
    ## save chatgpt_score
    name2score = {}
    for name in name2preds:
        print (f'====== {name} ======')
        for (predname, predreason) in tqdm.tqdm(name2preds[name]):
            gt_emotion = get_summarized_emotions(name2gt[name])
            pred_emotion = get_summarized_emotions(predreason)
            score = func_label_scoring(gt_emotion, pred_emotion)
            # 提取分数，添加错误处理
            if score is not None:
                match = re.search(r"[\d][.]*[\d]*", score)
                if match:
                    start, end = match.span()
                    score = float(score[start:end])
                else:
                    print(f"Warning: Could not extract score from: {score}")
                    score = 0.0  # 默认分数
            else:
                score = 0.0
            if name not in name2score: name2score[name] = []
            print (score, predreason)
            name2score[name].append((predname, score))
    # 使用pickle保存字典数据，避免numpy类型检查问题
    import pickle
    with open(save_path.replace('.npz', '.pkl'), 'wb') as f:
        pickle.dump({'name2score': name2score}, f)


###########################################
# analyze scores for different baselines 
###########################################
# select_names: list file contains several names. Here, select_names=[] represents to process on all names
def chatgpt_scoring_analyze(score_path, select_names=[], print_flag=True):
    whole_score = {}
    # 支持both .npz and .pkl files
    if score_path.endswith('.pkl'):
        import pickle
        with open(score_path, 'rb') as f:
            data = pickle.load(f)
        name2score = data['name2score']
    else:
        name2score = np.load(score_path, allow_pickle=True)['name2score'].tolist()
    for name in name2score:
        if len(select_names)!=0 and (name not in select_names):
            continue
        for (modelname, score) in name2score[name]:
            if modelname not in whole_score:
                whole_score[modelname] = []
            whole_score[modelname].append(score)
    
    for modelname in whole_score:
        scores = whole_score[modelname]
        if print_flag:
            print (f'processed sample numbers: {len(scores)}')
        meanscore = np.mean(scores)
        print (f'{modelname} == average score: {meanscore}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str, default="output/ov-merd-eval", help='dataset path')
    parser.add_argument('--openai_key', type=str, default="sk-3nE3ZDtBePFrQkJnADdYgVibJkcp2VXYyAoj5jBRdoyD6y1P", help='your chatgpt key')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    args = parser.parse_args()
    # 初始化OpenAI客户端
    global client
    client = OpenAI(
        api_key=args.openai_key,
        base_url="https://cn.gptapi.asia/v1" #https://jeniya.top/v1
    )
    
    # for clue overlap analysis
    clue_result_path = os.path.join(args.data_root, 'clue_score.pkl')
    chatgpt_clue_scoring_main(args.data_root, clue_result_path, args.debug)
    print("\n=== 线索重叠度评估结果 ===")
    chatgpt_scoring_analyze(clue_result_path)

    ## for label overlap analysis
    emo_result_path = os.path.join(args.data_root, 'emo_score.pkl')
    chatgpt_label_scoring_main(args.data_root, emo_result_path, args.debug)
    print("\n=== 情感重叠度评估结果 ===")
    chatgpt_scoring_analyze(emo_result_path)
    
    # 综合输出两个指标的结果
    print("\n" + "="*60)
    print("综合评估结果汇总")
    print("="*60)
    
    # 获取线索重叠度分数
    clue_scores = {}
    if os.path.exists(clue_result_path):
        with open(clue_result_path, 'rb') as f:
            clue_data = pickle.load(f)
        clue_name2score = clue_data['name2score']
        for name in clue_name2score:
            for (modelname, score) in clue_name2score[name]:
                if modelname not in clue_scores:
                    clue_scores[modelname] = []
                clue_scores[modelname].append(score)
    
    # 获取情感重叠度分数
    emo_scores = {}
    if os.path.exists(emo_result_path):
        with open(emo_result_path, 'rb') as f:
            emo_data = pickle.load(f)
        emo_name2score = emo_data['name2score']
        for name in emo_name2score:
            for (modelname, score) in emo_name2score[name]:
                if modelname not in emo_scores:
                    emo_scores[modelname] = []
                emo_scores[modelname].append(score)
    
    # 收集每个模型的综合结果
    all_models = set(clue_scores.keys()) | set(emo_scores.keys())
    model_results = {}
    
    for modelname in sorted(all_models):
        model_results[modelname] = {}
        
        if modelname in clue_scores:
            clue_mean = np.mean(clue_scores[modelname])
            clue_samples = len(clue_scores[modelname])
            model_results[modelname]['clue_overlap'] = clue_mean
            model_results[modelname]['clue_samples'] = clue_samples
        else:
            model_results[modelname]['clue_overlap'] = None
            model_results[modelname]['clue_samples'] = 0
            
        if modelname in emo_scores:
            emo_mean = np.mean(emo_scores[modelname])
            emo_samples = len(emo_scores[modelname])
            model_results[modelname]['emo_overlap'] = emo_mean
            model_results[modelname]['emo_samples'] = emo_samples
        else:
            model_results[modelname]['emo_overlap'] = None
            model_results[modelname]['emo_samples'] = 0
            
        # 计算平均分（如果两个指标都有数据）
        if modelname in clue_scores and modelname in emo_scores:
            overall_mean = (clue_mean + emo_mean) / 2
            model_results[modelname]['overall_mean'] = overall_mean
        else:
            model_results[modelname]['overall_mean'] = None
    
    # 统一输出所有模型的结果
    for modelname in sorted(model_results.keys()):
        result = model_results[modelname]
        
        clue_str = f"{result['clue_overlap']:.4f}" if result['clue_overlap'] is not None else "无数据"
        emo_str = f"{result['emo_overlap']:.4f}" if result['emo_overlap'] is not None else "无数据"
        
        print(f"{modelname}: Clue Overlap: {clue_str}, Emotion Overlap: {emo_str}")
    
    print("\n" + "="*60)
    print("评估完成！")
