'''
This is use for performance evaluation using EW-based metrics.
The main code is draw from .//desktop//main_nano_emox.py
'''

import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
import glob
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from toolkit.utils.read_files import *
from toolkit.utils.qwen import *
from toolkit.utils.functions import *
from nano_emox.evaluation.wheel import *


# 单次加载完模型，避免后续重复调用
def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params

## reason -> ov labels
def extract_openset_batchcalling(reason_root=None, reason_npz=None, update_npz=None, reason_csv=None, name2reason=None,
                                 store_root=None, store_npz=None, 
                                 modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if reason_root is not None:
        name2reason = func_get_name2reason(reason_root)
    elif reason_npz is not None:
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()
    elif reason_csv is not None:
        names = func_read_key_from_csv(reason_csv, 'name')
        reasons = func_read_key_from_csv(reason_csv, 'reason')
        name2reason = {}
        for (name, item) in zip(names, reasons):
            name2reason[name] = item
    elif update_npz is not None:
        name2reason = {}
        filenames = np.load(update_npz, allow_pickle=True)['filenames']
        fileitems = np.load(update_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2reason[name] = item

    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_openset_qwen(llm=llm, tokenizer=tokenizer,
                                                 sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    if store_root is not None:
        if not os.path.exists(store_root):
            os.makedirs(store_root)
        # save to folder
        for (name, response) in zip(whole_names, whole_responses):
            save_path = os.path.join(store_root, f'{name}.npy')
            np.save(save_path, response)
    elif store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses


## ov labels -> sentiment
def openset_to_sentiment_batchcalling(openset_npz=None, name2openset=None, 
                                      store_npz=None,
                                      modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if openset_npz is not None:
        name2openset = {}
        filenames = np.load(openset_npz, allow_pickle=True)['filenames']
        fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2openset[name] = item

    ## main process
    whole_names, whole_responses = list(name2openset.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2openset[name] for name in batch_names]
        batch_responses = openset_to_sentiment_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling_params, 
                                                    batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage [可以允许不存储结果，直接返回结果]
    if store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses


format_mapping = read_format2raws()          # level3 -> level2
raw_mapping = read_candidate_synonym_merge() # level2 -> level1
# 功能：input [gt, openset]; output: 12 个 EW-based metric 下的平均结果
def wheel_metric_calculation(gt_root=None, gt_csv=None, name2gt=None, 
                             openset_root=None, openset_npz=None, name2pred=None, 
                             process_names=None, inter_print=True, level='level1'):

    # 已 M-avg 为主指标
    # candidate_metrics = [
    #                     'case1', 'case2',
    #                     'case3_wheel1_level1', 'case3_wheel1_level2',
    #                     'case3_wheel2_level1', 'case3_wheel2_level2',
    #                     'case3_wheel3_level1', 'case3_wheel3_level2',
    #                     'case3_wheel4_level1', 'case3_wheel4_level2',
    #                     'case3_wheel5_level1', 'case3_wheel5_level2',
    #                     ]
    if level == 'level1':
        candidate_metrics = [
                            'case3_wheel1_level1',
                            'case3_wheel2_level1',
                            'case3_wheel3_level1',
                            'case3_wheel4_level1',
                            'case3_wheel5_level1',
                            ]
    elif level == 'level2':
        candidate_metrics = [
                            'case3_wheel1_level2',
                            'case3_wheel2_level2',
                            'case3_wheel3_level2',
                            'case3_wheel4_level2',
                            'case3_wheel5_level2',
                            ]

    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        precision, recall = calculate_openset_overlap_rate(gt_root=gt_root,
                                                           gt_csv=gt_csv,
                                                           name2gt=name2gt,
                                                           openset_root=openset_root, 
                                                           openset_npz=openset_npz, 
                                                           name2pred=name2pred,
                                                           process_names=process_names, 
                                                           metric=metric,
                                                           format_mapping=format_mapping,
                                                           raw_mapping=raw_mapping,
                                                           inter_print=inter_print)
        fscore = 2 * (precision * recall) / (precision + recall)
        whole_scores.append([fscore, precision, recall])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores


def hitrate_metric_calculation(name2gt=None, openset_root=None, openset_npz=None, name2pred=None, inter_print=True):

    # 已 M-avg 为主指标 [全部映射到 level1 的 label]
    candidate_metrics = [
                        'case3_wheel1_level1',
                        'case3_wheel2_level1',
                        'case3_wheel3_level1',
                        'case3_wheel4_level1',
                        'case3_wheel5_level1',
                        ]

    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        hitrate, mscore = calculate_openset_onehot_hitrate(name2gt=name2gt,
                                                           openset_root=openset_root, openset_npz=openset_npz, name2pred=name2pred,
                                                           metric=metric, format_mapping=format_mapping, raw_mapping=raw_mapping,
                                                           inter_print=inter_print)
        whole_scores.append([hitrate, mscore])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores

def hit_or_not(gt_ov, pred_ov):
    candidate_metrics = [
                        'case3_wheel1_level1',
                        'case3_wheel2_level1',
                        'case3_wheel3_level1',
                        'case3_wheel4_level1',
                        'case3_wheel5_level1',
                        ]
    for metric in candidate_metrics:
        if func_hit_or_not(gt_ov, pred_ov, metric=metric, format_mapping=format_mapping, raw_mapping=raw_mapping):
            return True
    return False
