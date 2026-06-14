import os
import sys
import re
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
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import re

from transformers.models.rt_detr.image_processing_rt_detr import max_across_indices

# Ensure the parent directory is in sys.path so 'nano_emox' package can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_emox import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
import numpy as np
from nano_emox.utils.read_files import *
from nano_emox.utils.qwen import *
from nano_emox.utils.functions import *
from nano_emox.evaluation.wheel import func_get_name2reason
from nano_emox.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls
from nano_emox.evaluation.ew_metric import *
from nano_emox.evaluation.wheel import *
from nano_emox.evaluation.eval_merg_exp import evaluate_avamerg_metrics, evaluate_mintrec_metrics

def search_for_result_root(input_dir, inter_print=True):
    candidates = glob.glob(input_dir + '*')
    root_candidates = [root for root in candidates if os.path.isdir(root)]
    print(f"root_candidates is:{root_candidates}" )
    if len(root_candidates) == 0:
        if inter_print: print ('No file exists!')
        return ''
    
    
    # 找到最新的评估结果root
    maxcount = 0
    maxtimestampe = 0
    targetroot = ''
    for root in root_candidates:
        match = re.search(r'(\d{11})$', root)
        timestamp = int(match.group(1)) if match else 0
        if timestamp > maxtimestampe:
            maxtimestampe = timestamp
            targetroot = root
    # 找到 files 最多的 root
    # for root in root_candidates:
    #     store_path = []
    #     for path in os.listdir(root):
    #         if path.startswith('checkpoint_') and path.find('-') == -1:
    #             store_path.append(path)
    #     count = len(store_path)
    #     if inter_print: print (root, '==>', count)
    #     if count > maxcount:
    #         maxcount = count
    #         targetroot = root
    
    if inter_print: print ('================================================')
    if inter_print: print ('Targetroot: ', targetroot)
    if inter_print: print ('Saved result files ', maxcount)
    # report last file info
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    if inter_print: print("Last result file creation time:", datetime.datetime.fromtimestamp(creation_time))
    if inter_print: print ('================================================')
    return targetroot


def func_read_datasetname(input_dir):
    # print (input_dir)
    supprot_datasets = list(config.DATA_DIR.keys())
    assert input_dir.find('/results-') != -1
    dataset = input_dir.split('/results-')[1].split('/')[0]
    for supprot_item in supprot_datasets:
        if supprot_item.lower() == dataset.lower():
            return supprot_item
    ValueError(f'cannot find suitable dataset for {input_dir}')


def get_dataset2cls(dataset):
    if dataset == 'MER2023':     return MER2023_Dataset()
    if dataset == 'MER2024':     return MER2024_Dataset()
    if dataset == 'MELD':        return MELD_Dataset()
    if dataset == 'IEMOCAPFour': return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':     return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':    return CMUMOSEI_Dataset()
    if dataset == 'SIMS':        return SIMS_Dataset()
    if dataset == 'SIMSv2':      return SIMSv2_Dataset()
    if dataset == 'MER2025OV':   return MER2025OV_Dataset()
    if dataset == 'AvaMERG':     return AvaMERG_Dataset()
    if dataset == 'OVMERD':      return OVMERD_Dataset()
    if dataset == 'MIntRec':      return MIntRec_Dataset()
    if dataset == 'MIntRec2':     return MIntRec2_Dataset()
    print ('dataset cls not provided!')
    return None


def get_discrete_or_dimension_flag(dataset):
    if dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
        return 'discrete'
    elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
        return 'dimension'
    elif dataset in ['MER2025OV', 'OVMERD']:
        return 'ovlabel'
    elif dataset in ['AvaMERG']:
        return 'avamerg'
    elif dataset in ['MIntRec', 'MIntRec2']:
        return 'mintrec'
    else:
        ValueError('unsupported dataset input')


def get_emo2idx_idx2emo(dataset_cls):
    emo2idx, idx2emo = {}, {}

    if hasattr(dataset_cls, 'get_emo2idx_idx2emo'): 
        emo2idx, idx2emo = dataset_cls.get_emo2idx_idx2emo()
        # post process [不同数据集的标签表示有些许差异，进行统一化处理]
        if 'happy' in emo2idx: emo2idx['joy']   = emo2idx['happy']
        if 'anger' in emo2idx: emo2idx['angry'] = emo2idx['anger']
        if 'sad'   in emo2idx: emo2idx['sadness'] = emo2idx['sad']
        if 'joy'   in emo2idx: emo2idx['happy'] = emo2idx['joy']
        if 'angry' in emo2idx: emo2idx['anger'] = emo2idx['angry']
    return emo2idx, idx2emo

def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


## similarity score for: openset <-> discrete
def calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    # epoch_root=(name2reason) => openset
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    # 计算 hitrate, mscore
    hitrate, mscore = hitrate_metric_calculation(name2gt=name2gt, openset_npz=openset_npz, inter_print=inter_print)
    return hitrate, mscore


def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    
    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
        
    # 计算 EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    fscore, precision, recall = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return fscore, precision, recall


## similarity score for: openset -> sentiment <-> sentiment
def calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):

    # 1. 抽取 openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    
    # 2. 将 openset 转成 [positive, negative, neutral]
    sentiment_npz = openset_npz[:-4]+'-sentiment.npz'
    if not os.path.exists(sentiment_npz):
        openset_to_sentiment_batchcalling(openset_npz=openset_npz, store_npz=sentiment_npz,
                                          llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # 3. 计算 scores
    ## 3.0 openset 自然语言形式标签 -> float
    name2pred = {}
    filenames = np.load(sentiment_npz, allow_pickle=True)['filenames']
    fileitems = np.load(sentiment_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        if item == 'positive':
            name2pred[name] = 1
        elif item == 'negative':
            name2pred[name] = -1
        elif item == 'neutral':
            name2pred[name] = 0
        else: # 其他无法操作的标签
            if inter_print: print ('error sample:', name, item)
            name2pred[name] = 0
    ## 3.1 conversion
    val_labels, val_preds = [], []
    for name in name2gt:
        val_labels.append(name2gt[name])
        val_preds.append(name2pred[name])
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    ## 3.2 metric calculation (name2gt, name2pred) -> scores
    non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
    accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
    fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')
    return fscore, accuracy


def calculate_avamerg_metrics(epoch_root, name2gt, inter_print=True):
    """
    计算AvaMERG数据集的评估指标：情感准确率(Acc)、Dist-1、Dist-2
    
    参数:
    - epoch_root: 包含推理结果的npz文件路径
    - name2gt: 样本名称到真实情感标签的映射
    - inter_print: 是否打印中间结果
    
    返回:
    - emotion_acc: 情感准确率
    - dist1: Dist-1指标
    - dist2: Dist-2指标
    """
    # 读取推理结果
    name2reason = np.load(epoch_root, allow_pickle=True)['name2reason'].item()
    
    # 使用eval_merg_exp中的评估函数
    results = evaluate_avamerg_metrics(name2gt, name2reason)
    
    emotion_acc = results["Emotion_Accuracy"]
    hitrate = results["Hitrate"]
    dist1 = results["Dist-1"]
    dist2 = results["Dist-2"]
    
    if inter_print:
        print(f'Emotion Accuracy: {emotion_acc:.4f}')
        print(f'Hitrate: {hitrate:.4f}')
        print(f'Dist-1: {dist1:.4f}')
        print(f'Dist-2: {dist2:.4f}')
    
    return emotion_acc, hitrate, dist1, dist2


def calculate_mintrec_metrics(epoch_root, name2gt, inter_print=True):
    """
    计算MIntRec数据集的评估指标：意图准确率(ACC)、加权F1(Weighted F1)、加权精确率(Weighted Precision)
    
    参数:
    - epoch_root: 包含推理结果的npz文件路径
    - name2gt: 样本名称到真实意图标签的映射
    - inter_print: 是否打印中间结果
    
    返回:
    - intent_acc: 意图准确率
    - weighted_f1: 加权F1分数
    - weighted_precision: 加权精确率
    """
    # 读取推理结果
    name2reason = np.load(epoch_root, allow_pickle=True)['name2reason'].item()
    
    # 使用eval_merg_exp中的评估函数
    results = evaluate_mintrec_metrics(name2gt, name2reason)
    
    intent_acc = results["Intent_Accuracy"]
    weighted_f1 = results["Weighted_F1"]
    weighted_precision = results["Weighted_Precision"]
    
    if inter_print:
        print(f'Intent Accuracy: {intent_acc:.4f}')
        print(f'Weighted F1: {weighted_f1:.4f}')
        print(f'Weighted Precision: {weighted_precision:.4f}')
    
    return intent_acc, weighted_f1, weighted_precision


def main_zeroshot_scores(input_dir, debug=False, test_epochs='', inter_print=True):

    # ## 如果 input_dir 不存在的话，那么需要去检索最匹配的路径
    if not os.path.exists(input_dir):
        input_dir = search_for_result_root(input_dir, inter_print)
    if inter_print: print (f'process root: {input_dir}')

    # read dataset infos
    dataset = func_read_datasetname(input_dir)
    disordim_flag = get_discrete_or_dimension_flag(dataset)
    if inter_print: print (f'process dataset: {dataset} => {disordim_flag}')
    dataset_cls = get_dataset2cls(dataset)
    if dataset_cls is None:
        raise ValueError(f'Dataset class not found for {dataset}')
    name2gt = dataset_cls.get_test_name2gt()
    if inter_print: print (f'target sample number: {len(name2gt)}')

    # discrete: 自然语言形式标签；dimension: float score
    if disordim_flag == 'discrete':
        _, idx2emo = get_emo2idx_idx2emo(dataset_cls)
        # => update (name2gt)
        for name in name2gt:
            gt = name2gt[name]
            if not isinstance(gt, str):
                name2gt[name] = idx2emo[gt]
    # print (name2gt) # debug

    # load model
    llm, tokenizer, sampling_params = None, None, None
    if debug == False:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25_7B')
    
    # main process
    whole_score1s, whole_score2s, whole_score3s, whole_score4s = [], [], [], []
    epoch_numbers = []  # 记录每个epoch的轮次
    for epoch_root in sorted(glob.glob(input_dir + '/*.npz')):

        if epoch_root.find('openset') != -1 or epoch_root.find('sentiment') != -1:
            continue

        # =============== process for {epoch_root} ===============
        epochname = os.path.basename(epoch_root)
        if inter_print: print (epochname)
        # 0. 判断 epoch 是不是在 test_epochs 内，否是就跳过这部分
        if test_epochs != '':
            run_epochs = [int(item) for item in test_epochs.split(',')]
            cur_epoch = int(epochname.split('_')[1])
            if cur_epoch not in run_epochs: continue
        
        # 提取轮次信息
        cur_epoch = int(epochname.split('_')[1])
        epoch_numbers.append(cur_epoch)

        # 1. score calculation
        if disordim_flag == 'discrete':
            hitrate, _ = calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'hitrate: {hitrate}')
            whole_score1s.append(hitrate)
            whole_score2s.append(0)
            whole_score3s.append(0)
            
        elif disordim_flag == 'dimension':
            fscore, acc = calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, acc: {acc}')
            whole_score1s.append(fscore)
            whole_score2s.append(acc)
            whole_score3s.append(0)
        
        elif disordim_flag == 'ovlabel':
            fscore, precision, recall = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, precision: {precision}, recall: {recall}')
            whole_score1s.append(fscore)
            whole_score2s.append(precision)
            whole_score3s.append(recall)
            
        elif disordim_flag == 'avamerg':
            emotion_acc, hitrate, dist1, dist2 = calculate_avamerg_metrics(epoch_root, name2gt, inter_print)
            if inter_print: print(f'emotion_acc: {emotion_acc}, hitrate: {hitrate}, dist1: {dist1}, dist2: {dist2}')
            whole_score1s.append(emotion_acc)
            whole_score2s.append(hitrate)
            whole_score3s.append(dist1)
            whole_score4s.append(dist2)
            
        elif disordim_flag == 'mintrec':
            intent_acc, weighted_f1, weighted_precision = calculate_mintrec_metrics(epoch_root, name2gt, inter_print)
            if inter_print: print(f'intent_acc: {intent_acc}, weighted_f1: {weighted_f1}, weighted_precision: {weighted_precision}')
            whole_score1s.append(intent_acc)
            whole_score2s.append(weighted_f1)
            whole_score3s.append(weighted_precision)

        if inter_print: print ('=========================')

    # whole_score1s => main metric
    best_index = np.argmax(whole_score1s)
    best_score1 = whole_score1s[best_index]
    best_score2 = whole_score2s[best_index]
    best_score3 = whole_score3s[best_index]
    best_score4 = whole_score4s[best_index]
    best_epoch = epoch_numbers[best_index]  # 获取最佳轮次
    
    if disordim_flag == 'discrete':
        if inter_print: print (f'{dataset}: best hitrate: %.4f; best mscore: %.4f (epoch: {best_epoch})' %(best_score1, best_score2))
    elif disordim_flag == 'dimension':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best acc: %.4f (epoch: {best_epoch})' %(best_score1, best_score2))
    elif disordim_flag == 'ovlabel':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best precision: %.4f; best recall: %.4f (epoch: {best_epoch})' %(best_score1, best_score2,best_score3))
    elif disordim_flag == 'avamerg':
        if inter_print: print (f'{dataset}: best emotion_acc: %.4f; best hitrate: %.4f; best dist1: %.4f; best dist2: %.4f (epoch: {best_epoch})' %(best_score1, best_score2, best_score3, best_score4))
    elif disordim_flag == 'mintrec':
        if inter_print: print (f'{dataset}: best intent_acc: %.4f; best weighted_f1: %.4f; best weighted_precision: %.4f (epoch: {best_epoch})' %(best_score1, best_score2, best_score3))
    # return the best scores and best epoch
    return best_score1, best_score2, best_score3, best_score4, best_epoch



def func_return_scores_one(modelname=None, dataset_candidates='merunibench'):
    ## => (process datasets)
    if dataset_candidates=='merunibench':
        process_datasets = ["mer2023", "mer2024", "meld", "iemocapfour", "cmumosi", "cmumosei", "sims", "simsv2","ovmerd"]
    elif dataset_candidates=='mer2025ov':
        process_datasets = ['mer2025ov']
    elif dataset_candidates=='avamerg':
        process_datasets = ['avamerg']
    elif dataset_candidates=='intent':
        process_datasets = ['mintrec','mintrec2']

    print_per_dataset, avg_score = [], []
    best_epochs = []  # 记录每个数据集的最佳轮次
    for dataset in process_datasets:
        process_root = f"output/results-{dataset}/{modelname}"
        #process_root = f"output/results-{dataset}/{modelname}/'emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250619185'"
        ## 计算指标
        score1, score2, score3, best_epoch = main_zeroshot_scores(process_root, debug=False, test_epochs='', inter_print=True)
        print_per_dataset.extend([score1])
        best_epochs.append(best_epoch)
        avg_score.append(score1)
    # append a avg value for ranking
    avg_score = np.mean(avg_score)
    print_per_dataset.append(avg_score)
    best_epochs.append(-1)  # 平均值没有对应的epoch
    # 格式化输出，包含轮次信息
    formatted_results = []
    for i, (score, epoch) in enumerate(zip(print_per_dataset, best_epochs)):
        if epoch == -1:  # 平均值
            formatted_results.append("| %.2f" % (score * 100))
        else:
            formatted_results.append("| %.2f(epoch %d)" % (score * 100, epoch))
    return formatted_results, avg_score


if __name__ == "__main__":
    """
    输出模型在每个数据集的评分结果\平均得分\评估模型准确度，选取最好的结果打印输出
    根据modelname选取评估的模型。
    模型候选 :
                nano_emox baseline
                "emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz"
                不分阶段训练
                "nano_emox_mercaption_outputhybird"  
                1阶段训练 对齐预训练 alignment pretraining
                "nano_emox_mercaption_outputhybird_phase1"  
                2阶段训练 多任务微调 multitask fine-tuning
                "nano_emox_mercaption_outputhybird_phase2"  
    数据候选 :
                "merunibench"
                "mer2025ov"
                "avamerg"
                "intent"
    """
    for modelname in [
                    "nano_emox_mercaption_outputhybird"
                    ]:
        print_per_dataset, avg_score = func_return_scores_one(modelname=modelname,dataset_candidates='merunibench')
        print (modelname, " ".join(print_per_dataset))
