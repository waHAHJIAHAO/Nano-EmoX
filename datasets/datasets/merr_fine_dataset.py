import os
import tqdm
import random
import numpy as np
import pandas as pd
import json

import decord
from decord import VideoReader

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from nano_emox.processors import transforms_video, AlproVideoTrainProcessor
from nano_emox.conversation.conversation_video import Conversation,SeparatorStyle
from nano_emox.datasets.datasets.base_dataset import BaseDataset
from nano_emox.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from nano_emox.models.ImageBind.data import load_audio, transform_audio
from toolkit.utils.functions import string_to_list

import config

# MERR_fine数据集类，与MER-caption保持一致的label_type_candidate结构
class MERRFine_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                    dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'MERRFine'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# 直接手动指定所有信息的存储路径 #################
        # 读取MERR_fine_grained.json标注文件
        annotation_path = config.PATH_TO_LABEL[self.dataset]
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # 处理标注数据
        self.annotation = []
        for sample_name, sample_data in annotation_data.items():
            # 提取各个字段
            visual_prior = ", ".join(sample_data.get('visual_prior_list', []))
            audio_prior = sample_data.get('audio_prior_list', '')
            emotion = sample_data.get('pseu_emotion', '')
            subtitle = sample_data.get('text', '')
            reason_caption = sample_data.get('smp_reason_caption', '')
            
            self.annotation.append({
                'name': sample_name,
                'subtitle': subtitle,
                'description': reason_caption,  # 对应MER-caption的description字段
                'ovlabel': emotion,  # 对应MER-caption的ovlabel字段
                'visual_prior': visual_prior,
                'audio_prior': audio_prior
            })
        
        self.label_type_candidates = ['description', 'ovlabel']
        
        # 设置数据路径（需要根据实际情况调整）
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]  # 视频文件路径
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]  # 音频文件路径
        face_root = None
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         ann_path='',
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
        
    def _get_video_path(self, sample):
        # 确保vis_root存在且为字符串
        if not self.vis_root or not isinstance(self.vis_root, str):
            return sample['name'] + '.mp4'
        
        # 尝试不同的视频格式，优先检查mp4，然后检查avi
        video_formats = ['.mp4', '.avi']
        for fmt in video_formats:
            full_video_fp = os.path.join(self.vis_root, sample['name'] + fmt)
            if os.path.exists(full_video_fp):
                return full_video_fp
        
        # 如果都不存在，返回默认的mp4路径（用于错误处理）
        return os.path.join(str(self.vis_root), sample['name'] + '.mp4')

    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp
    
    
    def __len__(self):
        return len(self.annotation)
    