import os
import tqdm
import copy
import json
import random
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import decord
from decord import VideoReader

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from nano_emox.processors import transforms_video, AlproVideoTrainProcessor
from nano_emox.conversation.conversation_video import Conversation,SeparatorStyle
from nano_emox.datasets.datasets.base_dataset import BaseDataset
from nano_emox.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from nano_emox.models.ImageBind.data import load_audio, transform_audio # 将上面的函数功能拆解为两个

from nano_emox import config
emos = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo

# 要让模型同时支持audio, video, text三部分输入信息才行
class MELD_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                       dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'MELD'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# 直接手动指定所有信息的存储路径 #################
        ## read train/test splits
        label_path = config.PATH_TO_LABEL[self.dataset]
        corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        train_names = [name                for name in corpus]
        train_emos  = [idx2emo[corpus[name]['emo']] for name in corpus]

        name2subtitle = {}
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = row['name']
            subtitle = row['english']
            if pd.isna(subtitle): subtitle=""
            name2subtitle[name] = subtitle
        self.name2subtitle = name2subtitle
        
        self.annotation = []
        for ii, name in enumerate(train_names): # only for training data
            emotion = train_emos[ii]
            subtitle = name2subtitle[name]
            self.annotation.append({'name': name, 
                                    'subtitle': subtitle, 
                                    'onehot': emotion})
        self.label_type_candidates = ['onehot_w_candidates', 'onehot_wo_candidates']
        
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= config.PATH_TO_RAW_FACE[self.dataset]

        self.candidate_labels = ",".join(set(train_emos))
        # print (f'candidate labels: {self.candidate_labels}')
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
    
    # 可能存在两种类型的数据格式
    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
        return full_video_fp
    
    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp

    def _get_face_path(self, sample):
        full_face_fp = os.path.join(self.face_root, sample['name'] + '.npy')
        return full_face_fp

    # 这部分是测试，假如 inference 的时候只支持 image，那就随机选择一帧作为输入
    def _get_image_path(self, sample):
        full_video_fp = self._get_video_path(sample)
        vr = VideoReader(uri=full_video_fp, height=224, width=224)

        # get index
        vlen = len(vr)
        index = random.randint(0, vlen-1)
        frame = vr[index]

        # read PIL.Image
        image = Image.fromarray(frame.numpy())
     
        return image

    # for inference
    def read_test_names(self):
        label_path  = config.PATH_TO_LABEL[self.dataset]
        test_corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        test_names  = [name for name in test_corpus]
        assert len(test_names) == 2610
        return test_names
    
    def get_test_name2gt(self):
        label_path  = config.PATH_TO_LABEL[self.dataset]
        corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        name2gt = {}
        for name in corpus:
            gt = corpus[name]['emo']
            name2gt[name] = gt
        return name2gt
    
    def get_emo2idx_idx2emo(self):
        return emo2idx, idx2emo