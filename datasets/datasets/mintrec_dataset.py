import os
import json
import random
from PIL import Image
from decord import VideoReader
from nano_emox.datasets.datasets.base_dataset import BaseDataset
import config

class MIntRec_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, 
                 dataset_cfg=None, model_cfg=None):
        """
        MIntRec训练数据集，支持多模态意图识别任务
        """
        self.dataset = 'MIntRec'
        
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data)

        self.candidate_labels = "complain, praise, apologise, thank, criticize, agree, taunt, flaunt, joke, oppose, comfort, care, inform, advise, arrange, introduce, leave, prevent, greet, ask for help"

        ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train_MIntRec_ann.json')
        name2label = {}
        name2subtitle = {}
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'messages' in item and 'videos' in item:
                # 提取用户消息和助手回复
                user_message = item['messages'][0]['content']
                assistant_message = item['messages'][1]['content']
                video_path = item['videos'][0]
                # name
                video_name = os.path.basename(video_path).replace('.mp4', '')
                name2label[video_name] = assistant_message
                name2subtitle[video_name] = user_message
        self.name2label = name2label
        
        self.annotation = []
        for name in name2label:
            self.annotation.append({
                'name': name,
                'subtitle': name2subtitle[name],
                'intent':name2label[name]
                })

        self.label_type_candidates = ['intent_recognition']

        # 设置数据路径
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= None

        # 使用基类初始化
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
        if self.vis_root is not None:
            return os.path.join(config.PATH_TO_RAW_VIDEO[self.dataset], sample['name'] + '.mp4')
        else:
            return None
        
    def _get_audio_path(self, sample):
        if self.wav_root is not None:
            return os.path.join(config.PATH_TO_RAW_AUDIO[self.dataset], sample['name'] + '.wav')
        else:
            return None

# inference methods
    def _get_testvideo_path(self,sample):
        """返回每轮speaker的视频文件路径"""
        video_filename = f"{sample['name']}.mp4"
        full_video_fp = os.path.join("/home/lab00/hjh/dataset/MIntRec/video", video_filename)
        return full_video_fp

    def _get_testaudio_path(self,sample):
        """返回每轮speaker的音频文件路径"""
        audio_filename = f"{sample['name']}.wav"
        full_audio_fp = os.path.join("/home/lab00/hjh/dataset/MIntRec/audio", audio_filename)
        return full_audio_fp

    def read_test_names(self):
        """读取测试集样本名称列表, 格式: dia{conversation_id}utt{turn_id}_{speaker_id}"""
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'test_MIntRec_ann.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)
        
        test_names = []
        for item in test_annotations:
            # 提取用户消息和助手回复
            video_path = item['videos'][0]
            video_name = os.path.basename(video_path).replace('.mp4', '')
            test_names.append(video_name)
        return test_names
    
    def get_test_name2gt(self):
        """获取测试集样本名称到真实标签的映射"""
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'test_MIntRec_ann.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)
        
        name2gt = {}
        for item in test_annotations:
            if 'messages' in item and 'videos' in item:
                assistant_message = item['messages'][1]['content']
                video_path = item['videos'][0]
                # name
                video_name = os.path.basename(video_path).replace('.mp4', '')
                name2gt[video_name] = assistant_message
        return name2gt

    @property
    def name2subtitle(self):
        """获取样本名称到对话记录的映射
        """
        name2subtitle = {}
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'test_MIntRec_ann.json')
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'messages' in item and 'videos' in item:
                # 提取用户消息和助手回复
                user_message = item['messages'][0]['content']
                video_path = item['videos'][0]
                video_name = os.path.basename(video_path).replace('.mp4', '')
                name2subtitle[video_name] = user_message

        return name2subtitle    