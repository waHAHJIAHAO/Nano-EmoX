import os
import json
import random
from PIL import Image
from decord import VideoReader
from nano_emox.datasets.datasets.base_dataset import BaseDataset
import config

class CREMA_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, 
                 dataset_cfg=None, model_cfg=None):
        """
        CREMA-D数据集预热音频分支
        """
        self.dataset = 'CREMA'
        
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data)

        self.candidate_labels = "anger, disgust, fear, happy, neutral, sad"

        description_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'crema_d_annotations.json')
        name2label = {}
        name2subtitle={}        
        with open(description_ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'messages' in item and 'audios' in item:
                # 提取用户消息和助手回复
                user_message = item['messages'][0]['content']
                assistant_message = item['messages'][1]['content']
                audio_path = item['audios'][0]

                # name
                audio_name = os.path.basename(audio_path).replace('.wav', '')
                name2label[audio_name] = assistant_message
                name2subtitle[audio_name] = user_message
        self.name2label = name2label
        self.name2subtitle = name2subtitle

        self.annotation = []
        for name in name2label:
            self.annotation.append({
                    'name': name,
                    'subtitle': name2subtitle[name],
                    'onehot':name2label[name]
            })

        self.label_type_candidates = ['emotion_recognition']

        # 设置数据路径
        vis_root = None
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
