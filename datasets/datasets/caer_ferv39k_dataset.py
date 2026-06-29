import os
import json
import random
from PIL import Image
from cv2.gapi import video
from decord import VideoReader
from nano_emox.datasets.datasets.base_dataset import BaseDataset
import config

class CAER_FERV39K_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, 
                 dataset_cfg=None, model_cfg=None):
        """
        CAER-FERV39K数据集预热音频分支
        """
        self.dataset = 'CAER_FERV39K'
        
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data)

        self.label_type_candidates = ['emotion_recognition']
        
        # 定义候选情感标签，用于 func_get_qa_onehot_w_candidates 方法
        self.candidate_labels = "anger, disgust, fear, happy, neutral, sad, surprise"

        description_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'caer_ferv39k_ann.json')
        name2label = {}
        name2subtitle={}        
        with open(description_ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'messages' in item and 'videos' in item:
                # 提取用户消息和助手回复
                user_message = item['messages'][0]['content']
                assistant_message = item['messages'][1]['content']
                video_path = item['videos'][0]

                # name - 支持mp4和avi两种格式
                video_name = os.path.basename(video_path)
                if video_name.endswith('.mp4'):
                    video_name = video_name.replace('.mp4', '')
                elif video_name.endswith('.avi'):
                    video_name = video_name.replace('.avi', '')
                else:
                    # 如果既不是mp4也不是avi，去掉最后一个点及其后面的扩展名
                    video_name = os.path.splitext(video_name)[0]
                name2label[video_name] = assistant_message
                name2subtitle[video_name] = user_message
        self.name2label = name2label
        self.name2subtitle = name2subtitle

        self.annotation = []
        for name in name2label:
            self.annotation.append({
                    'name': name,
                    'subtitle': name2subtitle[name],
                    'onehot':name2label[name]
            })

        # 设置数据路径
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = None
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
            mp4_path = os.path.join(config.PATH_TO_RAW_VIDEO[self.dataset], sample['name'] + '.mp4')
            avi_path = os.path.join(config.PATH_TO_RAW_VIDEO[self.dataset], sample['name'] + '.avi')
            if os.path.exists(mp4_path):
                return mp4_path
            elif os.path.exists(avi_path):
                return avi_path
            else:
                raise FileNotFoundError(f"Video file not found for sample '{sample['name']}'. Checked both .mp4 and .avi formats.")
        else:
            return None
        
    def _get_audio_path(self, sample):
        if self.wav_root is not None:
            return os.path.join(config.PATH_TO_RAW_AUDIO[self.dataset], sample['name'] + '.wav')
        else:
            return None
