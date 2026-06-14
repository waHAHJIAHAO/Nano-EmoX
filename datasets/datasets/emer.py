import os
import json
import random
from PIL import Image
import pandas as pd
from decord import VideoReader
from nano_emox.datasets.datasets.base_dataset import BaseDataset
from nano_emox import config
class EMER_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, 
                 dataset_cfg=None, model_cfg=None):
        """
        EMER实验数据集，用于进行情感推理任务
        """
        self.dataset = 'EMER'
        
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data)

        # 读取包含subtitles的标注文件
        csv_path = os.path.join(config.DATA_DIR[self.dataset], 'gt-eng.csv')
        df_labels = pd.read_csv(csv_path)
        
        # 检查必要的列是否存在
        if 'names' not in df_labels.columns:
            raise ValueError(f"CSV文件{csv_path}中缺少'names'列")
        if 'subtitles' not in df_labels.columns:
            raise ValueError(f"CSV文件{csv_path}中缺少'subtitles'列")
        
        # 构建name到subtitle的映射
        name2subtitle = {}
        for _, row in df_labels.iterrows():
            name = str(row['names'])
            subtitle = str(row['subtitles']) if pd.notna(row['subtitles']) else ''
            name2subtitle[name] = subtitle
        
        self.name2subtitle = name2subtitle
        names_list = list(name2subtitle.keys())
        self.annotation = [{'name': n, 'subtitle': name2subtitle[n]} for n in names_list]

        self.label_type_candidates = ['comprehensive_emotion_reasoning']

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

    def read_test_names(self):
        """读取测试集样本名称列表
        """
        test_names = []
        csv_path = os.path.join(config.DATA_DIR[self.dataset], 'gt-eng.csv') # 读取测试集标注文件
        df_names = pd.read_csv(csv_path)
        if 'names' not in df_names.columns:
            raise ValueError(f"CSV文件{csv_path}中缺少'names'列")
        names_list = df_names['names'].astype(str).tolist()
        
        for name in names_list:
            test_names.append(name) 
        return test_names