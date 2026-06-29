import os
import json
import random
import pandas as pd
import re
from typing import Dict, List, Optional

import torch

from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
import config


class AvaMERG_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                 dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'AvaMERG'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print(f'Read data type: ######{self.label_type}######')
            print(f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print(self.needed_data)
        
        self.label_type_candidates = ['empathic_response_with_coe','emotion_recognition']
        self.candidate_labels = "conflicted,sadness,weary,upset,frustrated,disappointment,uneasy,confused,stressed,burdened,worry,nervous,confusion,helpless,overwhelm,surprised,proud,lonely,guilty,hopeful,anxious,nostalgic,content,caring,ashamed"
        # 情感映射字典（基于AvaMERG原始代码的emotion projection）
        self.ed_emotion_projection = {
            'conflicted': 'anxious', 'vulnerability': 'afraid', 'helplessness': 'afraid',
            'sadness': 'sad', 'pensive': 'sentimental', 'frustration': 'annoyed',
            'weary': 'tired', 'anxiety': 'anxious', 'reflective': 'sentimental',
            'upset': 'disappointed', 'worried': 'anxious', 'fear': 'afraid',
            'frustrated': 'sad', 'fatigue': 'tired', 'lost': 'jealous',
            'disappointment': 'disappointed', 'nostalgia': 'nostalgic', 'exhaustion': 'tired',
            'uneasy': 'anxious', 'loneliness': 'lonely', 'fragile': 'afraid',
            'confused': 'jealous', 'vulnerable': 'afraid', 'thoughtful': 'sentimental',
            'stressed': 'anxious', 'concerned': 'anxious', 'tiredness': 'tired',
            'burdened': 'anxious', 'melancholy': 'sad', 'overwhelmed': 'anxious',
            'worry': 'anxious', 'heavy-hearted': 'sad', 'melancholic': 'sad',
            'nervous': 'anxious', 'fearful': 'afraid', 'stress': 'anxious',
            'confusion': 'anxious', 'inadequacy': 'ashamed', 'regret': 'guilty',
            'helpless': 'afraid', 'concern': 'anxious', 'exhausted': 'tired',
            'overwhelm': 'anxious', 'tired': 'tired', 'disappointed': 'sad',
            'surprised': 'surprised', 'excited': 'happy', 'angry': 'angry',
            'proud': 'happy', 'annoyed': 'angry', 'grateful': 'happy',
            'lonely': 'sad', 'afraid': 'fear', 'terrified': 'fear',
            'guilty': 'sad', 'impressed': 'surprised', 'disgusted': 'disgusted',
            'hopeful': 'happy', 'confident': 'happy', 'furious': 'angry',
            'anxious': 'sad', 'anticipating': 'happy', 'joyful': 'happy',
            'nostalgic': 'sad', 'prepared': 'happy', 'jealous': 'contempt',
            'content': 'happy', 'devastated': 'surprised', 'embarrassed': 'sad',
            'caring': 'happy', 'sentimental': 'sad', 'trusting': 'happy',
            'ashamed': 'sad', 'apprehensive': 'fear', 'faithful': 'happy'
        }
        
        # 读取AvaMERG数据集的标注文件
        ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        with open(ann_path, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)
        
        # 处理标注数据，将每个turn作为一个样本
        self.annotation = []
        for conversation in raw_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']
            listener_profile = conversation['listener_profile']
            topic = conversation['topic']
            
            for turn in conversation['turns']:
                turn_id = turn['turn_id']
                context = turn['context']
                dialogue_history = turn['dialogue_history']
                response = turn['response']
                chain_of_empathy = turn['chain_of_empathy']
                
                # 分离历史对话和当前用户输入
                # 历史对话：除最后一轮外的所有对话
                history_dialogue = dialogue_history[:-1] if len(dialogue_history) > 1 else []
                # 当前用户输入：最后一轮的utterance
                current_user_input = dialogue_history[-1]['utterance'] if dialogue_history else ""
                
                # 格式化历史对话
                history_text = ""
                for turn in history_dialogue:
                    role = turn.get('role', 'unknown')
                    utterance = turn.get('utterance', '')
                    history_text += f"{role}: {utterance}\n"
                history_text = history_text.strip()
                
                # 创建样本
                sample = {
                    'name': f"{conversation_id}_{turn_id}",  # 使用conversation_id和turn_id作为唯一标识
                    'conversation_id': conversation_id,
                    'turn_id': turn_id,
                    'context': context,
                    'current_user_input': current_user_input,  # 当前用户输入
                    'response': response,
                    'speaker_emotion': chain_of_empathy['speaker_emotion'],
                    'emotion_cause': chain_of_empathy['emotion_cause'],
                    'event_scenario': chain_of_empathy['event_scenario'],
                    'goal_to_response': chain_of_empathy['goal_to_response'],
                    'speaker_profile': speaker_profile,
                    'speaker_id': speaker_id,
                    'listener_profile': listener_profile,
                    'topic': topic,
                    'subtitle': history_text,  # 历史对话作为字幕
                }
                
                self.annotation.append(sample)
        
        # 设置数据路径
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]  # 视频文件路径
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]  # 音频文件路径
        face_root = None  # AvaMERG使用原始视频文件，不使用预提取的人脸NPY文件
        
        # 调用父类初始化
        super().__init__(vis_processor=vis_processor,
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         ann_path=ann_path,
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
    
    def transform_conv_id(self, conv_id):
        """转换对话ID，移除前导零（基于AvaMERG原始代码）"""
        return re.sub(r'^0+', '', conv_id)

    def get_prompt_for_multimodal(self, face_or_frame, subtitle, user_message):
        """重写父类方法，为AvaMERG数据集生成特定的多模态提示，融入COE思考方法"""
        # 对于AvaMERG，我们主要使用frame（原始视频）而不是预提取的face
        if face_or_frame == 'faceframe': # (face, frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant:"
        elif face_or_frame == 'face': # (face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'frame': # (frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "        
        elif face_or_frame == 'frame_text': # (frame, text)
            assert subtitle is not None
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. " \
                    + f"The dialogue history is: <Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'face_text': # (face, text)
            assert subtitle is not None
            prompt =  f"###Human: We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"The dialogue history is: <Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'face_frame_text':
            assert subtitle is not None
            prompt = f"###Human: we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Meanwhile, We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"The dialogue history is: <Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "

        elif face_or_frame == 'audio_text': # (audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"The dialogue history is: <Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        ## 后面都是增加 <Multi> token 后的结果    
        elif face_or_frame == 'multiface_text': # (multi, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The dialogue history is:<Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"The dialogue history is:<Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The dialogue history is:<Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, frame, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, We uniformly extract facial informations from video: <Video><FaceHere></Video>. "  \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The dialogue history is:<Dialogue History>{subtitle}</Dialogue History>. " \
                    + f"Now here is the dialogue context:\n. {user_message} ###Assistant: "
        return prompt

    def _get_video_path(self, sample):
        """返回每轮speaker的视频文件路径"""
        conversation_id = sample['conversation_id']
        turn_id = str(int(sample['turn_id']) + int(sample['turn_id']))
        speaker_id = sample['speaker_id']
        video_filename = f"dia{conversation_id}utt{turn_id}_{speaker_id}.mp4"
        full_video_fp = os.path.join(self.vis_root, video_filename)
        return full_video_fp
    
    def _get_audio_path(self, sample):
        """返回每轮speaker的音频文件路径"""
        conversation_id = sample['conversation_id']
        turn_id = str(int(sample['turn_id']) + int(sample['turn_id']))
        speaker_id = sample['speaker_id']
        audio_filename = f"dia{conversation_id}utt{turn_id}_{speaker_id}.wav"
        full_audio_fp = os.path.join(self.wav_root, audio_filename)
        return full_audio_fp
    
# inference methods 只获取每个conversation中的最后一个turn作为测试集
    def _get_testvideo_path(self,sample):
        """返回每轮speaker的视频文件路径"""
        video_filename = f"{sample['name']}.mp4"
        full_video_fp = os.path.join("/home/lab00/hjh/dataset/AvaMERG/test_video", video_filename)
        return full_video_fp

    def _get_testaudio_path(self,sample):
        """返回每轮speaker的音频文件路径"""
        audio_filename = f"{sample['name']}.wav"
        full_audio_fp = os.path.join("/home/lab00/hjh/dataset/AvaMERG/test_audio", audio_filename)
        return full_audio_fp

    def read_test_names(self):
        """读取测试集样本名称列表, 格式: dia{conversation_id}utt{turn_id}_{speaker_id}
        """
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'v_test_v5_0.json')
        if not os.path.exists(test_ann_path):
            # 如果没有单独的测试集，使用训练集的一部分作为测试
            test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)
        
        test_names = []
        for conversation in test_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']
            # 只获取最后一个turn
            last_turn = conversation['turns'][-1]
            turn_id = str(int(last_turn['turn_id'])+int(last_turn['turn_id']))
            test_names.append(f"dia{conversation_id}utt{turn_id}_{speaker_id}")
        return test_names
    
    def get_test_name2gt(self):
        """获取测试集样本名称到真实标签的映射
        """
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'v_test_v5_0.json')
        if not os.path.exists(test_ann_path):
            # 如果没有单独的测试集，使用训练集的一部分作为测试
            test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)
        
        name2gt = {}
        for conversation in test_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']
            # 只获取最后一个turn
            last_turn = conversation['turns'][-1]
            turn_id = str(int(last_turn['turn_id'])+int(last_turn['turn_id']))
            name = f"dia{conversation_id}utt{turn_id}_{speaker_id}"
            # 获取speaker情感标签
            speaker_emotion = last_turn['chain_of_empathy']['speaker_emotion']
            # 使用情感映射字典进行标准化
            if speaker_emotion in self.ed_emotion_projection:
                speaker_emotion = self.ed_emotion_projection[speaker_emotion]
            name2gt[name] = speaker_emotion
        
        return name2gt
    
    @property
    def name2subtitle(self):
        """获取样本名称到对话记录的映射
        """
        name2subtitle = {}
        # 读取测试集标注文件
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'v_test_v5_0.json')
        if not os.path.exists(test_ann_path):
            # 如果没有单独的测试集，使用训练集的一部分作为测试
            test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)

        for conversation in test_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']
            last_turn = conversation['turns'][-1]
            turn_id = str(int(last_turn['turn_id'])+int(last_turn['turn_id']))
            name = f"dia{conversation_id}utt{turn_id}_{speaker_id}"
            # 获取speaker情感标签
            dialogue_history = last_turn['dialogue_history']
            history_dialogue = dialogue_history[:-1] if len(dialogue_history) > 1 else []
            name2subtitle[name] = history_dialogue
        return name2subtitle

    @property    
    def name2currentInput(self):
        """获取样本名称到当前输入的映射
        """
        name2currentInput = {}
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'v_test_v5_0.json')
        if not os.path.exists(test_ann_path):
            # 如果没有单独的测试集，使用训练集的一部分作为测试
            test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)

        for conversation in test_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']
            # 只获取最后一个turn
            last_turn = conversation['turns'][-1]
            turn_id = str(int(last_turn['turn_id'])+int(last_turn['turn_id']))
            name = f"dia{conversation_id}utt{turn_id}_{speaker_id}"
            # 获取speaker情感标签
            dialogue_history = last_turn['dialogue_history']
            current_user_input = dialogue_history[-1]['utterance'] if dialogue_history else ""
            name2currentInput[name] = current_user_input
        return name2currentInput

    @property    
    def name2context(self):
        """获取样本名称到上下文的映射
        """
        name2context = {}
        test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'v_test_v5_0.json')
        if not os.path.exists(test_ann_path):
            # 如果没有单独的测试集，使用训练集的一部分作为测试
            test_ann_path = os.path.join(config.DATA_DIR[self.dataset], 'train.json')
        
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            test_annotations = json.load(f)

        for conversation in test_annotations:
            conversation_id = conversation['conversation_id']
            speaker_profile = conversation['speaker_profile']
            speaker_id = speaker_profile['ID']

            last_turn = conversation['turns'][-1]
            turn_id = str(int(last_turn['turn_id'])+int(last_turn['turn_id']))
            name = f"dia{conversation_id}utt{turn_id}_{speaker_id}"
            # 获取speaker情感标签
            context = last_turn['context']
            name2context[name] = context
        return name2context