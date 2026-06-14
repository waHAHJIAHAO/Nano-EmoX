# *_*coding:utf-8 *_*
import os

## 所有涉及 transformers 的模型存储路径
NANO_EMOX_ROOT = os.path.dirname(os.path.abspath(__file__))
EMOTION_WHEEL_ROOT = os.path.join(NANO_EMOX_ROOT, 'emotion_wheel')
RESULT_ROOT = os.path.join(NANO_EMOX_ROOT, 'output/results')


#######################
## 所有模型的存储路径
#######################
PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-1.5B-Instruct',
    'Qwen25_7B': 'models/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': 'models/clip-vit-large-patch14',
    'CLIP_VIT_LARGE_Plus': 'models/clip-vit-large-patch14',
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  'models/chinese-hubert-large',
    'HUBERT_LARGE_Plus':  'models/chinese-hubert-large',
}

PATH_TO_FACIAL = {
    'FaceXFormer': 'models/ckpt/model.pt',
}


#######################
## 所有数据集的存储路径
## 请根据实际部署环境修改 DATA_DIR 中的路径
#######################
DATA_DIR = {
    'MER2025OV':      'data/MER2025',
    'MERCaptionPlus': 'data/MER2025',
    'OVMERD':         'data/MER2025',
    'MER2023':        'data/MER-UniBench/mer2023-dataset-process',
    'MER2024':        'data/MER-UniBench/mer2024-dataset-process',
    'IEMOCAPFour':    'data/MER-UniBench/iemocap-process',
    'CMUMOSI':        'data/MER-UniBench/cmumosi-process',
    'CMUMOSEI':       'data/MER-UniBench/cmumosei-process',
    'SIMS':           'data/MER-UniBench/sims-process',
    'SIMSv2':         'data/MER-UniBench/simsv2-process',
    'MELD':           'data/MER-UniBench/meld-process',
    'AvaMERG':        'data/AvaMERG',
    'MERRFine':       'data/MER2023',
}
PATH_TO_RAW_AUDIO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'audio'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'audio'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'audio'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subaudio'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'audio'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
    'AvaMERG': os.path.join(DATA_DIR['AvaMERG'], 'audio'),
    'MERRFine': os.path.join(DATA_DIR['MERRFine'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'video','video'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'video','video'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'video','video'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo_new'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video_new'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
    'AvaMERG': os.path.join(DATA_DIR['AvaMERG'], 'video'),
    'MERRFine': os.path.join(DATA_DIR['MERRFine'], 'test3'),
}

PATH_TO_RAW_FACE = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'openface_face'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'openface_face'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'openface_face'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'openface_face'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'openface_face'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'openface_face'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'openface_face'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'subtitle_chieng.csv'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'subtitle_chieng.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
}
PATH_TO_LABEL = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'track2_train_ovmerd.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'track3_train_mercaptionplus.csv'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'xxx'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
    'AvaMERG': os.path.join(DATA_DIR['AvaMERG'], 'train.json'),
    'MERRFine': os.path.join(DATA_DIR['MERRFine'], 'MERR_fine_grained.json'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
TASK_ID = ['[Analysis]','[Recognition]','[Inference]','[Recogn_OpenVocabulary]','[Interaction]','[Clue]']
IGNORE_INDEX = -100
