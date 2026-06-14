import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Ensure the parent directory is in sys.path so 'nano_emox' package can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from nano_emox.tasks import *
from nano_emox.models import *
from nano_emox.runners import *
from nano_emox.processors import *
from nano_emox.datasets.builders import *
from nano_emox.common.config import Config
from nano_emox.common.dist_utils import get_rank
from nano_emox.common.registry import registry
from nano_emox.conversation.conversation_video import Chat
from nano_emox.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls

from nano_emox import config
from nano_emox.utils.read_files import *


# 采用的是这个文件下存储数量最多的 root
def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # 找到 files 最多的 root
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # 打印最后一个文件的创建时间 for targetroot
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


# case1: 默认 => last epoch
# case2: 指定 inference_cfg.test_epoch == a; 那就只跑这个 epoch 下的结果
# case3: 指定 inference_cfg.test_epochs == a-b; 跑最后一个
def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
        if inference_cfg.ckpt_name != '':
            ckpts = [ckpt3_root + '/' + inference_cfg.ckpt_name]
        assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
        return [ckpts[0]]
    
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" %(ckpt3_root)))[-1]
        last_epoch=  int(last_ckpt.split('_')[-3])
        if inference_cfg.ckpt_name != '':
            last_ckpt = ckpt3_root + '/' + inference_cfg.ckpt_name
        #assert last_epoch > 10, f'Error: too less training time to conduct automatic inference!'
        return [last_ckpt]
    
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch) 
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


# 因为我们目前只处理 merbench，这些是 video 的，需要和原始训练数据中的 video 数据对应的 face_or_frame 一致
def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    elif 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    elif 'avamerg' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['avamerg'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, f'must has the unified face_or_frame type'
    face_or_frame = list(set(face_or_frame_candidates))[0]
    return face_or_frame


def get_name2cls(dataset):
    if dataset == 'MER2023':          return MER2023_Dataset()
    if dataset == 'MER2024':          return MER2024_Dataset()
    if dataset == 'MELD':             return MELD_Dataset()
    if dataset == 'IEMOCAPFour':      return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':          return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':         return CMUMOSEI_Dataset()
    if dataset == 'SIMS':             return SIMS_Dataset()
    if dataset == 'SIMSv2':           return SIMSv2_Dataset()
    if dataset == 'MER2025OV':        return MER2025OV_Dataset()
    if dataset == 'AvaMERG':          return AvaMERG_Dataset()
    if dataset == 'OVMERD':           return OVMERD_Dataset()
    if dataset == 'MIntRec':          return MIntRec_Dataset()
    if dataset == 'MIntRec2':         return MIntRec2_Dataset()
    if dataset == 'EMER':             return EMER_Dataset()

    print ('dataset cls not provided!')
    return None



def get_user_message(dataset_cls, zeroshot, outside_user_message, emotion_reason_inference=False, name=None):
    user_message = None
    if outside_user_message is not None:
        user_message = outside_user_message
    if zeroshot: # 3个分支对应3个对比实验
        if hasattr(dataset_cls, 'func_get_qa_ovlabel'):
            user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True) 

        if args.dataset == 'avamerg':
            user_message = dataset_cls.func_get_qa_empathic_response_with_coe(sample=None, question_only=True, name=name)

        if args.dataset == 'mintrec' or args.dataset == 'mintrec2':
            user_message = dataset_cls.func_get_intention(sample = None,question_only=True)

        if args.dataset=='emer' and emotion_reason_inference:
            if hasattr(dataset_cls, 'func_get_qa_description'):
                user_message = dataset_cls.func_get_qa_description(sample=None, question_only=True)
    else:
        # 非zeroshot模式，使用数据集特定的问题
        if hasattr(dataset_cls, 'func_get_qa_empathic_response_with_coe'):
            user_message = dataset_cls.func_get_qa_empathic_response_with_coe(sample=None, question_only=True, name=name)
        else:
            user_message = "Please analyze the emotion and provide an appropriate response."
    
    # 确保user_message不为None
    if user_message is None:
        user_message = "Please analyze the emotion and provide an appropriate response."
    
    return user_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano-EmoX Inference Process")
    parser.add_argument("--cfg-path", default='xxx', help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument("--dataset", default='merbench', help="evaluate dataset")
    parser.add_argument('--zeroshot', action='store_true', default=False, help='whether testing on zeroshot performance?')
    parser.add_argument('--outside_user_message',  default=None, help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_face_or_frame', default=None, help="we use the outside face_or_frame, rather than dataset dependent.")
    parser.add_argument('--emotion_reason_inference', action='store_true', default=False, help='whether testing on emotion reason inference task?')
    parser.add_argument('--custom_save_name', default=None, help='custom name for save directory, if not provided, use default ckpt3_root basename')
    args = parser.parse_args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)
    inference_datasets = ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2','OVMERD']
    #inference_datasets = ['MER2023', 'MER2024']
    

    print ('======== Step1: cfg pre-analysis ========')
    # 支持 ckpt_root / ckpt_name 两种类型输入 => (ckpt3_root)
    # 默认情况是依据 os.path.basename(args.cfg_path) 找到 => (ckpt3_root)
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
        assert inference_cfg.ckpt_name.startswith(cfg_name) # 这块和 train 部分是相互配合下的结果
    else:
        print ('strat searching for suitable ckpt_root')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)
        #ckpt3_root = "output/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250619185"
    print ('processed ckpt3 root:')
    print (ckpt3_root)

    # (ckpt3_root) => processed epochs
    print ('processed ckpt3 epochs:')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s: print (os.path.basename(item))

    # => (face_or_frame) (这个需要与训练数据采用的 face_or_frame 相同)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print (f'Read data type: {face_or_frame}')
    print ('=======================================')


    ## main process for each ckpt3 candidates
    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        ##############################################################
        print (f'======== Step2: initial model; using ckpt_3: {os.path.basename(ckpt_3)} ========')
        model_cfg.ckpt_3 = ckpt_3 # ckpt_3 has the highest priority
        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_cfg.arch) 
            model = model_cls.from_config(model_cfg)
        if ii > 0:  # second-round: update trainable params (用新的 ckpt_3 参数覆盖)
            ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval() # !! reduce randomness during the inference
        chat = Chat(model, model_cfg, device=device)
        ##############################################################


        print ('======== Step3: Inferece ========')
        if args.dataset == 'inferenceData':
            process_datasets = inference_datasets
        elif args.dataset == 'avamerg':
            process_datasets = ['AvaMERG']
        elif args.dataset == 'emer':
            process_datasets = ['EMER']
        elif args.dataset == 'mintrec':
            process_datasets = ['MIntRec','MIntRec2']
        else:
            names = args.dataset.split(',')
            process_datasets = names
        print ('process datasets: ', process_datasets)

        ## for each dataset
        for dataset in process_datasets:
            print (f'current dataset: {dataset}')
            ## dataset_cls 内部在 train / inference 内部的更新
            dataset_cls = get_name2cls(dataset)
            dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor") # read vis processor
            img_processor_cfg = inference_cfg.get("img_processor") # read img processor
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms


            ## 读取每个数据集的内容
            test_names = dataset_cls.read_test_names()
            name2subtitle = dataset_cls.name2subtitle

            ## 定义结果存储位置，如果存在相应路径直接跳过
            # 使用custom_save_name参数控制存储目录名称，如果未提供则使用默认的ckpt3_root basename
            save_dir_name = args.custom_save_name if args.custom_save_name is not None else os.path.basename(ckpt3_root)
            save_root = os.path.join(inference_cfg.base_root + f'-{dataset.lower()}', # output/results-{dataset}/custom_name_or_ckpt3_name
                                    save_dir_name) 
            if not os.path.exists(save_root): os.makedirs(save_root)
            epoch = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
            save_path = '%s/%s.npz' %(save_root, epoch) # output/result-{dataset}/custom_name_or_ckpt3_name/epochname
            if os.path.exists(save_path): continue

            ## 主要处理函数 【费时的主要在这个部分】
            name2reason = {}
            for ii, name in enumerate(test_names):
                subtitle = name2subtitle[name]
                print (f'process on {ii}|{len(test_names)}: {name} | {subtitle}')

                # 转成 cls 里面的支持类型进行 path 读取
                sample = {'name': name}
    
                video_path, image_path, audio_path, face_npy = None, None, None, None
                if args.dataset == 'avamerg' or args.dataset == 'mintrec' or args.dataset == 'mintrec2':
                    if hasattr(dataset_cls, '_get_testvideo_path'): video_path = dataset_cls._get_testvideo_path(sample)
                    if hasattr(dataset_cls, '_get_testaudio_path'): audio_path = dataset_cls._get_testaudio_path(sample)
                    if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
                    if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)
                else:                  
                    if hasattr(dataset_cls, '_get_video_path'): video_path = dataset_cls._get_video_path(sample)
                    if hasattr(dataset_cls, '_get_audio_path'): audio_path = dataset_cls._get_audio_path(sample)
                    if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
                    if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)
                
                sample_data = dataset_cls.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)
                # print (sample_data['face'].shape)

                # => img_list
                audio_llms, frame_llms, face_llms, image_llms, multi_llms = None, None, None, None, None
                audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)  
                frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
                _,             image_llms = chat.postprocess_image(sample_data)
                if face_or_frame.startswith('multiface') and model_cfg.arch == 'nano_emox':
                    _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
                elif face_or_frame.startswith('multiframe'):
                    _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)
                elif face_or_frame.startswith('multiface'):
                    _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)
                img_list = {}
                img_list['audio'] = audio_llms
                img_list['frame'] = frame_llms
                img_list['face']  = face_llms
                img_list['image'] = image_llms
                img_list['multi'] = multi_llms

                # get prompt (if use emotion_reason_inference => emotion reason inference; elif use zeroshot => ov labels; else => dataset specific question)
                user_message = get_user_message(dataset_cls, args.zeroshot, args.outside_user_message, args.emotion_reason_inference, name)
                if args.dataset == 'avamerg':
                    user_message = user_message + f"current speaker input is:{dataset_cls.name2currentInput[name]}"
                prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
                
                # => call function
                response = chat.answer_sample(prompt=prompt, img_list=img_list,
                                            num_beams=1, temperature=1, do_sample=True, top_p=0.9, 
                                            max_new_tokens=1200, max_length=2000) # llama: max_token_num=2048
                name2reason[name] = response
                print (response)

                # if ii == 0: break # for debug

            ## 保存结果 - 根据是否为emotion_reason_inference任务决定保存格式
            if args.emotion_reason_inference:
                csv_data = []
                for name in name2reason:
                    # 只需要names和chi_reasons两列，pred_label列为空（评估脚本不使用）
                    csv_data.append({
                        'names': name,
                        'pred_label': '', 
                        'chi_reasons': name2reason[name]
                    })
                csv_save_path  = 'output/ov-merd-eval'
                c_save_path = os.path.join(csv_save_path , f'{model_cfg.arch}_epoch{epoch}.csv')
                df = pd.DataFrame(csv_data)
                df.to_csv(c_save_path, index=False, encoding='utf-8')
                print (f'Inference results saved to: {c_save_path}')
            else:
                print ('save results')
                np.savez_compressed(save_path, name2reason=name2reason)
