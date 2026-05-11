import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type
from torchvision.models import swin_b
import os

class FaceXFormerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states



class FaceXFormerEncoderOnly(nn.Module):
    """FaceXFormer编码器，用于MLLM中的面部特征编码
    输入: (bs, t, c, h, w) - batch_size, time_steps, channels, height, width
    输出: (bs, t, h, w) - batch_size, time_steps, height, width (特征图)
    """
    def __init__(self):
        super(FaceXFormerEncoderOnly, self).__init__()
        
        # Swin Transformer backbone
        swin_v2 = swin_b(weights='IMAGENET1K_V1')
        self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
        self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
        self.multi_scale_features = []

        self.num_encoder_blocks = 4
        self.hidden_sizes = [128, 256, 512, 1024]
        self.decoder_hidden_size = 256

        # 注册hook
        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))
        
        # 多尺度特征融合
        mlps = []
        for i in range(self.num_encoder_blocks):
            mlp = FaceXFormerMLP(input_dim=self.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=self.decoder_hidden_size * self.num_encoder_blocks,
            out_channels=self.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 重写train方法，确保模型始终保持评估模式
        def _disabled_train_facex_model(mode=True):
            return self
        self.train = _disabled_train_facex_model
        
    def save_features_hook(self, name):
        def hook(module, input, output):
            self.multi_scale_features.append(output.permute(0,3,1,2).contiguous()) 
        return hook
    
    def encode_single_frame(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, 256) - 池化后的特征向量
        """
        self.multi_scale_features.clear()
        
        # 通过backbone提取多尺度特征
        features = self.backbone(x).squeeze()
        
        # 处理多尺度特征
        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, 
                size=self.multi_scale_features[0].size()[2:], 
                mode="bilinear", 
                align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        
        # 融合多尺度特征
        fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1)) # (B, 256, H/4, W/4)
        fused_features = self.global_pool(fused_states)
        fused_features = fused_features.squeeze(-1).squeeze(-1)  # (b*t, d)
        return fused_features
    
    def forward(self, x):
        """前向传播
        输入: (bs, t, c, h, w)
        输出: (bs, t, 256) - 每帧池化后的特征向量
        """
        bs, t, c, h, w = x.shape
        
        # 重塑为 (bs*t, c, h, w) 以便批量处理
        x_reshaped = x.reshape(bs * t, c, h, w)
        # 编码每一帧，得到池化特征 (bs*t, 256)
        frame_features = self.encode_single_frame(x_reshaped)
        
        # 重塑回时序维度 frame_features: (bs, t, 256)
        _, c_feat = frame_features.shape
        frame_features = frame_features.view(bs, t, c_feat)
        
        return frame_features

class FacialEncoderFxformer(nn.Module):
    def __init__(self, llm_dim=768, num_facial_query_token=1, frozen_facial_llama_proj=False, model_path=None,device='cuda'):
        super().__init__()
        self.llm_dim = llm_dim
        self.device = device
        self.fxdim = 256

# ---------------------facexfomer encoder ---------------------
        self.facexformer = FaceXFormerEncoderOnly()
        if model_path is not None:
            ckpt = torch.load(model_path, map_location='cpu')
            self.facexformer.load_state_dict(ckpt['state_dict_backbone'], strict=False)
            self.facexformer.eval()
        
        # 确保FaceXFormerEncoderOnly始终保持完全冻结状态（包括MLP层）
        for param in self.facexformer.parameters():
            param.requires_grad = False

# ----------------------time attention pooling ---------------------
        self.query = nn.Parameter(torch.randn(num_facial_query_token, self.fxdim))  # (num_token, d)
        self.attn = nn.MultiheadAttention(embed_dim=self.fxdim, num_heads=4, batch_first=True)
        self.lynorm = nn.LayerNorm(self.fxdim)

# ---------------------projector---------------------
        self.face_proj = nn.Sequential(nn.Linear(self.fxdim, 768),
                                       nn.GELU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(768, self.llm_dim))

        if frozen_facial_llama_proj:
            for param in self.face_proj.parameters():
                param.requires_grad = False
        else:
            for param in self.face_proj.parameters():
                param.requires_grad = True
        
        # 重写train方法，确保FaceXFormerEncoderOnly始终保持评估模式，但允许时间池化层参与训练
        def _controlled_train_mode(mode=True):
            # 时间池化层和投影层可以根据mode参数切换训练/评估模式
            self.attn.train(mode)
            self.lynorm.train(mode)
            if not frozen_facial_llama_proj:
                self.face_proj.train(mode)
            
            # FaceXFormerEncoderOnly始终保持评估模式（包括所有子模块）
            self.facexformer.eval()
            
            return self
        
        self.train = _controlled_train_mode        

    def forward(self, x):
        fuse_feats = self.facexformer(x) # (b, t, d)

        b, t, d = fuse_feats.shape
        query = self.query.unsqueeze(0).repeat(b, 1, 1)  # (num_token, d) -> (b, num_token, d)
        fuse_feats, _ = self.attn(query, fuse_feats, fuse_feats)  # (b, num_token, d) 时间注意力压缩token num

        fuse_feats = self.face_proj(fuse_feats)
        return fuse_feats


def create_facial_encoder_fxformer(llm_dim=768,          
                                     num_facial_query_token=1,
                                     frozen_facial_llama_proj=False,
                                     model_path=None,
                                     device='cuda'):
    """
    Args:
    Returns:
        FacialEncoderInsightFace: 初始化的编码器
    """
    encoder = FacialEncoderFxformer(
        llm_dim=llm_dim,
        num_facial_query_token=num_facial_query_token,
        frozen_facial_llama_proj=frozen_facial_llama_proj,
        model_path=model_path,
        device=device
    )
    
    if torch.cuda.is_available() and device == 'cuda':
        encoder = encoder.cuda()
    
    return encoder

# 使用示例和数据流分析
if __name__ == "__main__":
    print("=== FaceXFormer Encoder-Only 数据流分析 ===")
    
    # 创建模型
    pretrained_path = os.path.join("models", "ckpt", "model.pt")
    device = 'cuda'

    # ckpt = torch.load(pretrained_path, map_location='cpu')
    #model = FaceXFormerEncoderOnly()
    # model.load_state_dict(ckpt['state_dict_backbone'], strict=False)
    # model.eval()
    model = create_facial_encoder_fxformer(llm_dim=1536,num_facial_query_token=4,model_path=pretrained_path,device='cuda')

    
    import decord
    from decord import VideoReader
    import random as rnd
    def load_video(video_path, n_frms=16, height=-1, width=-1, sampling="uniform", return_msg=False):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=height, width=width)

        vlen = len(vr)
        start, end = 0, vlen

        n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen

        if sampling == "uniform": # 均匀采样
            indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
        elif sampling == "headtail": # 前面随机采一半；后面随机采一半
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frms_update // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms_update // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        #########################################
        ## for vlen < n_frms, pad into n_frms
        while len(indices) < n_frms:
            indices.append(indices[-1])
        #########################################

        # get_batch -> T, H, W, C
        temp_frms = vr.get_batch(indices) # 这块报错 [h264 @ 0xc97e880] mmco: unref short failure => 这通常的是视频本身的问题
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

        if not return_msg:
            return frms

        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in indices])
        msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
        return frms, msg

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # 指定测试视频路径
    video_path = 'data/sample/sample_00000011.mp4' 
    #video_path = 'E:\\emotiondataset\\MMAFFIn\\videos\\MELD\\dia2_utt3.mp4'
    if os.path.exists(video_path):
        print(f"Loading video: {video_path}")
        try:
            # 加载视频帧
            video_result = load_video(video_path, n_frms=8, height=224, width=224, sampling="uniform")
            # 处理可能的tuple返回值
            if isinstance(video_result, tuple):
                video_frames = video_result[0]
            else:
                video_frames = video_result
            video_frames = video_frames.unsqueeze(0).to(device)  # 添加batch维度
            video_frames=video_frames.permute(0, 2, 1, 3, 4)
            print(f"Video loaded successfully: {video_frames.shape}")

            output = model(video_frames)
            print(f"facexencoder输出shape: {output.shape})")

        except Exception as e:
            bs, t, c, h, w = 2, 8, 3, 224, 224
            video_frames = torch.randn(bs, t, c, h, w)
            output = model(video_frames)
            print(f"facexencoder输出shape: {output.shape})")
    
    print("\nTest completed!")
