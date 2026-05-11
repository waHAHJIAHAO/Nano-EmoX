import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import fusion

#########################################
# 定义专家网络
#########################################
class WeightedGatedFusion(nn.Module):
    def __init__(self, d_audio, d_visual, d_output, dropout=0.1):
        super().__init__()
        # 投影层：将不同模态的特征投影到统一维度
        self.proj_audio = nn.Linear(d_audio, d_output)
        self.proj_visual = nn.Linear(d_visual, d_output)
        
        # 门控机制：学习模态权重
        self.gate = nn.Sequential(
            nn.Linear(d_audio + d_visual, d_output),
            nn.Sigmoid()  # 输出 0~1 的权重
        )
        
        # LayerNorm 和 Dropout 提高稳定性
        self.norm = nn.LayerNorm(d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, visual):
        # audio: (b, t_a, d_a), visual: (b, t_v, d_v)
        # 池化到统一时间步（可选，假设取平均）
        audio_pooled = torch.mean(audio, dim=1)  # (b, d_a)
        visual_pooled = torch.mean(visual, dim=1)  # (b, d_v)
        
        # 投影到统一维度
        audio_proj = self.proj_audio(audio_pooled)  # (b, d_output)
        visual_proj = self.proj_visual(visual_pooled)  # (b, d_output)
        
        # 门控权重
        gate_input = torch.cat([audio_pooled, visual_pooled], dim=-1)  # (b, d_a + d_v)
        gate_weights = self.gate(gate_input)  # (b, d_output)
        
        # 加权融合
        fused = gate_weights * audio_proj + (1 - gate_weights) * visual_proj  # (b, d_output)
        
        # 标准化和 dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        # 扩展时间维度（若需要 (b, 1, d_output)）
        fused = fused.unsqueeze(1)  # (b, 1, d_output)
        
        return fused

class simpleattention(nn.Module):
    def __init__(self, visual_hidden_size,audio_hidden_size,llm_hidden_size,num_multi_query_token,fusion_dim):
        super().__init__()
        self.fusion_dim = max(visual_hidden_size,audio_hidden_size) # 对齐特征
        self.llm_hidden_size = llm_hidden_size # LLM
        self.num_multi_query_token = num_multi_query_token # 融合特征token num
        self.multi_video_embs = nn.Linear(visual_hidden_size, fusion_dim) 
        self.multi_audio_embs = nn.Linear(audio_hidden_size, fusion_dim)
        self.attention_mlp = nn.Sequential(nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.fusion_dim, 2))
        self.multi_llama_proj = nn.Linear(self.fusion_dim, self.llm_hidden_size)

        for module in self.attention_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
    def forward(self,video_hidden_state,audio_hidden_state):
        video_hidden_state = torch.mean(video_hidden_state, dim=1) # [b, featdim1]
        audio_hidden_state = torch.mean(audio_hidden_state, dim=1) # [b, featdim2]
        video_hidden_state = self.multi_video_embs(video_hidden_state) # [b, maxdim]
        audio_hidden_state = self.multi_audio_embs(audio_hidden_state) # [b, maxdim]

        multi_hidden_state = torch.cat([video_hidden_state, audio_hidden_state], dim=1) # [b, maxdim * 2]
        attention = self.attention_mlp(multi_hidden_state) # [b, maxdim * 2] -> [b, maxdim] -> [b, 2]
        attention = torch.unsqueeze(attention, 2) # [b, 2, 1]

        multi_hidden2 = torch.stack([video_hidden_state, audio_hidden_state], dim=2) # [b, maxdim, 2]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [b, maxdim, 1]
        multi_hidden  = fused_feat.squeeze(dim=2) # [b, maxdim]

        # + multi_llama_proj
        inputs_llama = self.multi_llama_proj(multi_hidden) # [b, llmdim]
        inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_multi_query_token, -1) # [b, num_multi_token, llmdim]
        return inputs_llama

class simpleattention_no_proj(nn.Module):
    """专家网络，不进行LLM维度对齐，输出fusion_dim维度的特征"""
    def __init__(self, visual_hidden_size, audio_hidden_size, fusion_dim, num_multi_query_token):
        super().__init__()
        self.fusion_dim = max(visual_hidden_size, audio_hidden_size) # 对齐特征
        self.num_multi_query_token = num_multi_query_token # 融合特征token num
        self.multi_video_embs = nn.Linear(visual_hidden_size, fusion_dim) 
        self.multi_audio_embs = nn.Linear(audio_hidden_size, fusion_dim)
        self.attention_mlp = nn.Sequential(nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.fusion_dim, 2),
                                            nn.Softmax(dim=1),)
        # 不包含multi_llama_proj，输出保持在fusion_dim维度

        # 初始化所有线性层的权重
        nn.init.xavier_uniform_(self.multi_video_embs.weight, gain=0.1)
        nn.init.xavier_uniform_(self.multi_audio_embs.weight, gain=0.1)
        nn.init.zeros_(self.multi_video_embs.bias)
        nn.init.zeros_(self.multi_audio_embs.bias)
        
        for module in self.attention_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, video_hidden_state, audio_hidden_state):
        # 添加输入数值稳定性检查
        if torch.isnan(video_hidden_state).any() or torch.isinf(video_hidden_state).any():
            video_hidden_state = torch.nan_to_num(video_hidden_state, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(audio_hidden_state).any() or torch.isinf(audio_hidden_state).any():
            audio_hidden_state = torch.nan_to_num(audio_hidden_state, nan=0.0, posinf=1e6, neginf=-1e6)
            
        video_hidden_state = torch.mean(video_hidden_state, dim=1) # [b, featdim1]
        audio_hidden_state = torch.mean(audio_hidden_state, dim=1) # [b, featdim2]
        
        # 添加数值裁剪
        #video_hidden_state = torch.clamp(video_hidden_state, min=-10.0, max=10.0)
        #audio_hidden_state = torch.clamp(audio_hidden_state, min=-10.0, max=10.0)
        
        video_hidden_state = self.multi_video_embs(video_hidden_state) # [b, maxdim]
        audio_hidden_state = self.multi_audio_embs(audio_hidden_state) # [b, maxdim]
        
        # 添加投影后的数值稳定性处理
        #video_hidden_state = torch.clamp(video_hidden_state, min=-5.0, max=5.0)
        #audio_hidden_state = torch.clamp(audio_hidden_state, min=-5.0, max=5.0)
        #video_hidden_state = torch.nan_to_num(video_hidden_state, nan=0.0)
        #audio_hidden_state = torch.nan_to_num(audio_hidden_state, nan=0.0)

        multi_hidden_state = torch.cat([video_hidden_state, audio_hidden_state], dim=1) # [b, maxdim * 2]
        
        # 关键修复：添加attention计算的数值稳定性
        attention = self.attention_mlp(multi_hidden_state) # [b, maxdim * 2] -> [b, maxdim] -> [b, 2]
        #attention = torch.clamp(attention, min=-10.0, max=10.0)  # 裁剪attention分数
        #attention = torch.nan_to_num(attention, nan=0.0)  # 处理NaN
        
        # 使用softmax归一化attention权重，增加数值稳定性
        attention = torch.unsqueeze(attention, 2) # [b, 2, 1]

        multi_hidden2 = torch.stack([video_hidden_state, audio_hidden_state], dim=2) # [b, maxdim, 2]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [b, maxdim, 1]
        multi_hidden = fused_feat.squeeze(dim=2) # [b, maxdim]
        
        # 最终输出的数值稳定性处理
        #multi_hidden = torch.clamp(multi_hidden, min=-5.0, max=5.0)
        multi_hidden = torch.nan_to_num(multi_hidden, nan=0.0)
        
        outputs = torch.unsqueeze(multi_hidden, 1).expand(-1, self.num_multi_query_token, -1) # [b, num_multi_token, fusion_dim]
        return outputs

#########################################
# 定义路由网络
#########################################    

class DenseRoutingNetwork(nn.Module):
    def __init__(self, feature_dim, num_experts=3, noise_eps=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.noise_eps = noise_eps  # 用于增加随机性的噪声系数
        
        # 路由MLP（输出专家得分）
        self.router = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        # 初始化所有路由层的权重
        first_layer = self.router[0]
        nn.init.xavier_uniform_(first_layer.weight, gain=0.1)
        if first_layer.bias is not None:
            nn.init.zeros_(first_layer.bias)
            
        # 均匀初始化路由门控最后一层
        last_layer = self.router[-1]
        nn.init.uniform_(last_layer.weight, a=-0.02, b=0.02)  # 进一步减小初始化范围
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    def forward(self, features, training=True):
        # 输入特征数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        features = torch.clamp(features, min=-10.0, max=10.0)

        # 计算专家得分
        scores = self.router(features)
        scores = torch.clamp(scores, min=-5.0, max=5.0) # 更保守的分数裁剪
        scores = torch.nan_to_num(scores, nan=0.0)  # 处理可能的NaN
        
        # 添加噪声（训练时增加探索性）
        if training and self.noise_eps > 0:
            noise = torch.randn_like(scores) * self.noise_eps
            noise = torch.clamp(noise, min=-0.02, max=0.02)  # 进一步限制噪声范围
            scores += noise
            # 噪声添加后重新裁剪
            scores = torch.clamp(scores, min=-5.0, max=5.0)
            scores = torch.nan_to_num(scores, nan=0.0)
        
        # 改用更稳定的softmax归一化，避免sigmoid+除法的数值问题
        expert_weights = torch.softmax(scores, dim=1)
        
        # 确保数值稳定性
        expert_weights = torch.clamp(expert_weights, min=1e-6, max=1.0)
        expert_weights = torch.nan_to_num(expert_weights, nan=1.0/self.num_experts)  # 如果出现NaN，使用均匀分布
        
        # 重新归一化确保权重和为1
        weight_sum = torch.sum(expert_weights, dim=1, keepdim=True)
        expert_weights = expert_weights / (weight_sum + 1e-8)
        
        return expert_weights

class SparseRoutingNetwork(nn.Module):
    def __init__(self, feature_dim, num_experts=3, k=1, noise_eps=0.1):
        super().__init__()
        self.num_expert = num_experts
        self.k = k  # 每个样本选择的专家数量
        self.noise_eps = noise_eps  # 用于增加随机性的噪声系数
        
        # 路由MLP（输出专家得分）
        self.router = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        # 均匀初始化路由门控最后一层
        last_layer = self.router[-1]
        nn.init.uniform_(last_layer.weight, a=-0.05, b=0.05)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

        # 专家负载平衡损失的统计量
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def forward(self, features, training=True):
        # 计算专家得分
        scores = self.router(features)
        
        # 裁剪分数，避免数值不稳定
        scores = torch.clamp(scores, min=-20.0, max=20.0)
        
        # 添加噪声（训练时增加探索性）
        if training and self.noise_eps > 0:
            scores += torch.randn_like(scores) * self.noise_eps
        
        # 选择得分最高的k个专家
        top_k_values, top_k_indices = torch.topk(scores, self.k, dim=1)
        
        # 构建稀疏专家权重（未选中的专家权重为0）
        batch_size, num_experts = features.shape[0], scores.shape[1]
        expert_weights = torch.zeros(batch_size, num_experts, device=features.device)
        
        # 使用one-hot编码设置选中专家的权重
        for i in range(batch_size):
            # 对top_k_values进行裁剪，避免极端值
            normalized_values = torch.clamp(top_k_values[i], min=-10.0, max=10.0)
            expert_weights[i, top_k_indices[i]] = F.softmax(normalized_values, dim=0)
        
        # 更新专家使用统计（用于负载平衡）
        if training:
            self.expert_usage += expert_weights.sum(dim=0)
        
        return expert_weights
    
    def compute_load_balancing_loss(self, total_samples_in_period):
        """计算专家负载平衡损失（防止某些专家被过度使用）"""
        # 计算预期专家使用率（均匀分布） 每个专家选中概率为：前向传播次数/专家数
        num_experts = self.expert_usage.shape[0]  # 使用shape[0]代替size(0)
        expected_usage = torch.ones(num_experts, device=self.expert_usage.device) * (total_samples_in_period / self.num_expert) 
        
        # 确保expert_usage和expected_usage都分离计算图
        expert_usage_detached = self.expert_usage.detach().clone()
        expected_usage_detached = expected_usage.detach().clone()
        
        # 添加小常数避免数值问题
        expert_usage_detached = expert_usage_detached + 1e-5
        expected_usage_detached = expected_usage_detached + 1e-5
        
        # 先对数据进行归一化处理，增加数值稳定性
        expert_usage_norm = F.softmax(expert_usage_detached, dim=0)
        expected_usage_norm = F.softmax(expected_usage_detached, dim=0)
        
        # 使用L1损失代替MSE，避免梯度爆炸
        load_balance_loss = torch.mean(torch.abs(expert_usage_norm - expected_usage_norm))
        
        # 确保损失值在合理范围内
        load_balance_loss = torch.clamp(load_balance_loss, min=0.0, max=1.0)
        return load_balance_loss
    
    def reset_usage_stats(self):
        """重置专家使用统计"""
        with torch.no_grad():
            # 使用zeros_like创建新的零张量并赋值，避免直接调用fill_方法
            self.expert_usage = torch.zeros_like(self.expert_usage)

#########################################
# 定义多模态专家融合模型
#########################################

class DenseMoEFusion(nn.Module):
    def __init__(
        self, visual_hidden_size, 
        audio_hidden_size, fusion_dim, 
        llm_hidden_size, num_multi_query_token,
        num_experts=3
        ):
        super().__init__()
        # 特征对齐
        self.visual_projector = nn.Linear(visual_hidden_size, fusion_dim)
        self.audio_projector = nn.Linear(audio_hidden_size, fusion_dim)
        self.llm_dim = llm_hidden_size
        self.num_multi_query_token = num_multi_query_token
        self.fusion_dim = fusion_dim
        '''
        可以补充taskid处理流程，最后和特征在dim（-1）维度拼接后一起传入路由门控
        '''
        # 稠密路由网络 
        self.routing_network = DenseRoutingNetwork(
            feature_dim=fusion_dim*2,  # 拼接视觉和音频特征
            num_experts=num_experts
        )
        
        # 融合专家网络 - 使用不对齐LLM维度的版本
        self.experts = nn.ModuleList([simpleattention_no_proj(visual_hidden_size, audio_hidden_size,
                                    fusion_dim, num_multi_query_token) for _ in range(num_experts)])
        
        # 在DenseMoEFusion中添加LLM维度对齐的投影层
        self.llm_projector = nn.Linear(fusion_dim, llm_hidden_size)
    
    def forward(self, visual_features, audio_features, task_token=None):

        if torch.isnan(visual_features).any() or torch.isinf(visual_features).any():
            print("Warning: NaN or Inf detected in visual_features")
            visual_features = torch.nan_to_num(visual_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
            print("Warning: NaN or Inf detected in audio_features")
            audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        visual_hidden = visual_features.clone()
        audio_hidden = audio_features.clone()
        
        # 对视觉特征进行池化和投影
        visual_pooled = torch.mean(visual_features, dim=1)  # [b, visual_hidden_size]
        visual_proj = self.visual_projector(visual_pooled)
        visual_proj = torch.clamp(visual_proj, min=-10.0, max=10.0)
        visual_feat = torch.nan_to_num(visual_proj, nan=0.0, posinf=1e6, neginf=-1e6)

        # 对音频特征进行池化和投影
        audio_pooled = torch.mean(audio_features, dim=1)  # [b, audio_hidden_size]
        audio_proj = self.audio_projector(audio_pooled)
        audio_proj = torch.clamp(audio_proj, min=-10.0, max=10.0)
        audio_feat = torch.nan_to_num(audio_proj, nan=0.0, posinf=1e6, neginf=-1e6)

        # 拼接多模态特征
        multimodal_feat = torch.cat([visual_feat, audio_feat], dim=1)
        
        # 获取专家权重（稠密加权）
        expert_weights = self.routing_network(multimodal_feat) #shape:(b,num_experts)
        
        # 检查专家权重是否包含NaN或Inf
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            print("Warning: NaN or Inf detected in expert_weights")
            print(f"expert_weights stats: min={expert_weights.min()}, max={expert_weights.max()}, mean={expert_weights.mean()}")
            expert_weights = torch.nan_to_num(expert_weights, nan=0.33, posinf=1.0, neginf=0.0)  # 均匀分配权重
            expert_weights = expert_weights / (torch.sum(expert_weights, dim=1, keepdim=True) + 1e-8)
        
        # 计算所有专家的输出（稠密计算）
        batch_size, num_experts = expert_weights.shape[0], expert_weights.shape[1]
        
        # 计算专家输出，保持在fusion_dim维度
        expert_outputs_list = []
        for j in range(num_experts):
            expert_out = self.experts[j](visual_hidden, audio_hidden)  # [b, num_multi_query_token, fusion_dim]
            
            # 检查每个专家的输出
            if torch.isnan(expert_out).any() or torch.isinf(expert_out).any():
                print(f"Warning: NaN or Inf detected in expert {j} output")
                print(f"Expert {j} output stats: min={expert_out.min()}, max={expert_out.max()}, mean={expert_out.mean()}")
            
            expert_out = torch.nan_to_num(expert_out, nan=0.0, posinf=1e6, neginf=-1e6)
            expert_outputs_list.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs_list, dim=1)  # [b, num_experts, num_multi_query_token, fusion_dim]
        
        
        # 使用稠密加权聚合所有专家输出 - 逐个timestep级别的加权
        # expert_weights: [b, num_experts] -> [b, num_experts, 1, 1]
        expert_weights_expanded = expert_weights.unsqueeze(-1).unsqueeze(-1)  # [b, num_experts, 1, 1]
        
        # 加权聚合: [b, num_experts, num_multi_query_token, fusion_dim] * [b, num_experts, 1, 1]
        weighted_outputs = expert_outputs * expert_weights_expanded  # [b, num_experts, num_multi_query_token, fusion_dim]
        fused_output = torch.sum(weighted_outputs, dim=1)  # [b, num_multi_query_token, fusion_dim]
        fused_output = torch.nan_to_num(fused_output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 逐个timestep进行LLM维度对齐
        # 将 [b, num_multi_query_token, fusion_dim] reshape为 [b*num_multi_query_token, fusion_dim]
        batch_size, num_tokens, fusion_dim = fused_output.shape
        fused_output_reshaped = fused_output.view(-1, fusion_dim)  # [b*num_multi_query_token, fusion_dim]
        
        # 检查融合输出是否包含NaN或Inf
        if torch.isnan(fused_output_reshaped).any() or torch.isinf(fused_output_reshaped).any():
            print("Warning: NaN or Inf detected in fused_output before LLM projection")
            print(f"fused_output stats: min={fused_output_reshaped.min()}, max={fused_output_reshaped.max()}, mean={fused_output_reshaped.mean()}")
            fused_output_reshaped = torch.nan_to_num(fused_output_reshaped, nan=0.0, posinf=1.0, neginf=-1.0)
            # 进一步裁剪以确保数值稳定
            fused_output_reshaped = torch.clamp(fused_output_reshaped, min=-5.0, max=5.0)
        
        # 应用LLM投影层
        llm_output_reshaped = self.llm_projector(fused_output_reshaped)  # [b*num_multi_query_token, llm_dim]
        llm_output_reshaped = torch.nan_to_num(llm_output_reshaped, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 恢复原始形状
        final_output = llm_output_reshaped.view(batch_size, num_tokens, self.llm_dim)  # [b, num_multi_query_token, llm_dim]
        
        return final_output, expert_weights

class SparseMoEFusion(nn.Module):
    def __init__(
        self, visual_hidden_size, 
        audio_hidden_size, fusion_dim, 
        llm_hidden_size, num_multi_query_token,
        num_experts=3, k=1
        ):
        super().__init__()
        # 特征对齐
        self.visual_projector = nn.Linear(visual_hidden_size, fusion_dim)
        self.audio_projector = nn.Linear(audio_hidden_size, fusion_dim)
        self.llm_dim = llm_hidden_size
        self.num_multi_query_token = num_multi_query_token
        '''
        可以补充taskid处理流程，最后和特征在dim（-1）维度拼接后一起传入路由门控
        '''
        # 稀疏路由网络 
        self.routing_network = SparseRoutingNetwork(
            feature_dim=fusion_dim*2,  # 拼接视觉和音频特征
            num_experts=num_experts,  
            k=k
        )
        
        # 融合专家网络 b,t,d_a+d_v -> b,1,d_llm 输出已经对齐好LLM维度
        self.experts = nn.ModuleList([simpleattention(visual_hidden_size, audio_hidden_size,
                                    llm_hidden_size,num_multi_query_token,fusion_dim) for _ in range(num_experts)])
        
        # 输出层
        #self.output_proj = nn.Linear(fusion_dim, llm_hidden_size)  # 假设LLM输入维度为768
    
    def forward(self, visual_features, audio_features, task_token=None):
        if torch.isnan(visual_features).any() or torch.isinf(visual_features).any():
            print("Warning: NaN or Inf detected in visual_features")
            visual_features = torch.nan_to_num(visual_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
            print("Warning: NaN or Inf detected in audio_features")
            audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=1e6, neginf=-1e6)

        video_hidden_state = visual_features.clone()  # 用于专家网络
        audio_hidden_state = audio_features.clone()   # 用于专家网络
        visual_features_for_routing = visual_features.clone()  # 用于路由网络
        audio_features_for_routing = audio_features.clone()    # 用于路由网络

        # 对视觉特征进行池化和投影
        visual_features_for_routing = torch.mean(visual_features_for_routing, dim=1) # [b, visual_hidden_size]
        # 移除F.normalize操作，直接使用投影后的特征
        visual_proj = self.visual_projector(visual_features_for_routing)
        visual_proj = torch.clamp(visual_proj, min=-10.0, max=10.0)
        visual_feat = torch.nan_to_num(visual_proj, nan=0.0, posinf=1e6, neginf=-1e6)

        # 对音频特征进行池化和投影
        audio_features_for_routing = torch.mean(audio_features_for_routing, dim=1) # [b, audio_hidden_size]
        audio_proj = self.audio_projector(audio_features_for_routing)
        audio_proj = torch.clamp(audio_proj, min=-10.0, max=10.0)
        audio_feat = torch.nan_to_num(audio_proj, nan=0.0, posinf=1e6, neginf=-1e6)

        # 拼接多模态特征
        multimodal_feat = torch.cat([visual_feat, audio_feat], dim=1)
        
        # 获取专家权重（稀疏选择）
        expert_weights = self.routing_network(multimodal_feat) #shape:(b,num_experts)
        
        # 只计算被选中专家的输出（稀疏计算）
        batch_size, num_experts = expert_weights.shape[0], expert_weights.shape[1]
        expert_outputs = torch.zeros(batch_size, num_experts, self.llm_dim, 
                                    device=visual_feat.device)
        
        # 计算专家输出，并处理维度问题
        expert_outputs_list = []
        for j in range(num_experts):
            expert_out = self.experts[j](video_hidden_state, audio_hidden_state)
            # 确保输出维度正确：[b, num_multi_query_token, llm_dim] -> [b, llm_dim]
            if expert_out.dim() == 3:
                expert_out = expert_out[:, 0, :]  # 取第一个token的输出
            expert_out = torch.nan_to_num(expert_out, nan=0.0, posinf=1e6, neginf=-1e6)
            expert_outputs_list.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs_list, dim=1)  # [b, num_experts, llm_dim]
        
        # 确保expert_weights数值稳定
        expert_weights = torch.clamp(expert_weights, min=1e-6, max=1.0)
        expert_weights = expert_weights / (torch.sum(expert_weights, dim=1, keepdim=True) + 1e-6)
        expert_weights = torch.nan_to_num(expert_weights, nan=0.0)
        
        # 使用更安全的聚合方式，避免bmm可能的数值问题
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1e6, neginf=-1e6)
        # 维度扩充
        final_output = final_output.unsqueeze(1).expand(-1, self.num_multi_query_token, -1)
        # print(f"final_output shape is : {final_output.shape}")
        # 可以再写个FC处理专家的输出，看情况，若专家的实现没有对齐LLM维度就要补充

        return final_output, expert_weights

#########################################
# 定义多尺度融合编码器
#########################################

class MultiScaleCrossModalFusion(nn.Module):
    """
    多尺度跨模态融合编码器
    支持音频和视觉特征的多尺度Cross-Attention融合，包含残差连接提升稳定性
    """
    def __init__(self, audio_dim=1024, visual_dim=1024, hidden_dim=512,dropout=0.1):
        super().__init__()
        
        # 音频和视觉特征投影层
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # 三个交叉注意力层
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(3)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(3)
        ])
        
        # 门控网络 - 使用GELU替代ReLU
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, audio_multi, visual_multi):
        """
        前向传播
        Args:
            audio_multi: 多尺度音频特征 [batch_size, seq_len, audio_dim*3]
            visual_multi: 多尺度视觉特征 [batch_size, seq_len, visual_dim*3]
        Returns:
            final_feature: 融合后的特征 [batch_size, seq_len, hidden_dim]
        """
        # 切片多尺度特征
        audio_layers = torch.chunk(audio_multi, 3, dim=-1)  # 3个(b,t,1024)
        visual_layers = torch.chunk(visual_multi, 3, dim=-1)  # 3个(b,t,1024)
        
        fused_features = []
        for i, (audio_feat, visual_feat) in enumerate(zip(audio_layers, visual_layers)):

            audio_proj = self.audio_proj(audio_feat)  # (b,t,hidden_dim)
            visual_proj = self.visual_proj(visual_feat)  # (b,t,hidden_dim)
            
            fused_feat_t, _ = self.cross_attns[i](audio_proj, visual_proj, visual_proj)
        
            fused_feat_r = fused_feat_t + audio_proj  # 音频残差连接
            fused_feat = self.ffns[i](fused_feat_r)   # ffn残差

            fused_feat = fused_feat + fused_feat_r  

            fused_features.append(fused_feat)
        
        # 门控特征选择
        concat_features = torch.cat(fused_features, dim=-1)  # (b,t,hidden_dim*3)
        weights = self.gate_mlp(concat_features.mean(dim=1))  # (b,3)
        
        # 加权求和
        final_feature = sum(w.unsqueeze(1).unsqueeze(-1) * feat
                        for w, feat in zip(weights.unbind(-1), fused_features))
        
        return final_feature  # (b,t,hidden_dim)
