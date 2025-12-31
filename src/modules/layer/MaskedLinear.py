import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_agents, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_agents = n_agents

        # 1. 共享参数 theta_0 (Shared Parameters) [cite: 8]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 2. 针对每个智能体的掩码阈值参数 s_i (Learnable Thresholds)
        # 形状: (n_agents, out_features, in_features)
        # 初始化建议：让 sigma(s) 较小，初始接近全连接
        self.threshold_params = nn.Parameter(torch.Tensor(n_agents, out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # 初始化阈值参数，使得 sigma(s) 较小，Mask 初始全为 1
        nn.init.constant_(self.threshold_params, -5.0) 

    def get_masks(self):
        # 计算 Sigmoid 阈值
        thresholds = torch.sigmoid(self.threshold_params)
        
        # 扩展 weight 以匹配 n_agents 维度: (n_agents, out, in)
        weights_expanded = self.weight.unsqueeze(0).expand(self.n_agents, -1, -1)
        
        # 计算 Mask (公式 7 & 121) [cite: 115, 121]
        # 使用 Straight-Through Estimator (STE) 以允许梯度回传到 threshold_params
        # Forward: binary (0 or 1), Backward: pass through gradients
        condition = torch.abs(weights_expanded) > thresholds
        mask_hard = condition.float()
        mask_soft = torch.sigmoid(10 * (torch.abs(weights_expanded) - thresholds)) # 近似导数
        mask = (mask_hard - mask_soft).detach() + mask_soft
        
        return mask

    def forward(self, input):
        # input shape: (batch_size, n_agents, in_features) 或 (batch_size * n_agents, in_features)
        
        # 处理 PyMARL 输入形状，确保分离出 n_agents 维度
        is_batched = True
        if input.dim() == 2:
            # 假设输入是 (batch_size * n_agents, in_features)
            total_rows, in_feat = input.shape
            batch_size = total_rows // self.n_agents
            input = input.view(batch_size, self.n_agents, in_feat)
        
        # 获取掩码
        mask = self.get_masks() # (n_agents, out, in)
        
        # 应用掩码到共享权重: theta_i = theta_0 * M_i [cite: 115]
        # effective_weights shape: (n_agents, out, in)
        effective_weights = self.weight.unsqueeze(0) * mask
        
        # 执行线性变换
        # Input: (batch, n_agents, in)
        # Weights: (n_agents, out, in) -> 转置为 (n_agents, in, out) 用于矩阵乘法
        # Output: (batch, n_agents, out)
        # 使用 einsum 进行批量矩阵乘法
        output = torch.einsum('bai, aoi -> bao', input, effective_weights)
        
        if self.bias is not None:
            output = output + self.bias
            
        if is_batched:
            output = output.reshape(-1, self.out_features)
            
        return output