import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
import math

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_agents, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_agents = n_agents

        # 1. 共享参数 theta_0 (Shared Parameters)
        self.weight = nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 2. 针对每个智能体的掩码阈值参数 s_i (Learnable Thresholds)
        self.threshold_params = nn.Parameter(th.Tensor(n_agents, out_features, in_features))
        
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
        thresholds = th.sigmoid(self.threshold_params)
        weights_expanded = self.weight.unsqueeze(0).expand(self.n_agents, -1, -1)
        
        # 公式 7 & 121: 软阈值 Mask
        # 使用直通估计器 (STE)
        condition = th.abs(weights_expanded) > thresholds
        mask_hard = condition.float()
        mask_soft = th.sigmoid(10 * (th.abs(weights_expanded) - thresholds)) 
        # 用mask_hard来前向传播，但是在反向计算梯度的时候用的是mask_soft
        mask = (mask_hard - mask_soft).detach() + mask_soft
        
        return mask

    def forward(self, input):
        is_batched = True
        if input.dim() == 2:
            total_rows, in_feat = input.shape
            batch_size = total_rows // self.n_agents
            input = input.view(batch_size, self.n_agents, in_feat)
        
        mask = self.get_masks() 
        effective_weights = self.weight.unsqueeze(0) * mask
        
        # 批量矩阵乘法
        output = th.einsum('bai, aoi -> bao', input, effective_weights)
        
        if self.bias is not None:
            output = output + self.bias
            
        if is_batched:
            output = output.reshape(-1, self.out_features)
            
        return output

class KaleidoscopeAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(KaleidoscopeAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = MaskedLinear(input_shape, args.rnn_hidden_dim, self.n_agents)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = MaskedLinear(args.rnn_hidden_dim, args.n_actions, self.n_agents)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # 获取输入原本的 Batch 和 Agent 维度
        # inputs shape: (batch_size, n_agents, input_shape)
        b, a, e = inputs.size()

        # MaskedLinear 会自动处理输入的 view，但输出是 (b*a, out)
        x = F.relu(self.fc1(inputs))
        
        # RNN 需要展平的输入
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        
        # fc2 输出也是 (b*a, n_actions)
        q = self.fc2(hh)
        
        # 在返回之前，必须把维度变回 (batch_size, n_agents, ...)
        # 否则 ActionSelector 接不住
        return q.view(b, a, -1), hh.view(b, a, -1)
    
    def get_all_masks(self):
        return [self.fc1.get_masks(), self.fc2.get_masks()]
    
    def get_shared_weights(self):
        return [self.fc1.weight, self.fc2.weight]