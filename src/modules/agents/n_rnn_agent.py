import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    """
    [核心模块] 这是 "n_rnn" 智能体网络 (Agent Network)。
    它是一个基于 GRU (一种 RNN) 的神经网络，用于处理时间序列数据。
    它接收“局部观测” (obs) 和“上一时刻的隐藏状态” (hidden_state) 作为输入，
    并输出“Q值” (Q-Values)。
    """
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        # --- [网络架构定义] ---

        # 1. 输入层 (Fully Connected Layer 1)
        #    将扁平化后的输入 (obs + last_action + agent_id) 映射到 rnn 隐藏维度
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        # 2. 循环层 (Recurrent Layer)
        #    使用 GRUCell。它在每个时间步 t 接收 fc1 的输出 (x) 和 t-1 的隐藏状态 (h_in)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 3. 输出层 (Fully Connected Layer 2)
        #    将 rnn 的输出 (hh) 映射到每个动作的 Q 值
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # --- [可选的高级功能] ---

        # [可选] 是否使用层归一化 (Layer Normalization)
        # (在你的 qmix.yaml 中，use_layer_norm = False)
        if getattr(args, "use_layer_norm", False):
            # LayerNorm 通常加在 RNN 层之后，fc2 之前，用于稳定 RNN 的输出
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        # [可选] 是否使用正交初始化 (Orthogonal Initialization)
        # (在你的 qmix.yaml 中，use_orthogonal = False)
        if getattr(args, "use_orthogonal", False):
            # 一种特殊的权重初始化方法，有助于缓解梯度消失/爆炸，稳定训练
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        """
        [辅助函数] 创建一个初始的隐藏状态 (通常是全零张量)。
        这个函数在每个回合 (episode) 开始时被 `MAC` (控制器) 调用。
        """
        # .new(1, ...) 会在与模型参数*相同*的设备 (CPU 或 GPU) 上创建一个新张量
        # 形状为 (1, rnn_hidden_dim)
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        [核心] 执行一步 (one step) 前向传播。
        这个函数在*每个*时间步 `t` 都会被 `MAC` (控制器) 调用。
        """
        # inputs 形状: (batch_size * n_agents, input_shape) 
        #           或者 (batch_size, n_agents, input_shape) - 在 n_mac 中是后者
        # hidden_state 形状: (batch_size, n_agents, rnn_hidden_dim)

        # 获取输入的形状 (batch_size, n_agents, input_shape)
        # b = batch_size, a = n_agents, e = input_shape
        # (在 n_mac 的 _build_inputs 中，形状被塑造成 (b, a, e))
        b, a, e = inputs.size()

        # --- [处理数据形状] ---
        # 1. 将 (batch_size, n_agents, input_shape) 展平为 (batch_size * n_agents, input_shape)
        #    以便能将其喂给 nn.Linear (fc1)
        inputs = inputs.view(-1, e)
        
        # 2. [第 1 层: FC1 + ReLU]
        x = F.relu(self.fc1(inputs), inplace=True)
        
        # 3. 将 (batch_size, n_agents, rnn_hidden_dim) 展平为 (batch_size * n_agents, rnn_hidden_dim)
        #    以便能将其喂给 nn.GRUCell (rnn)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        
        # 4. [第 2 层: RNN]
        #    hh 是 t 时刻 *新*的隐藏状态
        hh = self.rnn(x, h_in)

        # 5. [第 3 层: FC2 (输出层)]
        if getattr(self.args, "use_layer_norm", False):
            # (如果启用) 在 fc2 之前应用 LayerNorm
            q = self.fc2(self.layer_norm(hh))
        else:
            # (标准路径)
            q = self.fc2(hh)

        # --- [返回结果] ---
        # 1. q (Q值): 
        #    - 形状从 (batch_size * n_agents, n_actions) 
        #    - 变回 (batch_size, n_agents, n_actions)
        # 2. hh (新的隐藏状态): 
        #    - 形状从 (batch_size * n_agents, rnn_hidden_dim) 
        #    - 变回 (batch_size, n_agents, rnn_hidden_dim)
        return q.view(b, a, -1), hh.view(b, a, -1)