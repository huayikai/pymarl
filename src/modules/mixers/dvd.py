import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. 定义多头图注意力层 (Multi-Head GAT) ---
class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MultiHeadGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim))
        nn.init.xavier_uniform_(self.att_a.data, gain=1.414)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        bs, n_agents, _ = h.size()
        h_prime = self.W(h) 
        h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
        h_prime = h_prime.permute(0, 2, 1, 3)
        
        h_i = h_prime.unsqueeze(3)
        h_j = h_prime.unsqueeze(2)
        
        h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1), 
                        h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)
        
        e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        e = self.leaky_relu(e)
        
        attention = F.softmax(e, dim=-1)
        h_new = th.matmul(attention, h_prime)
        h_new = F.elu(h_new)
        
        return h_new 


# --- 2. 定义 DVD Mixer ---
class DVDMixer(nn.Module):
    def __init__(self, args):
        super(DVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True) 
        
        self.rnn_hidden_dim = args.rnn_hidden_dim 
        self.n_heads = getattr(args, 'dvd_heads', 4)
        self.gat_dim = getattr(args, 'gat_embed_dim', 32)

        # 组件 1: GAT
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)
 
        self.combined_dim = self.gat_dim 

        # [关键修改]：输入状态维度增加了 1 (因为我们要拼接 RND uncertainty)
        # 如果是评估模式或者没有uncertainty，我们会补0，所以维度是固定的
        self.input_state_dim = self.state_dim + 1

        # 组件 2: 状态超网络 (生成 W1)
        # 输入维度使用 self.input_state_dim
        self.hyper_w_1_state = nn.Linear(self.input_state_dim, self.n_heads * self.embed_dim * self.combined_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # 组件 3: 第二层混合 (W_final)
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
            
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        if not self.abs:
            self.hyper_w_1_state.weight.data.mul_(0.01)
            if isinstance(self.hyper_w_final, nn.Linear):
                self.hyper_w_final.weight.data.mul_(0.01)
            else:
                self.hyper_w_final[-1].weight.data.mul_(0.01)

    # [关键修改]：添加 uncertainty=None 参数
    def forward(self, agent_qs, states, hidden_states, uncertainty=None):
        bs = agent_qs.size(0) 
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        # [关键修改]：处理 uncertainty 拼接
        if uncertainty is not None:
            # uncertainty shape: (bs * T, 1)
            uncertainty = uncertainty.reshape(-1, 1)
            # 拼接到 state 后面，作为 hyper_w_1_state 的输入
            states_augmented = th.cat([states, uncertainty], dim=1)
        else:
            # 如果没有提供 (比如 evaluation 时)，用 0 填充以匹配维度
            zero_uncertainty = th.zeros(states.size(0), 1).to(states.device)
            states_augmented = th.cat([states, zero_uncertainty], dim=1)

        # Step 1: GAT 采样
        graphs_out = self.gat(hidden_states) # (bs*T, heads, agents, gat_dim)
        graphs_final = graphs_out

        # Step 2: 计算 W1
        # 使用拼接后的 augmented state
        w1_state = self.hyper_w_1_state(states_augmented)
        w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.combined_dim)
        
        graphs_T = graphs_final.permute(0, 1, 3, 2)
        
        w1_heads = th.matmul(w1_state, graphs_T)
        
        if self.abs:
            w1_heads = th.abs(w1_heads)
            
        w1 = w1_heads.mean(dim=1)
        w1 = w1.permute(0, 2, 1)

        # Step 3: 混合
        # b1, w_final, v 依然只依赖原始 states (通常不需要 uncertainty 干扰 bias 和 V)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)

        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        w_final = self.hyper_w_final(states)
        if self.abs:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        v = self.V(states).view(-1, 1, 1)
        
        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        
        return q_tot