import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. 定义多头图注意力层 (Multi-Head GAT) ---
# 对应论文公式 (8), (9), (10)
# 这里的图是全连接的 (Fully Connected)，所以我们不需要邻接矩阵，直接做 Attention
class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, top_k=None):
        super(MultiHeadGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.top_k = top_k # 保存 k 值
        
        # W 矩阵: 将输入的 hidden_state 映射到 GAT 的特征空间
        # 输出维度: n_heads * hidden_dim
        self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        
        # Attention 向量 a: 用于计算节点间的注意力权重
        # 输入是拼接的两个节点特征 [Wh_i || Wh_j]，所以是 2 * hidden_dim
        self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim)) # 创建一个矩阵并且将其变成可学习的
        nn.init.xavier_uniform_(self.att_a.data, gain=1.414) # 初始化
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        # h shape: (batch_size, n_agents, input_dim)
        bs, n_agents, _ = h.size()
        
        # 1. 线性变换: h -> Wh
        # shape: (bs, n_agents, n_heads * hidden_dim)
        h_prime = self.W(h) 
        # 重塑为多头: (bs, n_agents, n_heads, hidden_dim)
        h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
        # 转置以便后续广播: (bs, n_heads, n_agents, hidden_dim)，变换维度
        h_prime = h_prime.permute(0, 2, 1, 3)
        
        # 2. 构建注意力机制 (Fully Connected Graph)
        # 我们需要计算所有 i 和 j 的组合。
        # 扩展维度进行广播:
        # h_i: (bs, heads, n_agents, 1, hidden_dim)
        # h_j: (bs, heads, 1, n_agents, hidden_dim)
        h_i = h_prime.unsqueeze(3)
        h_j = h_prime.unsqueeze(2)
        
        # 拼接: [Wh_i || Wh_j] -> (bs, heads, n_agents, n_agents, 2*hidden_dim)
        # repeat 使得维度匹配
        h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1), 
                        h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)
        
        # 计算 e_ij (公式 10)
        # (bs, heads, n_agents, n_agents, 2*hid) * (1, heads, 1, 1, 2*hid) -> sum -> scalar
        # attention score: (bs, heads, n_agents, n_agents)
        e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        e = self.leaky_relu(e)

        # 稀疏图
        if self.top_k is not None and 0 < self.top_k < n_agents:
            # 1. 找到前 k 个最大的值的索引
            _, topk_indices = th.topk(e, k=self.top_k, dim=-1)
            
            # 2. 创建基础 mask (Top-K 位置为 True)
            mask_bool = th.zeros_like(e, dtype=th.bool)
            mask_bool.scatter_(-1, topk_indices, True)
            
            # 3. [关键步骤] 强制保留对角线
            # 生成对角线 mask (1, 1, n, n)
            diag_mask = th.eye(n_agents, device=e.device, dtype=th.bool).view(1, 1, n_agents, n_agents)
            
            # 使用逻辑或 (|) 操作：保留 (Top-K节点) 或 (自己)
            mask_bool = mask_bool | diag_mask
            
            # 4. 应用 Mask: 将不需要保留的位置设为 -inf (Softmax 后变为 0)
            # 注意取反操作 ~mask_bool
            e = e.masked_fill(~mask_bool, -1e9)
        
        # 计算 alpha_ij (公式 9)
        attention = F.softmax(e, dim=-1) # 对 j 维度做 softmax
        
        # 3. 聚合信息 (公式 8)
        # 这里实现了加权求和的操作，其中attention就相当于权重，h_prime就是两个智能体之间的特征信息
        # alpha: (bs, heads, n_agents, n_agents)
        # h_prime (h_j): (bs, heads, n_agents, hidden_dim)
        # matmul: (bs, heads, n_agents, n_agents) @ (bs, heads, n_agents, hidden_dim)
        #      -> (bs, heads, n_agents, hidden_dim)
        h_new = th.matmul(attention, h_prime)
        
        # 应用激活函数 sigma (通常是 ELU 或 ReLU)
        h_new = F.elu(h_new)
        
        return h_new # 输出 G^d (batch_size, n_heads, n_agents, hidden_dim)


# --- 2. 定义 DVD Mixer ---
class DVDMixer(nn.Module):
    def __init__(self, args):
        super(DVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        # 强制 abs=False，或者依赖 args
        self.abs = getattr(self.args, 'abs', True) 
        
        self.rnn_hidden_dim = args.rnn_hidden_dim 
        self.n_heads = getattr(args, 'dvd_heads', 4)
        self.gat_dim = getattr(args, 'gat_embed_dim', 32)

        self.gat_top_k = getattr(args, 'gat_top_k', None)

        # 组件 1: GAT
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads, top_k=self.gat_top_k)

        # [修正 1] 移除残差连接的维度，回归纯粹的 DVD 逻辑
        # 这样能保证 GAT 真正起到 "Proxy Confounder" 的过滤作用 
        # self.combined_dim = self.gat_dim + self.rnn_hidden_dim 
        self.combined_dim = self.gat_dim 

        # 组件 2: 状态超网络 (生成 W1)
        self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.combined_dim)
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

        # [修正 2] 针对 Non-Monotonic 的特殊初始化
        # 如果 abs=False，我们将最后一层的权重初始化得非常小，避免初始 Q 值震荡
        self.init_weights()

        # 归一化
        # self.layernorm = nn.LayerNorm(self.embed_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 默认使用正交初始化或 Xavier
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        # 如果是非单调模式，特别处理超网络的输出层
        if not self.abs:
            # 使 W1 初始值很小
            self.hyper_w_1_state.weight.data.mul_(0.01)
            # 使 W_final 初始值很小
            if isinstance(self.hyper_w_final, nn.Linear):
                self.hyper_w_final.weight.data.mul_(0.01)
            else:
                # 如果是 Sequential，处理最后一层
                self.hyper_w_final[-1].weight.data.mul_(0.01)

    def forward(self, agent_qs, states, hidden_states):
        bs = agent_qs.size(0) 
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        # Step 1: GAT 采样
        graphs_out = self.gat(hidden_states) # (bs*T, heads, agents, gat_dim)
        
        # [修正 1] 移除残差拼接
        # graphs_combined = th.cat([graphs_out, h_expanded], dim=-1)
        # 直接使用 GAT 输出作为去混淆后的特征
        graphs_final = graphs_out

        # Step 2: 计算 W1
        w1_state = self.hyper_w_1_state(states)
        w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.combined_dim)
        
        graphs_T = graphs_final.permute(0, 1, 3, 2)
        
        w1_heads = th.matmul(w1_state, graphs_T)
        
        if self.abs:
            w1_heads = th.abs(w1_heads)
            
        w1 = w1_heads.mean(dim=1)
        w1 = w1.permute(0, 2, 1)

        # Step 3: 混合
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)

        # 源代码
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # # 添加layernorm
        # hidden = th.bmm(agent_qs, w1) + b1
        # # 插入 LayerNorm，防止数值爆炸
        # hidden = self.layernorm(hidden) 
        # hidden = F.elu(hidden)
        
        w_final = self.hyper_w_final(states)
        if self.abs:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        v = self.V(states).view(-1, 1, 1)
        
        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

###########################################
# 把自己的hidden_state加入了
###########################################
# class DVDMixer(nn.Module):
#     def __init__(self, args):
#         super(DVDMixer, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.state_dim = int(np.prod(args.state_shape))
#         self.embed_dim = args.mixing_embed_dim
#         self.abs = getattr(self.args, 'abs', True)
        
#         # DVD 特有参数
#         self.rnn_hidden_dim = args.rnn_hidden_dim 
#         self.n_heads = getattr(args, 'dvd_heads', 4)
#         self.gat_dim = getattr(args, 'gat_embed_dim', 32)

#         # --- 组件 1: 轨迹图生成器 (GAT) ---
#         self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)

#         # --- [修改点 1] ---
#         # 计算拼接后的维度: GAT输出维度 + 原始Hidden维度
#         self.combined_dim = self.gat_dim + self.rnn_hidden_dim

#         # 组件 2: 状态超网络 (用于生成 W1)
#         # 输出维度变大，因为我们要处理 (GAT特征 + 原始特征)
#         # Old: ... * self.gat_dim
#         # New: ... * self.combined_dim
#         self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.combined_dim)
        
#         self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

#         # --- 组件 3: 第二层混合 (W_final) ---
#         if getattr(args, "hypernet_layers", 1) == 1:
#             self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
#         else:
#             hypernet_embed = self.args.hypernet_embed
#             self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
#                                                nn.ReLU(inplace=True),
#                                                nn.Linear(hypernet_embed, self.embed_dim))
            
#         self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
#                                nn.ReLU(inplace=True),
#                                nn.Linear(self.embed_dim, 1))

#         # 初始化
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0)

#     def forward(self, agent_qs, states, hidden_states):
#         bs = agent_qs.size(0) 
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
#         hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

#         # -----------------------------------------------------------
#         # Step 1: GAT 采样
#         # -----------------------------------------------------------
#         # graphs_out: (bs*T, n_heads, n_agents, gat_dim)
#         graphs_out = self.gat(hidden_states)

#         # --- [修改点 2: 拼接残差连接] ---
#         # 目标: 将原始 hidden_states 拼接到 graphs_out 后面
        
#         # 1. 扩展 hidden_states 维度以匹配 Multi-Head
#         # (bs*T, n_agents, rnn_dim) -> (bs*T, 1, n_agents, rnn_dim) -> (bs*T, n_heads, n_agents, rnn_dim)
#         h_expanded = hidden_states.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

#         # 2. 在最后一个维度拼接
#         # 结果维度: (bs*T, n_heads, n_agents, gat_dim + rnn_dim)
#         graphs_combined = th.cat([graphs_out, h_expanded], dim=-1)

#         # -----------------------------------------------------------
#         # Step 2: 计算 W1
#         # -----------------------------------------------------------
#         # 2.1 生成状态表示 f_s(s)
#         # 注意这里维度是 combined_dim
#         w1_state = self.hyper_w_1_state(states)
#         w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.combined_dim)
        
#         # 2.2 调整维度准备矩阵乘法
#         # (bs*T, n_heads, agents, combined_dim) -> (bs*T, n_heads, combined_dim, agents)
#         graphs_T = graphs_combined.permute(0, 1, 3, 2)
        
#         # 2.3 MatMul
#         # (bs*T, heads, embed, combined) @ (bs*T, heads, combined, agents) 
#         # -> (bs*T, heads, embed, agents)
#         w1_heads = th.matmul(w1_state, graphs_T)
        
#         if self.abs:
#             w1_heads = th.abs(w1_heads)
            
#         w1 = w1_heads.mean(dim=1)
#         w1 = w1.permute(0, 2, 1)

#         # -----------------------------------------------------------
#         # Step 3: QMIX 标准流程
#         # -----------------------------------------------------------
#         b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
#         hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
#         w_final = self.hyper_w_final(states)
#         if self.abs:
#             w_final = th.abs(w_final)
#         w_final = w_final.view(-1, self.embed_dim, 1)
        
#         v = self.V(states).view(-1, 1, 1)
        
#         y = th.bmm(hidden, w_final) + v
#         q_tot = y.view(bs, -1, 1)
        
#         return q_tot