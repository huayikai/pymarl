import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# QMixer (来自 "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning")
# 这是 QMIX 算法的*核心*：混合网络 (Mixing Network)
class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        # [关键] 将 state_shape (例如 [48]) 展平为一维向量的维度 (例如 48)
        self.state_dim = int(np.prod(args.state_shape))

        # 混合网络的嵌入维度 (例如 32 或 64)
        self.embed_dim = args.mixing_embed_dim
        # [QMIX 核心] 是否对超网络生成的权重取*绝对值* (abs)
        # 这是为了保证 Q_tot 和 Q_i 之间的单调性 (monotonicity)
        self.abs = getattr(self.args, 'abs', True)

        # [关键] 超网络 (Hypernetwork)
        # QMIX 的混合网络 (W1, W_final) 的权重 *不是* 固定的
        # 它们是由一个“超网络”根据*当前*的全局状态 (state) *动态生成*的

        # `hypernet_layers` 定义了超网络本身有几层 (通常是 1 或 2)
        if getattr(args, "hypernet_layers", 1) == 1:
            # --- 1层超网络 (默认) ---
            # "第1层权重" (W1) 的生成器
            # 输入: 全局状态 (state_dim), 输出: W1 的*所有*元素 (embed_dim * n_agents)
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            # "第2层权重" (W_final) 的生成器
            # 输入: 全局状态 (state_dim), 输出: W_final 的*所有*元素 (embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            # --- 2层超网络 (更复杂，但原理相同) ---
            hypernet_embed = self.args.hypernet_embed # (例如 64)
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # 超网络 - "第1层偏置" (b1) 的生成器
        # (偏置 b1 也是*状态依赖*的)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # 超网络 - V(s) 网络
        # 这用于代替最后一层的偏置 (b_final)
        # 它会根据*状态* s，生成一个*值* V(s)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        """
        [核心] QMIX 的前向传播。
        它将所有智能体的 Q 值 (agent_qs) 和全局状态 (states) 混合成 Q_tot。
        
        Args:
            agent_qs (Tensor): 形状 (batch_size, T-1, n_agents) - 各智能体*已选择动作*的 Q 值
            states (Tensor):   形状 (batch_size, T-1, state_dim) - 全局状态
        """
        bs = agent_qs.size(0)
        
        # [数据塑形] 将 (bs, T-1, ...) 展平为 (bs * (T-1), ...)
        # 以便超网络 (nn.Linear) 可以进行 2D 批处理
        states = states.reshape(-1, self.state_dim)
        # (bs * T-1, n_agents) -> (bs * T-1, 1, n_agents) (为矩阵乘法做准备)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        
        # --- 第1层混合 ---
        
        # 1. [Hypernetwork] 用 state 生成 W1 的权重
        # (如果 self.abs=True, 则取*绝对值*以保证单调性)
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        # 2. [Hypernetwork] 用 state 生成 b1 的偏置
        b1 = self.hyper_b_1(states)
        
        # 3. 将 W1 重塑为矩阵 (bs*T-1, n_agents, embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        # 4. 将 b1 重塑为 (bs*T-1, 1, embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        # 5. [第1层混合] (Q_i * W1) + b1
        #    th.bmm 是批量矩阵乘法 (Batch Matrix Multiplication)
        #    (b*T, 1, n_agents) @ (b*T, n_agents, embed_dim) -> (b*T, 1, embed_dim)
        #    使用 ELU 激活函数
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # --- 第2层混合 (输出) ---
        
        # 6. [Hypernetwork] 用 state 生成 W_final 的权重，并取绝对值
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        # 7. 将 W_final 重塑为 (bs*T-1, embed_dim, 1)
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # 8. [Hypernetwork] 用 state 生成 V(s) (作为最后的偏置)
        v = self.V(states).view(-1, 1, 1)
        
        # 9. [第2层混合] (hidden * W_final) + V(s)
        #    (b*T, 1, embed_dim) @ (b*T, embed_dim, 1) -> (b*T, 1, 1)
        y = th.bmm(hidden, w_final) + v
        
        # 10. 将最终的 Q_tot 重塑为 (batch_size, T-1, 1)
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    
    # --- [以下 k() 和 b() 是用于分析/调试的辅助函数] ---
    # 它们在标准的训练循环中*不*被调用

    def k(self, states):
        """
        k(s) 用于计算和归一化每个 agent_q 对 Q_tot 的*贡献权重* (k_i)
        k_i = (w1_i * w_final_i) / sum(w1_j * w_final_j)
        """
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # (b, n_agents, embed_dim) @ (b, embed_dim, 1) -> (b, n_agents, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        # 归一化
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        """
        b(s) 用于计算 state 决定的*基线* (baseline) 值 (即 b_tot)
        b_tot = (b1 * w_final) + V(s)
        """
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        # (b, 1, embed_dim) @ (b, embed_dim, 1) -> (b, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b