import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. MultiHeadGAT (与 DVD-QMIX 共用) ---
class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MultiHeadGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim))
        nn.init.xavier_uniform_(self.att_a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        # h: (batch, n_agents, input_dim)
        bs, n_agents, _ = h.size()
        h_prime = self.W(h) 
        h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
        h_prime = h_prime.permute(0, 2, 1, 3) # (bs, heads, n_agents, hidden)
        
        h_i = h_prime.unsqueeze(3)
        h_j = h_prime.unsqueeze(2)
        
        # 全连接图注意力
        h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1), 
                        h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)
        
        e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        e = self.leaky_relu(e)
        attention = F.softmax(e, dim=-1)
        
        h_new = th.matmul(attention, h_prime)
        h_new = F.elu(h_new)
        return h_new


# --- 2. DVD-QPLEX 的 DMAQer ---
class DMAQer_DVD(nn.Module):
    def __init__(self, args):
        super(DMAQer_DVD, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = args.mixing_embed_dim
        
        # [DVD 参数]
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.dvd_heads = getattr(args, 'dvd_heads', 4)
        self.gat_dim = getattr(args, 'gat_embed_dim', 32)

        # [原始 QPLEX 组件] 保持不变，用于处理 V(s) 和 Q_lambda 的缩放
        hypernet_embed = self.args.hypernet_embed
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, self.n_agents))

        # [DVD 新增组件] 替换原有的 self.si_weight = DMAQ_SI_Weight(args)
        
        # 1. 轨迹图生成器
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.dvd_heads)
        
        # 2. 状态映射网络 (用于生成与图结合的权重)
        # 我们需要生成 (heads, 1, gat_dim) 形状的权重，以便与 Graph (heads, gat_dim, n_agents) 相乘
        # 输出维度: heads * 1 * gat_dim
        self.hyper_w_dvd = nn.Linear(self.state_dim, self.dvd_heads * self.gat_dim)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i, hidden_states):
        """
        计算优势函数部分 (Advantage)。
        这里使用 DVD 逻辑替代了原版 QPLEX 的 si_weight。
        """
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)
        
        # hidden_states: (bs, n_agents, rnn_dim)
        # 需要 reshape 匹配 states 的 batch 维度 (bs*seq_len)
        if hidden_states is not None:
            hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        # --- [DVD 核心逻辑: 计算 Mixing Weights (Credits)] ---
        
        # 1. 通过 GAT 获取轨迹图特征 G^d
        # graphs: (bs, heads, n_agents, gat_dim)
        graphs = self.gat(hidden_states)
        
        # 2. 生成状态特征 f(s)
        # w_dvd: (bs, heads * gat_dim)
        w_dvd = self.hyper_w_dvd(states)
        # reshape: (bs, heads, 1, gat_dim)
        w_dvd = w_dvd.view(-1, self.dvd_heads, 1, self.gat_dim)
        
        # 3. 结合 State 和 Graph
        # 我们需要得到每个 agent 的权重: (bs, heads, 1, n_agents)
        # graphs 转置: (bs, heads, gat_dim, n_agents)
        graphs_T = graphs.permute(0, 1, 3, 2)
        
        # MatMul: (heads, 1, gat) @ (heads, gat, agents) -> (heads, 1, agents)
        weights_dvd = th.matmul(w_dvd, graphs_T)
        
        # 4. 绝对值 (满足 IGM 且对应论文 |f(s)G|)
        weights_dvd = th.abs(weights_dvd)
        
        # 5. 后门调整 (对 heads 求平均)
        # (bs, heads, 1, n_agents) -> mean(dim=1) -> (bs, 1, n_agents)
        adv_w_final = weights_dvd.mean(dim=1).squeeze(1)
        
        # ---------------------------------------------------

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False, hidden_states=None):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            # 必须传入 hidden_states
            return self.calc_adv(agent_qs, states, actions, max_q_i, hidden_states)

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False, hidden_states=None):
        """
        注意：必须修改 Learner 传入 hidden_states
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        # 原版 QPLEX 的加权头逻辑 (用于 scaling, 通常保留)
        w_final = self.hyper_w_final(states)
        w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        if self.args.weighted_head:
            agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_q_i = w_final * max_q_i + v

        # 调用 calc 时传入 hidden_states
        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v, hidden_states=hidden_states)
        v_tot = y.view(bs, -1, 1)

        return v_tot