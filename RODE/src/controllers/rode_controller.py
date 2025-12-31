from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.roles import REGISTRY as role_REGISTRY
from modules.role_selectors import REGISTRY as role_selector_REGISTRY
import torch as th

from sklearn.cluster import KMeans
import numpy as np
import copy


# This multi-agent controller shares parameters between agents
# RODE 的多智能体控制器 (MAC)
class RODEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.role_interval = args.role_interval  # 角色更新间隔 (例如每 5 步重新选一次角色)

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.n_roles = 3  # 初始角色数量，后续会根据聚类结果动态调整
        self._build_roles()
        self.agent_output_type = args.agent_output_type

        # 初始化各个模块
        self.action_selector = action_REGISTRY[args.action_selector](args) # 动作选择器 (通常是 SoftEpsilonGreedy)
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args) # 角色选择器 (高层策略)
        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args) # 动作编码器 (用于学习动作的语义表示)

        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None # 存储当前每个智能体选择的角色索引
        self.n_clusters = args.n_role_clusters # 聚类的目标簇数量
        
        # role_action_spaces: 一个掩码矩阵 [n_roles, n_actions]
        # 值为1代表该角色可以执行该动作，为0代表禁止
        self.role_action_spaces = th.ones(self.n_roles, self.n_actions).to(args.device)

        # role_latent: 角色的特征向量 (通常是该角色包含的动作向量的中心)
        self.role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # --- 1. 获取环境原本允许的动作 ---
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        # --- 2. 前向传播，获取Q值和当前的角色 ---
        # agent_outputs: [batch, n_agents, n_actions]
        agent_outputs, role_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env=t_env)
        
        # --- 3. 构建角色掩码 (Role Mask) ---
        # 这一步非常关键：根据智能体当前选择的角色 (self.selected_roles)，
        # 从 self.role_action_spaces 中提取出允许的动作子集。
        
        # 维度变换: [n_roles, n_actions] -> [bs, n_agents, n_roles, n_actions] -> gather -> [bs, n_agents, n_actions]
        role_avail_actions = th.gather(
            self.role_action_spaces.unsqueeze(0).repeat(self.selected_roles.shape[0], 1, 1), 
            dim=1, 
            index=self.selected_roles.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.n_actions).long()
        ).squeeze()
        
        # role_avail_actions 现在是 [bs, n_agents, n_actions]，其中1表示角色允许，0表示角色禁止
        role_avail_actions = role_avail_actions.int().view(ep_batch.batch_size, self.n_agents, -1)

        # --- 4. 动作选择 ---
        # 调用 SoftEpsilonGreedyActionSelector
        # 注意这里传入了两个 avail_actions：
        # 1. avail_actions: 环境规则 (死了不能动)
        # 2. role_avail_actions: 角色规则 (医疗兵不能用大炮)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs],
                                                            role_avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, self.selected_roles, role_avail_actions

    def forward(self, ep_batch, t, test_mode=False, t_env=None):
        agent_inputs = self._build_inputs(ep_batch, t)

        # --- Role Selection (高层策略) ---
        # 更新角色选择器的 RNN 隐藏状态
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)
        role_outputs = None
        
        # RODE 不会每一步都切换角色，而是每隔 role_interval 步更新一次
        if t % self.role_interval == 0:
            # 这里的输入不仅有状态，还有 role_latent (角色的语义特征)
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            # 采样选出具体的角色 ID
            self.selected_roles = self.role_selector.select_role(role_outputs, test_mode=test_mode, t_env=t_env).squeeze()
            # self.selected_roles 形状: [bs * n_agents]

        # --- Action Selection (底层策略) ---
        # 更新动作网络的 RNN 隐藏状态
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        roles_q = []
        # RODE 的特点：每个角色可能有独立的 Q 网络 (self.roles[i])
        # 计算所有可能角色的 Q 值
        for role_i in range(self.n_roles):
            # 输入不仅有状态，还有 action_repr (动作的语义特征)
            role_q = self.roles[role_i](self.hidden_states, self.action_repr)  # [bs * n_agents, n_actions]
            roles_q.append(role_q)
            
        roles_q = th.stack(roles_q, dim=1)  # [bs*n_agents, n_roles, n_actions]
        
        # 只取当前被选中的那个角色的 Q 值
        # 如果我是角色0，我就只看 role_0_net 算出来的 Q 值
        agent_outs = th.gather(roles_q, 1, self.selected_roles.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.n_actions))
        # [bs * n_agents, 1, n_actions]

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
               (None if role_outputs is None else role_outputs.view(ep_batch.batch_size, self.n_agents, -1))

    def init_hidden(self, batch_size):
        # 初始化两套 RNN 隐藏状态：一套用于选动作，一套用于选角色
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.role_hidden_states = self.role_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        # 收集所有子网络的参数用于优化器更新
        params = list(self.agent.parameters())
        params += list(self.role_agent.parameters())
        for role_i in range(self.n_roles):
            params += list(self.roles[role_i].parameters())
        params += list(self.role_selector.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        if other_mac.n_roles > self.n_roles:
            self.n_roles = other_mac.n_roles
            self.roles = copy.deepcopy(other_mac.roles)
        else:
            for role_i in range(self.n_roles):
                self.roles[role_i].load_state_dict(other_mac.roles[role_i].state_dict())
        self.role_selector.load_state_dict(other_mac.role_selector.state_dict())

        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.role_action_spaces = copy.deepcopy(other_mac.role_action_spaces)
        self.role_latent = copy.deepcopy(other_mac.role_latent)
        self.action_repr = copy.deepcopy(other_mac.action_repr)

    def cuda(self):
        self.agent.cuda()
        self.role_agent.cuda()
        for role_i in range(self.n_roles):
            self.roles[role_i].cuda()
        self.role_selector.cuda()
        self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.role_agent.state_dict(), "{}/role_agent.th".format(path))
        for role_i in range(self.n_roles):
            th.save(self.roles[role_i].state_dict(), "{}/role_{}.th".format(path, role_i))
        th.save(self.role_selector.state_dict(), "{}/role_selector.th".format(path))

        th.save(self.action_encoder.state_dict(), "{}/action_encoder.th".format(path))
        th.save(self.role_action_spaces, "{}/role_action_spaces.pt".format(path))
        th.save(self.role_latent, "{}/role_latent.pt".format(path))
        th.save(self.action_repr, "{}/action_repr.pt".format(path))

    def load_models(self, path):
        self.role_action_spaces = th.load("{}/role_action_spaces.pt".format(path),
                                          map_location=lambda storage, loc: storage).to(self.args.device)
        self.n_roles = self.role_action_spaces.shape[0]
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.role_agent.load_state_dict(th.load("{}/role_agent.th".format(path), map_location=lambda storage, loc: storage))
        for role_i in range(self.n_roles):
            try:
                self.roles[role_i].load_state_dict(th.load("{}/role_{}.th".format(path, role_i),
                                                   map_location=lambda storage, loc: storage))
            except:
                self.roles.append(role_REGISTRY[self.args.role](self.args))
            self.roles[role_i].update_action_space(self.role_action_spaces[role_i].detach().cpu().numpy())
            if self.args.use_cuda:
                self.roles[role_i].cuda()
        self.role_selector.load_state_dict(th.load("{}/role_selector.th".format(path),
                                           map_location=lambda storage, loc: storage))

        self.action_encoder.load_state_dict(th.load("{}/action_encoder.th".format(path),
                                                    map_location=lambda storage, loc:storage))
        self.role_latent = th.load("{}/role_latent.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.action_repr = th.load("{}/action_repr.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.role_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_roles(self):
        self.roles = [role_REGISTRY[self.args.role](self.args) for _ in range(self.n_roles)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def update_role_action_spaces(self):
        """
        RODE 的核心逻辑：根据动作表示的相似性，动态聚类生成角色。
        这个函数通常在训练开始前或者定期调用。
        """
        # 1. 获取动作编码器学到的动作表示 (Action Representations)
        action_repr = self.action_encoder()
        action_repr_array = action_repr.detach().cpu().numpy()  # [n_actions, action_latent_d]

        # 2. 使用 K-Means 将动作聚类成 k 个簇
        k_means = KMeans(n_clusters=self.n_clusters, random_state=0).fit(action_repr_array)

        # 3. 根据聚类结果构建由 0/1 组成的空间掩码
        spaces = []
        for cluster_i in range(self.n_clusters):
            # 属于这个簇的动作置为 1 (True)，其他为 0
            spaces.append((k_means.labels_ == cluster_i).astype(float))

        # --- 特殊处理逻辑 (Heuristics) ---
        # 下面这段逻辑主要针对 StarCraft II (SMAC) 的动作空间特性进行了微调
        o_spaces = copy.deepcopy(spaces)
        spaces = []

        for space_i ,space in enumerate(o_spaces):
            _space = copy.deepcopy(space)
            # 动作 0 (No-Op) 和 1 (Stop) 通常是特殊的，先置0看看剩余动作的情况
            _space[0] = 0.
            _space[1] = 0.

            # 如果一个簇只包含 No-Op 和 Stop，保留它
            if _space.sum() == 2.:
                spaces.append(o_spaces[space_i])
            
            # 如果一个簇包含很多动作 (通常是攻击动作)，强制把前6个动作设为1
            # 在 SMAC 中，0-5 通常是移动指令 (Move N/S/E/W)
            # 这意味着：攻击型角色也应该拥有移动的能力
            if _space.sum() >= 3:
                _space[:6] = 1.
                spaces.append(_space)

        # 确保所有角色都能 No-Op (防止死锁)
        for space in spaces:
            space[0] = 1.

        # 兜底逻辑：如果生成的角色太少，强行复制凑数
        if len(spaces) < 3:
            spaces.append(spaces[0])
            spaces.append(spaces[1])

        print('>>> Role Action Spaces', spaces)

        # 4. 动态更新角色网络数量
        n_roles = len(spaces)
        if n_roles > self.n_roles:
            for _ in range(self.n_roles, n_roles):
                # 如果聚类出了更多角色，实例化新的角色网络
                self.roles.append(role_REGISTRY[self.args.role](self.args))
                if self.args.use_cuda:
                    self.roles[-1].cuda()

        self.n_roles = n_roles

        # 5. 将生成的空间掩码应用到角色网络上
        for role_i, space in enumerate(spaces):
            self.roles[role_i].update_action_space(space)

        # 6. 更新类的成员变量
        self.role_action_spaces = th.Tensor(np.array(spaces)).to(self.args.device).float()
        
        # 重新计算 Role Latent (动作簇的质心)
        # 公式: Sum(Action_Vectors * Mask) / Count(Actions)
        self.role_latent = th.matmul(self.role_action_spaces, action_repr) / self.role_action_spaces.sum(dim=-1, keepdim=True)
        self.role_latent = self.role_latent.detach().clone()
        self.action_repr = action_repr.detach().clone()

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        # 用于训练 Action Encoder
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])