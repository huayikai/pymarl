# 导入 agent_REGISTRY，用于查找并创建神经网络
from modules.agents import REGISTRY as agent_REGISTRY
# 导入 action_REGISTRY，用于查找并创建动作选择器
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

# 这是一个多智能体控制器（MAC），它在所有智能体之间共享参数。
#
# ------------------------------------------------------------------------------------------
# 关键特性: DOPMAC
#
# 这是一个为“策略梯度”算法（Policy Gradient）设计的去中心化控制器。
# 它的*独特之处*在于它在 `forward` 方法中*直接*注入了 Epsilon-Greedy 探索。
# 它将神经网络的策略（policy）与一个均匀随机策略进行“混合”（blending）。
# ------------------------------------------------------------------------------------------
class DOPMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # 1. 获取（去中心化的）输入形状
        input_shape = self._get_input_shape(scheme)
        # 2. 构建底层的神经网络（参数共享）
        self._build_agents(input_shape)
        # 3. 记录输出类型，DOPMAC 假设这里是 "pi_logits"（策略logits）
        self.agent_output_type = args.agent_output_type

        # 4. 实例化动作选择器（例如 MultinomialActionSelector）
        self.action_selector = action_REGISTRY[args.action_selector](args)

        # 5. 初始化RNN隐藏状态
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        选择动作。这是在环境中“执行”时调用的。
        """
        # 1. 获取可用动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        # 2. 调用 forward() 获取“混合后”的动作概率
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        
        # 3. [关键] 调用动作选择器，从“混合后”的概率中 *采样* 一个动作
        #    注意：这里的 agent_outputs 已经是被 epsilon “污染”过的概率了
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """
        执行前向传播，计算“探索性”的动作概率。
        """
        # 1. 构建（去中心化的）输入
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 2. 通过神经网络获取原始的 logits
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 3. [核心逻辑] 如果输出是策略 logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # 3a. 在 softmax 之前掩码不可用动作
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e11 # 设置为极小的负数

            # 3b. 将 logits 转换为基础概率
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
            # 3c. [DOP 特有逻辑] 如果不是测试模式，则注入 Epsilon 噪声
            if not test_mode:
                # Epsilon "地板"（floor）
                epsilon_action_num = agent_outs.size(-1) # 动作总数
                if getattr(self.args, "mask_before_softmax", True):
                    # 更精确地计算 *可用* 动作的数量
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                # [DOP 关键注释] "DOP 两次向概率添加噪声（另一次在 action_selectors.py 中）"
                #
                # 3d. [关键] Epsilon-Greedy 概率“混合”
                #    新概率 = (1 - ε) * (原始策略概率) + (ε) * (均匀随机概率)
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                     + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # 3e. 再次掩码，确保混合后的概率中，不可用动作的概率为 0
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        # 4. 返回最终的（混合了噪声的）概率
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """
        初始化RNN隐藏状态
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        a = 1 # (这行代码 a = 1 似乎是调试残留，没有实际作用)

    # --- 以下是与 BasicMAC 相同的辅助函数 ---

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """
        构建（参数共享的）神经网络实例
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        构建（去中心化的）输入张量
        """
        bs = batch.batch_size
        inputs = []
        # 1. 局部观测
        inputs.append(batch["obs"][:, t])  # b1av
        
        # 2. [可选] 上一时刻动作
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
                
        # 3. [可选] 智能体ID
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # 4. 拼接
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        计算（去中心化的）输入形状
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape