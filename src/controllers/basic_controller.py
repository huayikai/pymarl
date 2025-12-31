# 导入 agent_REGISTRY，这是一个全局字典，用于通过名字（如"rnn"）查找对应的智能体类
from modules.agents import REGISTRY as agent_REGISTRY
# 导入 action_REGISTRY，这是一个全局字典，用于通过名字（如"epsilon_greedy"）查找对应的动作选择器类
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

# 这是一个多智能体控制器（MAC），它在所有智能体之间共享参数。
#
# ------------------------------------------------------------------------------------------
# 关键区别：
# 这是“基础”MAC，与你之前看的 CentralBasicMAC 不同。
# 1. 它是“去中心化”的：它*只能*访问局部的 "obs"，*不能*访问全局的 "state"。
# 2. 它是“执行者”：它*包含*一个 "action_selector"（动作选择器），
#    负责从神经网络的输出（Q值或logits）中选择一个*离散的*动作。
#
# 简而言之，这个 BasicMAC 是在环境中“执行”和“收集数据”时使用的。
# 就是用来玩游戏的，只能通过局部的obs去计算下一步的动作，和central_controller的定位不同
# ------------------------------------------------------------------------------------------
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        
        # 1. 根据数据蓝图（scheme）计算神经网络的输入维度
        input_shape = self._get_input_shape(scheme)
        
        # 2. 构建底层的（去中心化）神经网络
        self._build_agents(input_shape)
        
        # 3. 记录智能体（神经网络）的输出类型（例如 "pi_logits" 或 "q"）
        self.agent_output_type = args.agent_output_type

        # 4. [关键] 实例化一个动作选择器（例如 EpsilonGreedyActionSelector）
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        self.save_probs = getattr(self.args, 'save_probs', False)

        # 5. 初始化RNN的隐藏状态
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        这是用于“执行”的主函数。它调用 forward() 来获取Q值/logits，
        然后使用 action_selector 来选择一个离散动作。
        """
        # 1. 获取当前时间步 t_ep 的可用动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        # 2. 调用 self.forward() 来获取神经网络的输出（Q值或概率）
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        
        # 3. [关键] 调用动作选择器，根据Q值/概率和探索策略（如epsilon）来选择一个离散动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """
        在单个时间步 `t` 上执行神经网络的前向传播。
        它只返回Q值或概率，*不*选择动作。
        """
        # 1. 构建（去中心化的）输入
        agent_inputs = self._build_inputs(ep_batch, t)
        
        avail_actions = ep_batch["avail_actions"][:, t]
        
        if test_mode:
            # 在测试模式下，将神经网络设置为评估（evaluation）模式（例如关闭dropout）
            self.agent.eval()
            
        # 2. 调用底层神经网络（self.agent）的前向传播
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 3. [关键] 如果输出是策略logits（用于策略梯度算法）
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # 在 softmax 之前，将不可用动作的 logits 设置为一个非常小的负数
                # 这是为了确保它们在 softmax 后的概率接近于 0
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5  # 掩码操作

            # 应用 softmax 将 logits 转换为概率
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        # 4. 将输出重塑为 (批次大小, 智能体数量, 动作数量)
        #    对于Q-learning，这就是Q值；对于策略梯度，这就是动作概率
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """
        为RNN初始化隐藏状态。
        """
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # 扩展隐藏状态以匹配 (批次大小, 智能体数量, 隐藏维度) 的形状
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        """
        辅助函数：返回底层神经网络的参数（用于优化器）。
        """
        return self.agent.parameters()

    def load_state(self, other_mac):
        """
        辅助函数：从另一个MAC加载状态字典。
        """
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """
        辅助函数：将底层神经网络移动到CUDA设备。
        """
        self.agent.cuda()

    def save_models(self, path):
        """
        辅助函数：保存底层神经网络的模型。
        """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """
        辅助函数：加载底层神经网络的模型。
        """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """
        构建*一个*智能体实例（参数共享）。
        注意：它使用的是 self.args.agent，而不是 self.args.central_agent。
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        构建（去中心化的）输入张量。
        """
        bs = batch.batch_size
        inputs = []
        
        # 1. [关键] 添加局部观测（observation）。
        #    注意：它*永远不会*添加全局状态（state）！
        inputs.append(batch["obs"][:, t])  # b1av
        
        # 2. [可选] 添加上一时刻的动作
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        
        # 3. [可选] 添加智能体ID（用于参数共享的网络区分智能体）
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # 4. 拼接所有输入
        #    - x.reshape(bs, self.n_agents, -1): 保持智能体维度（n_agents）
        #    - th.cat([...], dim=-1): 在特征维度（dim=-1）上拼接
        #    - 最终形状：(bs, n_agents, total_input_shape)
        #    - 这与 CentralBasicMAC 不同，后者将 bs 和 n_agents 展平了。
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        辅助函数：计算（去中心化的）神经网络的*总*输入维度。
        """
        # 基础输入：观测维度
        input_shape = scheme["obs"]["vshape"]
        
        # [可选] 加上上一时刻动作的维度
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
            
        # [可选] 加上智能体ID的维度（即 n_agents）
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape