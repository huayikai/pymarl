# 导入 agent_REGISTRY，这是一个全局字典，用于通过名字（如"rnn"）查找对应的智能体类
from modules.agents import REGISTRY as agent_REGISTRY
# 导入 action_REGISTRY，这是一个全局字典，用于通过名字（如"epsilon_greedy"）查找对应的动作选择器类
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# 这是一个多智能体控制器（MAC），它在所有智能体之间共享参数。
# 它是一个高级封装器，负责管理底层的神经网络（self.agent）和数据（ep_batch）。
# 负责给网络输入，然后从网络中取出数据
class CentralBasicMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        
        # 1. 根据数据蓝图（scheme）和配置（args）计算神经网络的输入维度
        input_shape = self._get_input_shape(scheme)
        
        # 2. 构建底层的神经网络
        self._build_agents(input_shape)
        
        # 3. 记录智能体（神经网络）的输出类型（例如 "pi" 表示策略，"q" 表示Q值）
        self.agent_output_type = args.agent_output_type

        # 4. 初始化RNN的隐藏状态
        self.hidden_states = None

    def forward(self, ep_batch, t, test_mode=False):
        """
        在单个时间步 `t` 上执行前向传播。
        """
        # 1. 从批次数据（ep_batch）的第 t 步构建神经网络的输入
        agent_inputs = self._build_inputs(ep_batch, t)
        
        # 2. 调用底层神经网络（self.agent）的前向传播
        #    - agent_inputs: 当前时间步的输入
        #    - self.hidden_states: 上一个时间步的隐藏状态
        #    - 返回:
        #        - agent_outs: 当前时间步的输出（例如 Q值 或 策略logits）
        #        - self.hidden_states: 当前时间步更新后的隐藏状态
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 3. 将输出重塑为 (批次大小, 智能体数量, 动作数量, -1) 的形状
        #    -1 通常用于 COMA 算法中的 Q 值，对于 QMIX/VDN 通常为 (batch_size, n_agents, n_actions)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1)

    def init_hidden(self, batch_size):
        """
        为RNN初始化隐藏状态。
        """
        # 1. 调用底层智能体网络的 init_hidden() 来获取一个智能体的初始隐藏状态
        # 2. .unsqueeze(0): 增加一个批次维度
        # 3. .expand(batch_size, self.n_agents, -1): 
        #    将其复制扩展，以匹配 (批次大小, 智能体数量, 隐藏状态维度) 的形状
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

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
        th.save(self.agent.state_dict(), "{}/central_agent.th".format(path))

    def load_models(self, path):
        """
        辅助函数：加载底层神经网络的模型。
        """
        self.agent.load_state_dict(th.load("{}/central_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """
        构建*一个*智能体实例。
        这就是“参数共享”的实现方式：所有智能体都使用这同一个 self.agent 实例。
        """
        # 从智能体注册表（agent_REGISTRY）中，根据配置（self.args.central_agent）查找并实例化一个智能体类
        self.agent = agent_REGISTRY[self.args.central_agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        从数据批次（batch）的第 t 步构建神经网络的输入张量。
        这是实现“中心化”输入的地方。
        """
        bs = batch.batch_size
        # batch是一个字典
        inputs = []
        
        # 1. 添加局部观测（observation）
        # batch是batch_size*time*n_agents*obs_size，这里只提取了t时刻的维度，维度缩减为了batch_size*n_agents*obs_size
        inputs.append(batch["obs"][:, t])  
        
        # 2. [特定配置] 如果是 central_rnn_big，则使用*全局状态* `state` 替换 `obs`。
        #    这是“中心化训练” (Centralized Training) 的关键体现。
        if self.args.central_agent == "central_rnn_big":
            # 因为state是全部智能体共享的，所以没有n_agent维度，所以我们需要进行扩充
            # (batch_size, 1, state_dim) -> (batch_size, n_agents, state_dim)
            # 让每个智能体都接收到相同的全局状态
            inputs[0] = (batch["state"][:,t].unsqueeze(1).repeat(1,self.args.n_agents,1))
        
        # 3. [可选] 添加上一时刻的动作（one-hot编码）
        if self.args.obs_last_action:
            # 这里我们也切片了
            # 维度是从batch_size*time*n_agents*action_size -> batch_size*n_agents*action_size
            if t == 0: # 如果是第一步，没有上一时刻的动作，则添加全零张量
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        
        # 4. [可选] 添加智能体ID（one-hot编码）
        #    这有助于参数共享的网络区分它正在为哪个智能体做决策
        if self.args.obs_agent_id:
            # (n_agents, n_agents) 的单位矩阵
            # -> (1, n_agents, n_agents) 
            # -> (bs, n_agents, n_agents) 扩展以匹配批次大小
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # 5. 拼接所有输入
        #    - x.reshape(bs*self.n_agents, -1): 
        #      将每个输入张量从 (bs, n_agents, ...) 展平为 (bs*n_agents, ...)
        #    - th.cat([...], dim=1): 
        #      在特征维度（dim=1）上将所有输入拼接起来。
        #    - 最终形状：(bs * n_agents, total_input_shape)
        #      这样就可以一次性送入参数共享的 self.agent 网络
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        辅助函数：计算神经网络的*总*输入维度。
        这个函数的逻辑必须与 _build_inputs 中的拼接逻辑完全一致。
        """
        input_shape = scheme["obs"]["vshape"] # 基础输入：观测维度
        
        if self.args.central_agent == "central_rnn_big":
            # 如果使用全局状态，则输入维度变为全局状态的维度
            input_shape += scheme["state"]["vshape"]
            input_shape -= scheme["obs"]["vshape"]
            
        # [可选] 加上上一时刻动作的维度
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] # 注意 [0]
            
        # [可选] 加上智能体ID的维度（即 n_agents）
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape