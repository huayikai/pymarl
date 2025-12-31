from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC # 导入 BasicMAC 作为父类


# 这是一个多智能体控制器，它在智能体之间共享参数
# LICA MAC (可能代表 LICA 算法) 是 BasicMAC 的一个子类
class LICAMAC(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        重写 select_actions 方法。
        """
        # 仅为 bs 中选定的批次元素选择动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        # [核心改动 1]
        # 当调用 forward 时，传递一个 gumbel 标志。
        # 如果不是测试模式（即在训练/收集数据），gumbel=True
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, gumbel=(not test_mode))
        
        # 将 forward 的输出（logits 或 probs）传递给动作选择器
        return self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def forward(self, ep_batch, t, test_mode=False, gumbel=False):
        """
        重写 forward 方法，增加了一个 gumbel 标志。
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 从底层神经网络获取原始输出（logits）和隐藏状态
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 如果输出类型是策略 logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # 在 softmax 之前，掩码不可用动作
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5

            # [核心改动 2]
            # 如果 gumbel 标志为 True (即在训练/收集数据时)
            if gumbel:
                # *不*执行 softmax，直接返回原始的 logits
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            # [默认行为] 如果 gumbel=False (即在测试时)
            # 正常执行 softmax，将 logits 转换为概率
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        # 返回最终结果（测试时是概率，训练时是 logits）
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)