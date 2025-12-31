import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac # Multi-Agent Controller (智能体网络)
        self.logger = logger
        
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')

        # --- 初始化混合网络 (Mixer) ---
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # --- 初始化优化器 ---
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # --- 初始化目标网络 (Target Network) ---
        # 这种深拷贝虽然有点浪费内存（复制了动作选择器等），但适用于任何 MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # --- 1. 获取 Batch 中的数据 ---
        # 注意切片 [:, :-1]，因为训练时我们需要用 t 时刻的状态预测 t+1 时刻的价值
        # 且 batch 数据通常包含直到结束后的一个 padding步
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float() # mask用于区分真实数据和填充(padding)数据
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # 如果上一时刻终止了，后续的 mask 全为 0
        avail_actions = batch["avail_actions"]
        
        # --- 2. 计算当前网络的 Q 值 (Estimated Q-Values) ---
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # 遍历时间步，计算每个时刻每个智能体对所有动作的 Q 值
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 堆叠时间维度 [batch_size, times, n_agents, n_actions]

        # --- 3. 提取实际执行动作的 Q 值 ---
        # gather: 根据 actions 索引从 mac_out 中取出对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [batch_size, times, n_agents]
        chosen_action_qvals_back = chosen_action_qvals
        
        # --- 4. 计算目标网络的 Q 值 (Target Q-Values) ---
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # 我们不需要第 0 个时间步的 target Q，因为是用 t 时刻预测 t+1
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # --- 5. 屏蔽不可行动作 (Mask out unavailable actions) ---
        # 将不可执行动作的 Q 值设为负无穷，防止在 max 操作中被选中
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # --- 6. 计算 Target Max Q (Double DQN 逻辑) ---
        if self.args.double_q:
            # Double DQN: 使用"当前网络"选动作，使用"目标网络"算价值
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999 # 同样需要屏蔽不可行动作
            # 找到当前网络认为的最优动作索引
            t = mac_out_detach[:, 1:].max(dim=3, keepdim=True)
            cur_max_actions = t[1]
            # 根据索引去目标网络取值
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # 普通 DQN: 直接取目标网络的最大值
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # --- 7. 混合 Q 值 (Mixing) ---
        # 如果使用了 Mixer (QMIX/VDN)，将个体 Q 值聚合成联合 Q 值 Q_tot
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # --- 8. 计算 TD Target (时序差分目标) ---
        # Bellman 方程: y = r + gamma * max Q(s', a')
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # --- 9. 计算 TD Error ---
        td_error = (chosen_action_qvals - targets.detach()) #[batch_size, times, 1]

        mask = mask.expand_as(td_error) # 确保 mask 维度与 td_error 一致

        # --- 10. 计算 Loss ---
        # 这里的 mask 作用是忽略掉 batch 中为了对齐长度而填充(padding)的数据，不让它们影响梯度
        masked_td_error = td_error * mask

        # L2 Loss (均方误差), 取平均值
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
        
        # --- 11. 优化 (Optimization) ---
        self.optimiser.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # --- 12. 更新目标网络 (Hard Update) ---
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # --- 13. 记录日志 ---
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))