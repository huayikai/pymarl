import copy
import torch as th
from torch.optim import Adam, RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from modules.exploration.rnd import RNDModel

def build_td_lambda_targets(rewards, terminated, mask, target_qvals, n_agents, gamma, td_lambda):
    ret = target_qvals.new_zeros(*target_qvals.shape)
    # 修正说明: 
    # 1. 我们只需要判断序列的最后一步 (t=-1) 是否结束。
    # 2. terminated[:, -1] 的维度是 [Batch, 1]，与 target_qvals[:, -1] 完全匹配。
    # 3. 这样相乘不会触发错误的广播。
    ret[:, -1] = target_qvals[:, -1] * (1 - terminated[:, -1])
    # --- [修复结束] ---

    # 递归计算前序时间步
    for t in range(ret.shape[1] - 2, -1, -1):
        q_next = target_qvals[:, t]
        g_next = ret[:, t + 1]
        
        # TD(lambda) 核心公式
        ret[:, t] = td_lambda * gamma * g_next + (1 - td_lambda) * gamma * q_next
        
        # 应用 Reward 和 Termination
        # 注意: 这里 terminated[:, t] 维度也是 [Batch, 1]，计算安全
        ret[:, t] = rewards[:, t] + ret[:, t] * (1 - terminated[:, t])
        
    return ret

class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        # --- 初始化 Mixer ---
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "qmix_without_abs":
            self.mixer = Mixer(args)
        else:
            raise ValueError("Mixer error")
            
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        # --- 初始化 Optimizer ---
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        # --- 初始化 RND (用于非单调探索) ---
        # 论文引用: RND 鼓励探索未知状态 [cite: 258, 259]
        self.rnd = RNDModel(args).to(self.device)
        self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=args.rnd_lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # -------------------------------------------------------------------
        # 1. 数据准备与 RND 奖励计算
        # -------------------------------------------------------------------
        # 获取 batch 中的数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # 自动计算序列长度
        batch_size = batch.batch_size
        seq_len = actions.shape[1] # T

        # RND 计算内在奖励
        if hasattr(self, "rnd"):
            # 输入 next_state (s_{t+1})
            state_inputs = batch["state"][:, 1:]
            # 截断到与 actions 相同的长度 (防止 batch 长度不一致)
            state_inputs = state_inputs[:, :seq_len]
            
            rnd_predict_error = self.rnd(state_inputs)
            intrinsic_rewards = rnd_predict_error.detach()
            
            # 论文公式: r = r_ext + beta * r_int [cite: 263]
            rewards = rewards[:, :seq_len] + self.args.rnd_beta * intrinsic_rewards

        # -------------------------------------------------------------------
        # 2. 计算当前 Q_tot (Online Network)
        # -------------------------------------------------------------------
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        
        # 只需要跑到 seq_len (包含 t=0 到 T-1)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # (batch, T_max, n_agents, n_actions)

        # 取出 t=0 到 T-1 的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # (batch, T, n_agents)

        # 混合 Q 值得到 Q_tot
        # Mixer 输入: (batch, T, n_agents) -> 输出: (batch, T, 1)
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # -------------------------------------------------------------------
        # 3. 计算目标 Q_tot (Target Network) - SARSA Style
        # -------------------------------------------------------------------
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)

            # SARSA 关键点: 我们需要 s_{t+1} 下的 Q 值
            # 取出 t=1 到 T 的部分
            target_qvals_next = target_mac_out[:, 1:]
            
            # 论文核心: 不使用 max, 而是使用 buffer 中实际存储的 next_action (SARSA) [cite: 242, 244]
            # y_SARSA = r + gamma * Q(s', a')
            next_actions = batch["actions"][:, 1:]
            
            # 对齐长度 (防止 buffer 溢出)
            target_len = min(target_qvals_next.shape[1], next_actions.shape[1])
            target_qvals_next = target_qvals_next[:, :target_len]
            next_actions = next_actions[:, :target_len]
            
            # Gather Q(s_{t+1}, a_{t+1})
            target_chosen_qvals = th.gather(target_qvals_next, 3, next_actions).squeeze(3)

            # Target Mixer
            target_q_tot = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:1+target_len])

        # -------------------------------------------------------------------
        # 4. 计算 TD(lambda) Targets
        # -------------------------------------------------------------------
        # 确保所有张量长度对齐
        T_min = min(chosen_action_qvals.shape[1], target_q_tot.shape[1], rewards.shape[1], mask.shape[1])
        
        chosen_action_qvals = chosen_action_qvals[:, :T_min]
        target_q_tot = target_q_tot[:, :T_min]
        rewards = rewards[:, :T_min]
        mask = mask[:, :T_min]
        terminated = terminated[:, :T_min]

        # 调用 TD(lambda) 计算函数
        # 论文使用了 TD(lambda) 来平滑非单调情况下的学习信号 
        targets = build_td_lambda_targets(rewards, terminated, mask, target_q_tot, 
                                          self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # -------------------------------------------------------------------
        # 5. 计算 Loss 并更新
        # -------------------------------------------------------------------
        td_error = (chosen_action_qvals - targets.detach())
        masked_td_error = mask * td_error
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 更新 RND
        if hasattr(self, "rnd"):
            # 注意：rnd_predict_error 之前只取了 :seq_len，这里也要对齐 T_min
            rnd_loss = (rnd_predict_error[:, :T_min] * mask).sum() / mask.sum()
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
            
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("rnd_loss", rnd_loss.item(), t_env)
                self.logger.log_stat("intrinsic_reward_mean", intrinsic_rewards.mean().item(), t_env)

        # 更新 Target Network
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 记录日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask.sum().item()), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask.sum().item()), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask.sum().item()), t_env)
            self.log_stats_t = t_env
            
        return {}

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
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))