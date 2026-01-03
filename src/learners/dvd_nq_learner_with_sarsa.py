import copy
import torch as th
import numpy as np
from torch.optim import Adam, RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.dvd import DVDMixer
from modules.exploration.rnd import RNDModel

class RunningMeanStd:
    # 动态统计均值和方差，用于RND误差的归一化
    def __init__(self, shape=()):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def build_td_lambda_targets(rewards, terminated, mask, target_qvals, n_agents, gamma, td_lambda):
    # TD(lambda) 目标计算
    ret = target_qvals.new_zeros(*target_qvals.shape)
    ret[:, -1] = target_qvals[:, -1] * (1 - terminated[:, -1])
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + (1 - td_lambda) * gamma * target_qvals[:, t]
        ret[:, t] = rewards[:, t] + ret[:, t] * (1 - terminated[:, t])
    return ret

class DVDNQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.target_update_tau = 0.01

        # 初始化 Mixer
        if args.mixer == "dvd":
            self.mixer = DVDMixer(args)
        elif args.mixer == "qmix_without_abs":
            self.mixer = Mixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not supported in DVDNQLearner")

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        # 优化器
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # --- RND 模块初始化 ---
        self.use_rnd = getattr(args, "use_rnd", False)
        if self.use_rnd:
            self.rnd = RNDModel(args).to(self.device)
            self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=getattr(args, "rnd_lr", 5e-4))
            self.rnd_ms = RunningMeanStd()
            
            # [双重门控参数]
            # 1. 时间门控：预热步数，在此之前不给予内在奖励，防止初始 Loss 爆炸
            self.rnd_warmup_steps = getattr(args, "rnd_warmup_steps", 50000)
            # 2. 信号门控：阈值，只有归一化后的误差大于 mean + k*std 才视为有效探索
            self.rnd_gate_threshold = getattr(args, "rnd_gate_threshold", 1.0) 

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # 初始化统计变量
        rnd_loss_item = 0
        intrinsic_rewards_mean = 0
        
        # 初始化 uncertainty 容器 (用于传给 Mixer 的超网络)
        # 默认全为0，如果不使用 RND 或者 RND 处于后期衰减关闭状态
        all_uncertainty_norm = th.zeros(batch.batch_size, batch.max_seq_length, 1).to(self.device)

        # RND Beta 线性衰减
        rnd_decay_steps = 6000000 
        if t_env < rnd_decay_steps:
            progress = t_env / rnd_decay_steps
            current_rnd_beta = self.args.rnd_beta * (1.0 - progress)
        else:
            current_rnd_beta = 0.0

        # ==============================================================================
        # Part 1: RND 处理 (计算误差、归一化、双重门控、生成 Mixer 输入)
        # ==============================================================================
        if self.use_rnd:
            all_states = batch["state"]
            
            # 1. 计算原始误差 (Raw Error)
            # 重要：这里必须 detach，因为我们不需要 Mixer 的梯度反传回 RND 网络
            # RND 网络有自己的优化器
            all_rnd_error_raw = self.rnd(all_states).detach() 

            # 2. 更新统计量 (Running Mean/Std)
            # 只使用有效的时间步 (mask=1) 来更新统计，避免 padding 数据的干扰
            valid_mask = batch["filled"]
            valid_errors = all_rnd_error_raw[valid_mask == 1]
            if valid_errors.numel() > 0:
                self.rnd_ms.update(valid_errors.cpu().numpy())

            # 3. 全局归一化 (Global Normalization) -> 用于 Mixer 输入
            # 解决了输入非平稳性问题：无论训练处于什么阶段，输入给 Mixer 的特征分布都接近 N(0,1)
            std = self.rnd_ms.var**0.5 + 1e-6
            all_uncertainty_norm = (all_rnd_error_raw - self.rnd_ms.mean) / std
            
            # 裁剪防止极值破坏超网络稳定性
            all_uncertainty_norm = th.clamp(all_uncertainty_norm, -5, 5)

            # 4. 计算内在奖励 (用于 Q-Learning)
            if current_rnd_beta > 1e-5:
                # 取出 t=0 到 T-1 的部分作为当前步的奖励
                intrinsic_rewards = all_uncertainty_norm[:, :-1]

                # [门控机制 1: 信号门控] 
                # 过滤掉底噪，只奖励显著新颖的状态 (> mean + 1.0 * std)
                # 这能有效防止 "Noisy TV" 问题（智能体在普通状态刷微小误差）
                gate_mask = (intrinsic_rewards > self.rnd_gate_threshold).float()
                intrinsic_rewards_gated = intrinsic_rewards * gate_mask

                # [门控机制 2: 时间门控/预热]
                # 在 Predictor 未收敛前，不给 Agent 加奖励，防止初始 Q 值爆炸
                if t_env < self.rnd_warmup_steps:
                    intrinsic_rewards_final = th.zeros_like(intrinsic_rewards_gated)
                else:
                    intrinsic_rewards_final = intrinsic_rewards_gated

                # 叠加到外在奖励
                # 注意维度对齐
                min_len = min(rewards.shape[1], intrinsic_rewards_final.shape[1])
                rewards = rewards[:, :min_len] + current_rnd_beta * intrinsic_rewards_final[:, :min_len]
                
                intrinsic_rewards_mean = intrinsic_rewards_final.mean().item()

            # 5. 训练 RND 网络 (Predictor)
            # 重新前向传播以获取带梯度的 Graph
            # 这里不需要 gate 或 warmup，Predictor 需要一直训练以适应环境
            pred_error_with_grad = self.rnd(all_states) 
            loss_input = pred_error_with_grad[:, :-1] # 对齐 mask
            
            min_len_loss = min(loss_input.shape[1], mask.shape[1])
            mask_loss = mask[:, :min_len_loss]
            
            rnd_loss = (loss_input[:, :min_len_loss] * mask_loss).sum() / mask_loss.sum()
            
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            # 梯度裁剪，保护 RND 网络
            th.nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), 1.0)
            self.rnd_optimizer.step()
            rnd_loss_item = rnd_loss.item()

        # ==============================================================================
        # Part 2: Agent (MAC) 前向传播
        # ==============================================================================
        self.mac.agent.train()
        mac_out = []
        hidden_states_list = [] 
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # 收集 Hidden States (batch, n_agents, rnn_dim) 用于 DVDMixer 的 GAT
            cur_h = self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
            hidden_states_list.append(cur_h)

        mac_out = th.stack(mac_out, dim=1)
        whole_hidden_states = th.stack(hidden_states_list, dim=1)
        
        # 截取 t=0 到 T-1 的 hidden states
        hidden_states_main = whole_hidden_states[:, :-1]
        
        # 获取选定动作的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # ==============================================================================
        # Part 3: Target Network & Target Mixer
        # ==============================================================================
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_hidden_states_list = []
            self.target_mac.init_hidden(batch.batch_size)

            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                cur_target_h = self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
                target_hidden_states_list.append(cur_target_h)

            target_mac_out = th.stack(target_mac_out, dim=1)
            whole_target_hidden_states = th.stack(target_hidden_states_list, dim=1)

            # SARSA Target 计算逻辑
            target_qvals_next = target_mac_out[:, 1:]
            next_actions = batch["actions"][:, 1:]
            
            # 对齐长度
            target_len = min(target_qvals_next.shape[1], next_actions.shape[1])
            target_qvals_next = target_qvals_next[:, :target_len]
            next_actions = next_actions[:, :target_len]
            
            # 获取 Q(s', a')
            target_chosen_qvals = th.gather(target_qvals_next, 3, next_actions).squeeze(3)

            # --- Target Mixer 调用 ---
            if self.args.mixer == "dvd":
                target_hs_next = whole_target_hidden_states[:, 1:1+target_len]
                target_states_next = batch["state"][:, 1:1+target_len]
                
                # [Mixer 注入点] 获取 t+1 时刻的归一化不确定性
                # 这里的 all_uncertainty_norm 是归一化且稳定的，适合做超网络输入
                target_uncertainty_input = all_uncertainty_norm[:, 1:1+target_len]
                
                target_q_tot = self.target_mixer(
                    target_chosen_qvals, 
                    target_states_next, 
                    target_hs_next,
                    uncertainty=target_uncertainty_input 
                )
            else:
                target_q_tot = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:1+target_len])

        # ==============================================================================
        # Part 4: Loss 计算 & Update
        # ==============================================================================
        
        # 数据对齐截断
        T_min = min(chosen_action_qvals.shape[1], target_q_tot.shape[1], hidden_states_main.shape[1])
        chosen_action_qvals = chosen_action_qvals[:, :T_min]
        hidden_states_main = hidden_states_main[:, :T_min]
        target_q_tot = target_q_tot[:, :T_min]
        rewards = rewards[:, :T_min] # 这里使用的是包含了 RND 奖励的 rewards
        mask = mask[:, :T_min]
        terminated = terminated[:, :T_min]

        # 计算 TD(lambda) Targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_q_tot, 
                                             self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # --- Online Mixer 调用 ---
        if self.args.mixer == "dvd":
            # [Mixer 注入点] 获取 t=0 到 T-1 的归一化不确定性
            online_uncertainty_input = all_uncertainty_norm[:, :T_min]
            
            online_q_tot = self.mixer(
                chosen_action_qvals, 
                batch["state"][:, :T_min], 
                hidden_states_main,
                uncertainty=online_uncertainty_input
            )
        else:
            online_q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :T_min])

        # TD Loss
        td_error = (online_q_tot - targets.detach())
        masked_td_error = mask * td_error
        loss_td = (masked_td_error ** 2).sum() / mask.sum()

        total_loss = loss_td

        # 反向传播
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 软更新
        self._update_targets_soft()

        # 日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask.sum().item()), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask.sum().item()), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask.sum().item()), t_env)

            if self.use_rnd:
                self.logger.log_stat("rnd_loss", rnd_loss_item, t_env)
                self.logger.log_stat("intrinsic_rewards_mean", intrinsic_rewards_mean, t_env)
                self.logger.log_stat("rnd_beta", current_rnd_beta, t_env)
                            
            self.log_stats_t = t_env

        return {}

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_targets_soft(self):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_tau) + param.data * self.target_update_tau
            )
        
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_update_tau) + param.data * self.target_update_tau
                )

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