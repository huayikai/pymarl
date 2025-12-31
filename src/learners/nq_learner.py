import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer       # 这就是 qmix.yaml 里的 "nmix"
from modules.mixers.vdn import VDNMixer     # VDN 混合器
from modules.mixers.qatten import QattenMixer # Qatten 混合器
from envs.matrix_game import print_matrix_status # (一个用于调试的辅助函数)
# [核心] 导入 TD(λ) 和 Q(λ) 目标计算函数
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num

class NQLearner: # "NQ"Learner 指的是 n-step Q-Learning (或 TD(Lambda))
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac # mac: "主"多智能体控制器 (n_mac)
        self.logger = logger
        
        self.last_target_update_episode = 0 # 计数器：上次更新目标网络的时间
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        # [关键] 1. 开始收集所有需要训练的参数，首先是 mac (即 n_rnn_agent) 的参数
        self.params = list(mac.parameters())

        # 根据配置 (args.mixer) 创建 "主" 混合网络
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
            
        # [关键] 2. 创建一个 "目标" 混合网络，它是 "主" 混合网络的*克隆体*
        # (它用于计算 Q-Learning 目标，权重是“旧”的)
        self.target_mixer = copy.deepcopy(self.mixer)
        
        # [关键] 3. 将 "主" 混合网络的参数也加入到 "待训练" 列表中
        # (这使得 mac 和 mixer 可以被*同一个*优化器*端到端*地训练)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters())) # 打印 Mixer 网络的参数量

        # 4. 根据配置创建优化器 (Adam 或 RMSprop)
        #    注意: `params=self.params` -> 优化器会同时更新 mac 和 mixer
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # [关键] 5. 创建一个 "目标" MAC，它是 "主" MAC 的*克隆体*
        self.target_mac = copy.deepcopy(mac)
        
        # 6. 初始化日志计时器 (设为负数，以确保在 t=0 时记录第一次日志)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # 7. 设置优先经验回放 (PER) (在你的配置中, use_per=False)
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
            
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        """
        [核心] 训练函数。
        这是在 `run_sequential` 中每次 `buffer.can_sample()` 为 True 时被调用的。
        """
        # --- [第 1 步: 准备数据和掩码 (Mask)] ---
        
        # 从批次中获取数据，[;, :-1] 表示选取*除最后一步之外*的所有数据
        # 因为 (s_t, a_t, r_t) 对应 (s_{t+1})，最后一步没有 s_{t+1}
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        
        # 创建掩码 (Mask)
        # 1. 基础掩码：只选取“已填充”(filled) 的、非最后一步的数据
        mask = batch["filled"][:, :-1].float()
        # 2. 终止掩码：如果一个回合在 t-1 步终止了，那么 t 步的数据是无效的，
        # 因此 t 步的 mask 必须为 0。
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # --- [第 2 步: 计算 "主网络" Q 值] ---
        # (即 Q_tot(s_t, a_t))
        
        self.mac.agent.train() # 设置为训练模式 (例如 启用 dropout/batchnorm)
        mac_out = []
        # 初始化 RNN 隐藏状态
        self.mac.init_hidden(batch.batch_size)
        
        # [关键] 逐个时间步 unroll RNN，获取所有时间步的 Q 值
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t) # (batch_size, n_agents, n_actions)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # (batch_size, max_seq_length, n_agents, n_actions)

        # [关键] "采 Q"
        # mac_out 包含了所有动作的 Q 值，但我们只需要*已采取*动作的 Q 值
        # th.gather() 会根据 actions (索引) 从 mac_out (源) 中"抓取"对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # (batch_size, T-1, n_agents) 每一个已经执行的动作的Q值
        chosen_action_qvals_ = chosen_action_qvals # (后面 q_lambda 会用)

        # --- [第 3 步: 计算 "目标网络" Q 值] ---
        # (这部分*不*需要计算梯度)
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            # 同样地，unroll "目标" RNN
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # 堆叠所有时间步
            target_mac_out = th.stack(target_mac_out, dim=1)  # (batch_size, max_seq_length, n_agents, n_actions)
            # 上面的步骤相当于在计算td-target

            # --- [核心: Double Q-Learning] ---
            # 1. 从 "主" 网络 (mac_out) 中*选择*动作
            mac_out_detach = mac_out.clone().detach() # 复制并分离梯度
            mac_out_detach[avail_actions == 0] = -9999999 # 掩码不可用动作
            # 找到 Q_main 最大的动作的 *索引* (a_t+1)
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            # 2. 从 "目标" 网络 (target_mac_out) 中*评估*该动作
            #    (即 Q_target(s_t+1, a_t+1))
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # --- [核心: TD(λ) 目标构建] ---
            
            # 3. 将 "目标" Q 值通过 "目标" Mixer 得到 Q_tot_target
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            # 4. (检查是使用 Q(λ) 还是 TD(λ)，在你的配置中 q_lambda=False)
            if getattr(self.args, 'q_lambda', False):
                # (Q(λ) 的复杂逻辑...)
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                       self.args.gamma, self.args.td_lambda)
            else:
                # [关键] 在你的配置中，使用 TD(λ)
                # 这个辅助函数会根据 (r, terminated, mask, Q_tot_target, gamma, lambda)
                # 计算出 n-step 的 TD(λ) 目标 y
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                        self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # --- [第 4 步: 计算损失 (Loss)] ---
        
        # 1. 将 "主" 网络的 Q 值 (chosen_action_qvals) 通过 "主" Mixer
        #    得到 Q_tot(s_t, a_t)
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # 2. [核心] TD-Error = (Q_tot - 目标 y)
        #    (targets.detach() 确保梯度*不*会流经目标网络)
        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2) # 平方

        # 3. 应用你之前计算的掩码
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # 4. 重要性采样的东西
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # 5. [核心] Loss = (1 / 有效步数) * (所有有效步的平方TD-Error之和)
        loss = L_td = masked_td_error.sum() / mask.sum()

        # --- [第 5 步: 优化 (Backpropagation)] ---
        
        # 1. 清空梯度
        self.optimiser.zero_grad()
        # 2. 反向传播 (梯度会流经 mixer 和 mac)
        loss.backward()
        # 3. 梯度裁剪 (防止梯度爆炸)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # 4. 优化器更新 *所有* 参数 (mac + mixer)
        self.optimiser.step()

        # --- [第 6 步: 更新目标网络 (如果需要)] ---
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets() # (调用下面的辅助函数)
            self.last_target_update_episode = episode_num

        # --- [第 7 步: 记录日志] ---
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # (这就是你之前问的 "负数初始化" 技巧生效的地方)
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # Q_taken_mean: 实际采取动作的 Q_tot 均值
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # target_mean: 目标 y 值的均值
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # (用于调试矩阵游戏的打印)
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # --- [第 8 步: 返回 PER 的优先级 (如果使用)] ---
        info = {}
        if self.use_per:
            # (计算 TD-Error 作为 PER 的优先级)
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # ... (优先级归一化) ...
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                 / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        """
        [辅助函数] "冷复制" (Cold Copy)
        将 "主" 网络的权重*复制*到 "目标" 网络。
        """
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """
        [辅助函数] 将所有组件移动到 GPU。
        """
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        """
        [辅助函数] 保存所有*可训练*的组件。
        """
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path)) # [关键] 保存优化器状态

    def load_models(self, path):
        """
        [辅助函数] 加载所有*可训练*的组件。
        """
        self.mac.load_models(path)
        # [关键] 目标网络*也*从主网络的文件加载，确保它们在*开始*时是一致的
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))