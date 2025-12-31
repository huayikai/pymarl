import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn as nn
import numpy as np
import math

class KaleidoscopeLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
            
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # [Kaleidoscope] 初始化重置计时器
        self.last_reset_t = 0

        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
            
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # --- [第 1 步: 准备数据] ---
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # --- [第 2 步: 计算 Q_tot] ---
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- [第 3 步: 计算 Targets] ---
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # --- [第 4 步: 计算 Loss] ---
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # 原始 TD Loss
        loss_td = masked_td_error.sum() / mask.sum()

        # --- [Kaleidoscope 核心 1: 多样性正则化] ---
        # 计算多样性 Loss: sum || theta * (M_i - M_j) ||
        loss_div = self.calculate_diversity_loss()
        
        # 计算自适应系数 beta (论文 Appendix A.1.1)
        # beta^d = (|L_td| / |J_div|) * beta
        # 使用 .item() 或 .detach() 避免梯度回传到系数计算
        if loss_div > 1e-6:
            scaling_factor = (loss_td.detach() / (loss_div.detach() + 1e-6)) * getattr(self.args, 'diversity_beta', 0.1)
        else:
            scaling_factor = 0.0

        # [cite_start]总 Loss = TD Loss - beta * Diversity (因为要最大化多样性) [cite: 614]
        loss = loss_td - scaling_factor * loss_div

        # --- [第 5 步: 优化] ---
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # --- [Kaleidoscope 核心 2: 周期性重置] ---
        # [cite_start]论文 Section 3.3 [cite: 165]
        if (t_env - self.last_reset_t) >= getattr(self.args, 'reset_interval', 100000):
            self.reset_dead_parameters()
            self.last_reset_t = t_env
            self.logger.console_logger.info(f"Kaleidoscope: Reset dead parameters at t_env={t_env}")

        # --- [第 6 步: 更新目标网络] ---
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # --- [第 7 步: 日志] ---
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_div", loss_div.item(), t_env) # 记录多样性 Loss
            self.logger.log_stat("beta_adaptive", scaling_factor.item() if isinstance(scaling_factor, th.Tensor) else scaling_factor, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # --- [第 8 步: PER 返回] ---
        info = {}
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                       / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    # =========================================================================
    # [Kaleidoscope] 辅助函数
    # =========================================================================

    def calculate_diversity_loss(self):
        """
        计算论文公式 (8) 和 (16): || theta_0 * (M_i - M_j) ||_1
        """
        loss_div = 0
        agent = self.mac.agent
        
        # 假设 agent 提供了这两个接口 (需要修改 Agent 代码实现)
        try:
            masks_list = agent.get_all_masks()        # [Mask_fc1, Mask_fc2]
            weights_list = agent.get_shared_weights() # [Weight_fc1, Weight_fc2]
        except AttributeError:
            # 如果 Agent 不是 Kaleidoscope Agent，返回 0
            return th.tensor(0.0).to(self.device)

        # [cite_start]论文 Appendix A.1.1: 越靠近输出层权重越大 (w_l = 2^l) [cite: 595]
        # 假设网络结构是 FC1 -> GRU -> FC2，这里只对 FC 层做了 Mask
        layer_weights = [1.0, 2.0] 
        
        for layer_idx, (masks, weight_shared) in enumerate(zip(masks_list, weights_list)):
            # masks shape: (n_agents, out_dim, in_dim)
            n_agents = masks.shape[0]
            w_l = layer_weights[layer_idx] if layer_idx < len(layer_weights) else 1.0
            
            weight_abs = th.abs(weight_shared).detach() # 停止 theta_0 的梯度 [cite: 591]
            
            layer_div_sum = 0
            # 计算成对差异 (Sum over i, j where i != j)
            # 为了效率，可以使用广播机制而不是双重循环
            # masks: (N, Out, In) -> (N, 1, Out, In) 和 (1, N, Out, In)
            masks_i = masks.unsqueeze(1) 
            masks_j = masks.unsqueeze(0)
            
            # diff: (N, N, Out, In)
            diff_matrix = th.abs(masks_i - masks_j)
            
            # 加权差异: || theta * diff ||_1
            # (N, N, Out, In) * (Out, In) -> sum
            weighted_diff = (diff_matrix * weight_abs).sum()
            
            loss_div += w_l * weighted_diff
            
        return loss_div

    def reset_dead_parameters(self):
        """
        [cite_start]实现论文 Section 3.3: 重置那些在所有 Agent 中都被 Mask 掉的参数 [cite: 165]
        """
        agent = self.mac.agent
        
        # 获取所有支持 Mask 的层 (MaskedLinear)
        # 假设 agent.layers 是一个列表，或者手动指定
        masked_layers = [agent.fc1, agent.fc2] 
        
        reset_prob = getattr(self.args, 'reset_prob', 0.1)
        
        with th.no_grad():
            for layer in masked_layers:
                if not hasattr(layer, 'get_masks'): continue

                # 1. 获取 Mask: (n_agents, out, in)
                masks = layer.get_masks()
                
                # 2. 查找 "Dead Weights" (所有 agent 的 mask 都是 0)
                # sum(dim=0) == 0 表示没有任何一个 agent 激活了这个权重
                active_counts = masks.sum(dim=0)
                dead_mask = (active_counts == 0) # Boolean tensor
                
                if dead_mask.sum() == 0:
                    continue
                
                # 3. 生成重置选择矩阵 (仅针对 Dead Weights 以概率 p 重置)
                # mask: (out, in)
                should_reset = (th.rand_like(layer.weight) < reset_prob) & dead_mask
                
                # 4. 重置 theta_0 (Weight)
                # 使用 Kaiming 初始化生成新权重
                new_weights = th.empty_like(layer.weight)
                nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
                layer.weight.data[should_reset] = new_weights[should_reset]
                
                # 5. 重置 Thresholds s_i
                # 将 s_i 重置为一个较小的值 (e.g., -5), 使得 sigmoid(s) 接近 0, 从而 mask 变为 1 (重新激活)
                # 对所有 agent 的对应位置都进行重置
                for i in range(layer.n_agents):
                    layer.threshold_params.data[i][should_reset] = -5.0

    # =========================================================================

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