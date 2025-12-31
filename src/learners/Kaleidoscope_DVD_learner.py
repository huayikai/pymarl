import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.dvd import DVDMixer # [引用 DVD Mixer]
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn as nn
import numpy as np
import math

class KaleidoscopeDVDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        self.params = list(mac.parameters())

        # --- [1. Mixer 初始化: 兼容 DVD] ---
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "dvd": # 新增 DVD 支持
            self.mixer = DVDMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
            
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # --- [2. Kaleidoscope 初始化] ---
        self.last_reset_t = 0

        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
            
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # --- [Step 1: 准备数据] ---
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # --- [Step 2: 主网络前向传播 (带 Hidden States 收集)] ---
        self.mac.agent.train()
        mac_out = []
        hidden_states_list = [] # [DVD] 用于收集隐藏状态
        
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # [DVD] 收集每一步的 hidden_states，并 clone 防止梯度问题
            # 注意: 这里假设 mac.hidden_states 形状需要 reshape
            cur_h = self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
            hidden_states_list.append(cur_h)
            
        mac_out = th.stack(mac_out, dim=1)
        
        # [DVD] 处理 Hidden States
        whole_hidden_states = th.stack(hidden_states_list, dim=1)
        hidden_states_main = whole_hidden_states[:, :-1] # 去掉最后一步 (用于 Q_tot 计算)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- [Step 3: 目标网络前向传播] ---
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_hidden_states_list = [] # [DVD]
            
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                
                # [DVD] Target Hidden States
                cur_target_h = self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
                target_hidden_states_list.append(cur_target_h)

            target_mac_out = th.stack(target_mac_out, dim=1)
            # [DVD]
            whole_target_hidden_states = th.stack(target_hidden_states_list, dim=1)

            # Double Q-Learning Logic
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # [Step 3.1: Target Mixer 计算 (兼容 DVD)]
            if self.args.mixer == "dvd":
                # DVD 需要传入 hidden states
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], whole_target_hidden_states)
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            # 计算 Targets (TD-Lambda)
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                if self.args.mixer == "dvd":
                     qvals = self.target_mixer(qvals, batch["state"], whole_target_hidden_states)
                else:
                     qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # --- [Step 4: 计算 Loss] ---
        
        # [Step 4.1: Main Mixer 计算 (兼容 DVD)]
        if self.args.mixer == "dvd":
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], hidden_states_main)
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # 1. 基础 TD Loss
        loss_td = masked_td_error.sum() / mask.sum()

        # 2. [Kaleidoscope] 多样性正则化 Loss
        loss_div = self.calculate_diversity_loss()
        
        # 3. [Kaleidoscope] 自适应系数 Beta
        if loss_div > 1e-6:
            scaling_factor = (loss_td.detach() / (loss_div.detach() + 1e-6)) * getattr(self.args, 'diversity_beta', 0.1)
        else:
            scaling_factor = 0.0

        # 4. 总 Loss (最大化差异 = 最小化 -Diversity)
        loss = loss_td - scaling_factor * loss_div

        # --- [Step 5: 优化] ---
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # --- [Step 6: Kaleidoscope 周期性重置] ---
        if (t_env - self.last_reset_t) >= getattr(self.args, 'reset_interval', 100000):
            self.reset_dead_parameters()
            self.last_reset_t = t_env
            self.logger.console_logger.info(f"Kaleidoscope: Reset dead parameters at t_env={t_env}")

        # --- [Step 7: 更新 Target 网络] ---
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # --- [Step 8: 日志] ---
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_div", loss_div.item(), t_env) # Kaleidoscope
            self.logger.log_stat("beta_adaptive", scaling_factor.item() if isinstance(scaling_factor, th.Tensor) else scaling_factor, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        info = {}
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                       / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    # =========================================================================
    # [Kaleidoscope] 辅助函数 (保持不变)
    # =========================================================================

    def calculate_diversity_loss(self):
        """计算论文公式 (8) 和 (16): || theta_0 * (M_i - M_j) ||_1"""
        loss_div = 0
        agent = self.mac.agent
        
        try:
            masks_list = agent.get_all_masks()
            weights_list = agent.get_shared_weights()
        except AttributeError:
            return th.tensor(0.0).to(self.device)

        layer_weights = [1.0, 2.0] 
        
        for layer_idx, (masks, weight_shared) in enumerate(zip(masks_list, weights_list)):
            n_agents = masks.shape[0]
            w_l = layer_weights[layer_idx] if layer_idx < len(layer_weights) else 1.0
            weight_abs = th.abs(weight_shared).detach()
            
            masks_i = masks.unsqueeze(1) 
            masks_j = masks.unsqueeze(0)
            diff_matrix = th.abs(masks_i - masks_j)
            weighted_diff = (diff_matrix * weight_abs).sum()
            loss_div += w_l * weighted_diff
            
        return loss_div

    def reset_dead_parameters(self):
        """实现论文 Section 3.3: 重置那些在所有 Agent 中都被 Mask 掉的参数"""
        agent = self.mac.agent
        # 需要确保 agent 有 fc1/fc2 属性，或者通过 modules 遍历
        masked_layers = []
        if hasattr(agent, "fc1"): masked_layers.append(agent.fc1)
        if hasattr(agent, "fc2"): masked_layers.append(agent.fc2)

        reset_prob = getattr(self.args, 'reset_prob', 0.1)
        
        with th.no_grad():
            for layer in masked_layers:
                if not hasattr(layer, 'get_masks'): continue

                masks = layer.get_masks()
                active_counts = masks.sum(dim=0)
                dead_mask = (active_counts == 0)
                
                if dead_mask.sum() == 0:
                    continue
                
                should_reset = (th.rand_like(layer.weight) < reset_prob) & dead_mask
                
                new_weights = th.empty_like(layer.weight)
                nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
                layer.weight.data[should_reset] = new_weights[should_reset]
                
                for i in range(layer.n_agents):
                    layer.threshold_params.data[i][should_reset] = -5.0

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