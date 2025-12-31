import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
from utils.th_utils import get_parameters_num
from modules.mixers.qatten import QattenMixer
from torch.optim import Adam

# --- ResQ 核心辅助函数 ---
# 根据条件生成掩码 w_r (Mask)
# resq_version == "v3": 对应论文中的掩码逻辑
# 如果是当前最优动作(condition为True)，ws=0 (关闭残差)
# 如果是非最优动作(condition为False)，ws=1 (开启残差)
def get_ws(resq_version, condition, td_error):
    if resq_version == "v3":
        ws = th.where(condition, th.zeros_like(td_error), th.ones_like(td_error)) 
    return ws

class RestQLearnerCentral:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        # --- 1. 初始化主网络流 (Main Stream) ---
        # 对应论文中的 Q_tot，负责 IGM 和最优动作选择
        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # 目标主网络
        self.target_mac = copy.deepcopy(mac)

        # --- 2. 初始化残差网络流 (Residual Stream) ---
        # 对应论文中的 Q_res，负责修补非单调数值
        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        args.is_res_mixer = True;  # added 20220502
        if "rest_mixer" in vars(args):
            # (根据参数选择残差混合器类型，通常也是 QMixer 或 VDN)
            if args.rest_mixer == "vdn":
                self.rest_mixer = VDNMixer()
            elif args.rest_mixer == "qmix":
                self.rest_mixer = QMixer(args)
            elif args.rest_mixer == "qatten":
                self.rest_mixer = QattenMixer(args)
            else:
                self.rest_mixer = QMixerCentralFF(args)
        else:
            self.rest_mixer = QMixerCentralFF(args)

        # --- 3. 初始化中心化网络流 (Central/Unconstrained Stream) ---
        # 对应 WQMIX 中的 Q* 或 ResQ 中用于辅助训练的无限制网络
        # 它不受单调性限制，提供更准确的全局 Target
        args.is_res_mixer = False
        if self.args.central_mixer in ["ff", "atten", "vdn"]:
            if self.args.central_loss == 0:
                # 如果不计算中心化 Loss，就复用主网络（退化情况）
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # 前馈网络，无单调性限制
                elif self.args.central_mixer == "vdn":
                    self.central_mixer = VDNMixer()
                else:
                    raise Exception("Error with central_mixer")

                assert args.central_mac == "basic_central_mac"
                self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) 
                self.target_central_mac = copy.deepcopy(self.central_mac)
                self.params += list(self.central_mac.parameters())

                # 初始化残差网络的 MAC (Multi-Agent Controller)
                self.rest_mac = copy.deepcopy(self.mac) # added for RESTQ
                self.rest_target_mac = copy.deepcopy(self.rest_mac)# added for RESTQ
                self.params += list(self.rest_mac.parameters())# added for RESTQ
        else:
            raise Exception("Error with qCentral")
        
        # 将所有网络的参数加入优化器列表
        self.params += list(self.central_mixer.parameters())
        self.params += list(self.rest_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.rest_target_mixer = copy.deepcopy(self.rest_mixer) # added for RESTQ
        
        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters()) + list(self.rest_mixer.parameters())))

        # 初始化优化器
        if hasattr(self, "optimizer"):
            if getattr(self, "optimizer") == "Adam":
                self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取 Batch 数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # --- A. 前向传播：主网络 (Main Stream) ---
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [batch, time, agents, actions]
        # 提取当前动作对应的 Q 值 (Q_i)
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) 
        chosen_action_qvals = chosen_action_qvals_agents

        # --- B. 前向传播：残差网络 (Residual Stream) ---
        rest_mac_out = []
        self.rest_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_agent_outs = self.rest_mac.forward(batch, t=t)
            rest_mac_out.append(rest_agent_outs)
        rest_mac_out = th.stack(rest_mac_out, dim=1) 
        # 提取残差 Q 值
        rest_chosen_action_qvals = th.gather(rest_mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- C. 计算目标值：主网络 Target ---
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[:], dim=1) 
        target_mac_out[avail_actions[:, :] == 0] = -9999999 # Mask unavailable

        # --- D. 计算目标值：残差网络 Target ---
        rest_target_mac_out = []
        self.rest_target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            rest_target_agent_outs = self.rest_target_mac.forward(batch, t=t)
            rest_target_mac_out.append(rest_target_agent_outs)
        rest_target_mac_out = th.stack(rest_target_mac_out[:], dim=1) 
        rest_target_mac_out[avail_actions[:, :] == 0] = -9999999 

        # --- E. Double DQN 目标动作选择 ---
        if self.args.double_q:
            # 使用 Main Policy (主网络) 来选择最优动作
            mac_out_detach = mac_out.clone().detach() 
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            # 主网络 Target Q
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)

            # 残差网络 Target Q
            # [关键点]：残差网络必须使用与主网络相同的 argmax 动作！
            # 这保证了 IGM 原则，即最优动作完全由 Q_tot 决定。
            # 就是说先让主网络选动作，然后用target网络计算Q值，同时残差网络用这个动作计算残差
            rest_mac_out_detach = rest_mac_out.clone().detach()
            rest_mac_out_detach[avail_actions == 0] = -9999999
            rest_cur_max_action_targets, rest_cur_max_actions = cur_max_action_targets, cur_max_actions 
            rest_target_max_agent_qvals = th.gather(rest_target_mac_out[:,:], 3, rest_cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")

        # --- F. 前向传播：中心化网络 (Central Stream) ---
        # 用于计算更准确的 Target，指导 Main 和 Residual 训练
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1) 
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3) 

        # --- G. 计算中心化网络 Target ---
        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1) 
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999 
        # 使用 Main Policy 的最优动作
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)

        # --- H. 混合 Q 值 (Mixing Phase) ---
        # 1. 计算 Q_tot (主单调网络)
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        # 2. 计算中心化 Target
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])
        
        # 3. 计算 Q_res (残差网络)
        rest_chosen_action_qvals_ = rest_chosen_action_qvals.unsqueeze(3).repeat(1,1,1,self.args.central_action_embed).squeeze(3)
        Q_r = rest_chosen_action_qvals = self.rest_mixer(rest_chosen_action_qvals_, batch["state"][:,:-1]) # added for RESTQ
        
        # [关键点] ResQ 约束：强制残差 Q_r <= 0
        negative_abs = getattr(self.args, 'residual_negative_abs', False)
        if negative_abs:
            Q_r = - Q_r.abs()

        # 计算 TD Targets (使用中心化网络的 Target)
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # 计算 TD Error (初步计算，用于生成掩码)
        """should clean this 4 lines"""
        td_error = (chosen_action_qvals - (targets.detach()))
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        """should clean this 4 lines"""

        # --- I. 训练中心化网络 (Central Loss) ---
        # 让中心化网络拟合真实回报，作为 Ground Truth
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1]) 
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # --- J. 计算掩码 (Masking / Weighting) ---
        if self.args.hysteretic_qmix: # OW-QMIX 逻辑 (Optimistic Weighting)
            w_r = th.where(td_error < 0, th.ones_like(td_error)*1, th.zeros_like(td_error)) 
            w_to_use = w_r.mean().item() 
        else: # CW-QMIX 或 ResQ 逻辑
            # 判断当前动作是否是主网络认为的最优动作
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            
            # 设置条件：如果是最优动作
            if self.args.condition == "max_action":
                condition = is_max_action
            elif self.args.condition == "max_larger":
                condition = is_max_action | qtot_larger
            
            # [关键点] ResQ 的掩码计算
            # 如果不使用 mask (nomask=True)，则一直加上残差
            # 否则根据 ResQ 版本计算 w_r (通常：最优动作 w_r=0, 非最优 w_r=1)
            nomask = getattr(self.args, 'nomask', False)
            if nomask:
                w_r = th.ones_like(td_error);
            else:
                w_r = get_ws(self.args.resq_version, condition, td_error)
            w_to_use = w_r.mean().item() 

        # --- K. 计算 ResQ 的 TD Error ---
        # 公式：Q_jt = Q_tot + w_r * Q_res
        # Loss = ( (Q_tot + w_r * Q_res) - y )^2
        td_error = (chosen_action_qvals + w_r.detach() * rest_chosen_action_qvals - (targets.detach()))
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask       
        qmix_loss = (masked_td_error ** 2).sum() / mask.sum()

        # --- L. 约束 Loss (Constraint Loss) ---
        noopt_loss2 = None
        if self.args.resq_version in ["v3"]:
            # 惩罚 Q_r > 0 的部分，确保 Q_r 是负的
            # 如果前面已经使用了 .abs() 强制取负，这个 loss 理论上为 0
            Q_r_ = th.max(Q_r, th.zeros_like(Q_r))
            noopt_loss1 = (((Q_r_ * mask) ** 2).sum()) / mask.sum()
            noopt_loss = noopt_loss1 
            
            # 总 Loss = 主Loss + 中心化Loss + 约束Loss
            loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + self.args.noopt_loss * noopt_loss
        
        # --- M. 优化更新 ---
        self.optimiser.zero_grad()
        loss.backward()

        # 记录梯度范数用于 Log
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item() ** 2
        agent_norm = agent_norm ** (1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        # 更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 记录日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            if noopt_loss is not None:
                self.logger.log_stat("noopt_loss", noopt_loss.item(), t_env)
            if noopt_loss1 is not None:
                self.logger.log_stat("noopt_loss1", noopt_loss1.item(), t_env)
            if noopt_loss2 is not None:
                self.logger.log_stat("noopt_loss2", noopt_loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.rest_target_mac.load_state(self.rest_mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.rest_target_mixer.load_state_dict(self.rest_mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.rest_mac.cuda()
        self.rest_target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.rest_mixer.cuda()
            self.rest_target_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    # TODO: Model saving/loading is out of date!
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