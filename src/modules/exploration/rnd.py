import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNDModel(nn.Module):
    def __init__(self, args):
        super(RNDModel, self).__init__()
        self.args = args
        input_shape = args.state_shape # RND 输入通常是全局状态 s_t 
        
        # Predictor network: 可训练，用于预测 Target 的输出
        self.predictor = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # 输出特征维度
        )
        
        # Target network: 固定，随机初始化
        self.target = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 冻结 Target 网络的参数，不进行更新 
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, states):
        """
        输入: states (batch, seq_len, state_dim)
        输出: intrinsic_rewards, predictor_loss
        """
        # 计算 Target 和 Predictor 的输出
        target_feature = self.target(states)
        predict_feature = self.predictor(states)
        
        # 预测误差即为内在奖励 (Intrinsic Reward)
        # r_int = || hat_g(s) - g(s) ||^2
        prediction_error = (predict_feature - target_feature).pow(2).sum(dim=-1, keepdim=True)
        
        return prediction_error