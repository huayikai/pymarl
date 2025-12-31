# 导入环境注册表
from envs import REGISTRY as env_REGISTRY
# 导入 functools.partial，用于创建一个“函数模板”
from functools import partial
# 导入核心数据容器
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:
    """
    这是一个“运行器”(Runner)，它的工作是：
    1. 在环境中运行*一个*完整的“回合”(Episode)。
    2. 收集所有 (s, a, r, o, ...) 数据。
    3. 将收集到的数据打包成一个 EpisodeBatch 并返回。

    注意：这个特定的运行器 (EpisodeRunner) *不支持*并行。
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # [关键] 这个 EpisodeRunner 强制要求 `batch_size_run` 必须为 1。
        # 它*不*支持并行环境。
        assert self.batch_size == 1

        # 从注册表中创建环境实例 (例如 StarCraft2Env)
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # 获取环境的回合最大步数
        self.episode_limit = self.env.episode_limit
        # t: 当前回合的内部时间步计数器
        self.t = 0

        # t_env: [关键] 全局环境总时间步计数器 (用于日志和测试)
        self.t_env = 0

        # 用于统计和日志记录的列表
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # 上次记录日志的时间 (用-1000000来强制在 t=0 时记录一次日志)
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        """
        这个 setup 方法被 `run_sequential` 调用，用于接收核心组件。
        """
        # [关键] partial(EpisodeBatch, ...) 创建了一个“函数模板”
        # 调用 self.new_batch() 就等同于调用 EpisodeBatch(scheme, groups, ...)
        # 这是一个创建新批次（“采集车”）的便捷方式
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        # 保存多智能体控制器 (MAC) 的实例
        self.mac = mac

    def get_env_info(self):
        # 返回环境信息 (n_agents, n_actions, state_shape 等)
        return self.env.get_env_info()

    def save_replay(self):
        # 保存回放
        self.env.save_replay()

    def close_env(self):
        # 关闭环境
        self.env.close()

    def reset(self):
        """
        重置一个新回合。
        """
        # [关键] 创建一个新的、空的 EpisodeBatch (即“采集车”)
        self.batch = self.new_batch()
        # 重置星际争霸2环境
        self.env.reset()
        # 重置回合内时间步计数器
        self.t = 0

    def run(self, test_mode=False):
        """
        [核心函数] 运行一个完整的、单独的回合。
        """
        # 1. 重置所有东西
        self.reset()

        terminated = False
        episode_return = 0
        # 初始化 MAC 的 RNN 隐藏状态 (batch_size 永远为 1)
        self.mac.init_hidden(batch_size=self.batch_size)

        # --- 回合主循环 ---
        # 只要回合没结束
        while not terminated:

            # [第 1 步: 收集“转换前”数据 (S_t, O_t, A_t^avail)]
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            # [第 2 步: 将数据 (state, obs) 存入“采集车”]
            self.batch.update(pre_transition_data, ts=self.t)

            # [第 3 步: MAC 选择动作]
            # 将*到目前为止*的所有数据 (self.batch) 传给 MAC (用于 RNN)
            # MAC 会根据 t_ep (当前步) 来获取数据
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # 将动作移回 CPU (准备存入 buffer 和 env.step)
            cpu_actions = actions.to("cpu").numpy()

            # [第 4 步: 在环境中执行动作]
            # actions[0] 因为 batch_size 恒为 1
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            # [第 5 步: 收集“转换后”数据 (A_t, R_t, T_t)]
            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                # (注意: "terminated" 在PyMARL中是指“非正常”终止，
                #  而 episode_limit 导致的终止不算 terminated)
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # [第 6 步: 将数据 (action, reward) 存入“采集车”]
            self.batch.update(post_transition_data, ts=self.t)

            # [第 7 步: 时间步 + 1]
            self.t += 1
        
        # --- 回合结束 (while 循环退出) ---

        # [第 8 步: 收集最后一步的 (S_t+1, O_t+1)]
        # (这对于计算 Q(s', a') 非常重要)
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # [第 9 步: 计算最后一步的动作 (A_t+1)]
        # 这对于 TD(lambda) 和 Q-Learning 计算目标 Q 值是必需的
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        cpu_actions = actions.to("cpu").numpy()
        # [第 10 步: 存储最后一步的动作]
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        # --- [日志记录] ---
        # 根据 test_mode 决定是更新训练统计还是测试统计
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # 更新统计字典 (例如 "battle_won")
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # [关键] 只有在*训练模式*下，才增加全局环境总步数
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # 如果是测试模式，并且测试回合数已满 (例如 32 个)，则记录日志
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        # 如果是训练模式，并且达到了日志间隔 (例如 10000 步)
        elif not test_mode and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix)
            # 额外记录 epsilon (探索率) 的值
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            # 重置日志计时器
            self.log_train_stats_t = self.t_env

        # [核心] 返回这个装满了的、完整的 EpisodeBatch (“采集车”)
        return self.batch

    def _log(self, returns, stats, prefix):
        """
        辅助函数：执行实际的日志记录 (发送给 Sacred 和 TensorBoard)
        """
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        # 清空列表，为下一个日志周期做准备
        returns.clear()

        # 记录所有其他统计数据 (如 battle_won_mean)
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        # 清空字典，为下一个日志周期做准备
        stats.clear()