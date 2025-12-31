# 导入环境注册表
from envs import REGISTRY as env_REGISTRY
# 导入 functools.partial，用于创建“函数模板”
from functools import partial
# 导入核心数据容器
from components.episode_buffer import EpisodeBatch
# [核心] 导入 Python 的多进程库
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# 基于 OpenAI Baselines 的 SubprocVecEnv
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    """
    [核心] 这是一个“并行运行器”(Parallel Runner)。
    它使用 Python 的 `multiprocessing` 库来一次性 *并行* 运行多个环境实例
    (例如 `batch_size_run = 8`)。
    这极大地加快了数据收集的速度。
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run # 并行环境的数量 (例如 8)

        # [关键] 创建“管道”(Pipe) 用于进程间通信
        # self.parent_conns: 主进程（Runner）持有
        # self.worker_conns: 子进程（EnvWorker）持有
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        
        # 获取环境的构造函数 (例如 StarCraft2Env)
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [] # 存储所有子进程的列表
        
        # [关键] 为每个并行环境创建一个 *独立的* 子进程 (Process)
        for i, worker_conn in enumerate(self.worker_conns):
            # target=env_worker: 告诉子进程去运行 `env_worker` 函数
            # args=(...): 传递子进程的连接端和环境构造函数
            ps = Process(target=env_worker, 
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        # 启动所有子进程
        for p in self.ps:
            p.daemon = True # 设置为守护进程 (主进程退出时自动退出)
            p.start()

        # [通信] 向*第一个*子进程发送 "get_env_info" 命令
        self.parent_conns[0].send(("get_env_info", None))
        # [通信] 接收*第一个*子进程返回的环境信息
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0 # 回合内的步数
        self.t_env = 0 # 全局总环境步数

        # 日志统计
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        """
        设置 Runner，与 EpisodeRunner 相同：
        1. 创建 EpisodeBatch 模板 (new_batch)
        2. 保存 MAC (控制器)
        """
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        # 保存蓝图信息，因为 reset 时需要
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        # [注意] ParallelRunner 不支持保存回放
        pass

    def close_env(self):
        # [通信] 向所有子进程发送 "close" 命令
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        """
        重置所有并行环境。
        """
        # 1. 创建一个新的、空的 EpisodeBatch (“采集车”)
        self.batch = self.new_batch()

        # 2. [通信] 向所有子进程发送 "reset" 命令
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        # 3. [通信] 收集所有子进程返回的*初始* (s, o, avail_a)
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        # 4. 将这 8 个环境的初始数据存入“采集车”的 t=0 步
        self.batch.update(pre_transition_data, ts=0)

        # 5. 重置回合内步数
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        """
        [核心函数] 并行地运行 `batch_size` 个回合。
        """
        # 1. 重置所有环境和“采集车”
        self.reset()

        all_terminated = False
        # [关键] 所有统计变量都从单个值变成了列表 (list)，长度为 8
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size) # 初始化 8 个隐藏状态
        terminated = [False for _ in range(self.batch_size)] # 跟踪每个环境是否终止
        # [关键] 跟踪*尚未*终止的环境的索引
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = [] 
        
        save_probs = getattr(self.args, "save_probs", False)
        
        # [核心循环] 只要*任何一个*环境还在运行
        while True:

            # [第 1 步: MAC 决策]
            # [关键] `bs=envs_not_terminated`
            # MAC 只为*尚未*终止的环境选择动作
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            cpu_actions = actions.to("cpu").numpy()

            # [第 2 步: 存储动作]
            # 将选择的动作 (actions) 存入“采集车”的第 t 步
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # [第 3 步: 向子进程发送动作]
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # 如果这个环境还活着
                    if not terminated[idx]: # (双重检查)
                        # [通信] 发送 "step" 命令和对应的动作
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            # [第 4 步: 检查是否所有环境都已终止]
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break # 如果都终止了，退出 while 循环

            # 准备接收子进程的返回数据
            post_transition_data = { "reward": [], "terminated": [] }
            pre_transition_data = { "state": [], "avail_actions": [], "obs": [] }

            # [第 5 步: 从子进程接收数据]
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]: # 只接收还“活着”的进程的
                    # [通信] [关键] `recv()` 是一个*阻塞*操作
                    # 主进程会在这里*等待*，直到子进程 `idx` 完成 env.step() 并发回数据
                    data = parent_conn.recv()
                    
                    # 收集 `t` 时刻的数据 (r, t)
                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1 # 增加全局步数

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"]) # 收集 battle_won 等信息
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True # 真实终止 (非时间耗尽)
                    terminated[idx] = data["terminated"] # 更新这个环境的终止状态
                    post_transition_data["terminated"].append((env_terminated,))

                    # 收集 `t+1` 时刻的数据 (s', o', avail_a')
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # [第 6 步: 存储数据]
            # 存储 (r, t) 到第 t 步
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 时间步 + 1
            self.t += 1

            # 存储 (s', o', avail_a') 到第 t+1 步
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            
        # --- `while` 循环结束 (所有环境都已终止) ---

        if not test_mode:
            self.t_env += self.env_steps_this_run # 更新全局步数

        # --- [日志记录] ---
        # (这部分逻辑与 EpisodeRunner 几乎相同，只是现在处理的是 8 个回合的数据)
        
        # 收集 `env.get_stats()`
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        # 合并 8 个环境的统计数据
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # [核心] 返回这个装满了 8 个回合数据的 EpisodeBatch
        return self.batch

    def _log(self, returns, stats, prefix):
        # (日志记录逻辑，与 EpisodeRunner 相同)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    """
    [核心] 这是*在子进程中运行*的函数。
    它就像一个“远程遥控机器人”，等待主进程的指令。
    """
    # 1. 在这个*新*的进程中创建环境实例
    env = env_fn.x()
    
    # 2. 永久循环，等待命令
    while True:
        # 3. [通信] 阻塞并等待主进程发来命令
        cmd, data = remote.recv()

        if cmd == "step":
            # 4. 执行一步
            actions = data
            reward, terminated, env_info = env.step(actions)
            # 5. 获取新数据
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            # 6. [通信] 将*所有*结果打包发回给主进程
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            # [通信] 发回初始数据
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            # [通信] 关闭环境和连接，退出循环
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            # [通信] 发回环境信息
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            # [通信] 发回统计信息 (如 battle_won)
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    一个辅助工具类。
    Python 的 `multiprocessing` 默认使用 `pickle` 来序列化进程间传递的数据。
    `pickle` 无法处理一些复杂的 Python 对象（如 `partial` 函数）。
    `cloudpickle` 是一个更强大的序列化库。
    这个类“欺骗” `multiprocessing`，让它在序列化时使用 `cloudpickle`。
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)