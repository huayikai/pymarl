import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN  # 一个方便的工具，可以用 S.arg 代替 S['arg'] 来访问字典
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

# --- 导入你之前看过的所有“注册表”(REGISTRY) ---
from learners import REGISTRY as le_REGISTRY      # 学习器 (Learner) 注册表
from runners import REGISTRY as r_REGISTRY      # 运行器 (Runner) 注册表
from controllers import REGISTRY as mac_REGISTRY  # 控制器 (MAC) 注册表
# --- 导入你之前看过的核心组件 ---
from components.episode_buffer import ReplayBuffer # 经验回放缓冲区
from components.transforms import OneHot           # One-Hot 转换器

from smac.env import StarCraft2Env # 导入星际争霸2环境

def get_agent_own_state_size(env_args):
    """
    (这是一个辅助函数，仅用于 qatten 算法)
    创建一个临时的 SC2 环境实例，只是为了获取一些环境参数。
    """
    sc_env = StarCraft2Env(**env_args)
    # qatten 参数设置 (只在 qatten 中使用)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):
    """
    这是由 Sacred (@ex.main) 调用的*第一个*函数 (在你的代码中它被命名为 my_main)。
    它的工作是：
    1. 检查和设置配置。
    2. 设置日志记录器 (Logger 和 TensorBoard)。
    3. 调用 `run_sequential` 来 *真正* 开始训练。
    4. 训练结束后，清理所有线程并退出。
    """

    # 1. 检查配置参数是否合理 (例如，如果 use_cuda=True 但没 GPU，就设为 False)
    _config = args_sanity_check(_config, _log)

    # 2. 将 Sacred 的配置字典 `_config` 转换为更易于访问的 SimpleNamespace 对象 `args`
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # 3. 设置日志记录器
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                         indent=4,
                                         width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 4. [关键] 配置 TensorBoard 日志记录器
    # 创建一个唯一的代币 (token)，用于命名 TensorBoard 日志文件夹
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        # 设置 tb_logs 的路径
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc) # 初始化 TensorBoard 写入器

    # 5. 设置 Sacred 日志记录器（用于记录 loss, return 等标量）
    logger.setup_sacred(_run)

    # 6. [核心] 调用下面的 run_sequential 函数，开始运行和训练
    run_sequential(args=args, logger=logger)

    # --- 训练结束后的清理工作 ---
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # 确保框架真正退出
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    """
    一个辅助函数，用于执行“测试/评估”。
    """
    for _ in range(args.test_nepisode): # 运行 N 个测试回合 (例如 32 个)
        runner.run(test_mode=True) # [关键] 以 test_mode=True 模式运行

    if args.save_replay: # 如果设置了，保存回放
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    """
    这是*真正*的“总指挥”。
    它负责：
    1.  **实例化**所有核心组件 (Runner, Buffer, MAC, Learner)。
    2.  **加载**模型（如果有 checkpoint）。
    3.  **运行**主训练循环 (Main Training Loop)。
    """

    # 1. [初始化 运行器 (Runner)]
    #    - Runner 负责与环境交互、收集数据。
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 2. [获取环境信息]
    #    - 初始化 Runner (和环境) 后，我们可以获取环境的详细信息。
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]   # 智能体数量 (例如 3m 地图为 3)
    args.n_actions = env_info["n_actions"] # 动作空间大小 (例如 3m 地图为 9)
    args.state_shape = env_info["state_shape"] # 全局状态的维度
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # 3. [定义数据蓝图 (Scheme)]
    #    - 这是你之前问过的 `EpisodeBatch` 的“蓝图”。
    #    - 它定义了缓冲区需要存储哪些数据，以及它们的形状和类型。
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"}, # 局部观测
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long}, # 动作
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int}, # 可用动作
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float}, # 动作概率
        "reward": {"vshape": (1,)}, # 奖励
        "terminated": {"vshape": (1,), "dtype": th.uint8}, # 终止信号
    }
    groups = {
        "agents": args.n_agents # 定义 "agents" 组有几个成员
    }
    preprocess = {
        # [关键] 告诉缓冲区：当 "actions" 数据进来时，
        # 自动使用 OneHot 转换器，并将结果存为 "actions_onehot"
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 4. [初始化 经验回放缓冲区 (ReplayBuffer)]
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
    # 5. [初始化 多智能体控制器 (MAC)]
    #    - MAC 负责根据观测选择动作 (你之前看过的 BasicMAC, n_mac 等)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 6. [设置 Runner]
    #    - 告诉 Runner：你将使用这个 MAC 来选择动作，
    #      并且你收集的数据必须符合这个 scheme 蓝图。
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 7. [初始化 学习器 (Learner)]
    #    - Learner 负责从 Buffer 采样、计算损失、更新网络。
    #    - (这就是你上一个问题看的 __init__ 方法！)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda() # 将 Learner 的所有组件 (mac, mixer, target_mac...) 移动到 GPU

    # 8. [加载模型 (Checkpoint Loading)]
    #    - 如果在配置中指定了 checkpoint_path，这里会加载已训练的模型。
    if args.checkpoint_path != "":
        # ... (寻找最新或最接近的 checkpoint) ...
        timestep_to_load = ... # (省略的加载逻辑)
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path) # [关键] Learner 负责加载所有网络权重
        runner.t_env = timestep_to_load # [关键] 将环境总时间步数重置为加载的步数

        if args.evaluate or args.save_replay: # 如果只是为了评估或保存回放
            evaluate_sequential(args, runner)
            return

    # --- [主训练循环 (Main Training Loop)] ---
    episode = 0
    last_test_T = -args.test_interval - 1 # 上次测试的时间，设置为负的可以让训练开始的时候立即进行一次测试
    last_log_T = 0 # 上次打印日志的时间
    model_save_time = 0 # 上次保存模型的时间

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # [循环开始] 只要环境总步数 (t_env) 还没达到 t_max
    while runner.t_env <= args.t_max:

        # --- [第 1 步: 数据收集 (Rollout)] ---
        # (在 `with th.no_grad()` 中，因为这只是前向传播，不需要计算梯度)
        with th.no_grad():
            # [关键] Runner 运行一个完整的“工作单元” (例如一个回合)
            episode_batch = runner.run(test_mode=False)
            # 将收集到的数据 (一个 EpisodeBatch) 存入回放池
            buffer.insert_episode_batch(episode_batch)

        # --- [第 2 步: 训练 (Training)] ---
        # 检查缓冲区中的数据是否已*足够*进行一次采样
        if buffer.can_sample(args.batch_size):
            # (这个 if 块是用于梯度累积的，可以暂时忽略)
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            # [关键] 从回放池中 *随机采样* 一个批次的数据
            episode_sample = buffer.sample(args.batch_size) # (例如 32 个回合)

            # (数据预处理：裁剪掉未填充的“未来”步骤)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            # (如果数据在 CPU 上，将其移动到 GPU)
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # [关键] 执行一次训练！
            # 将采样的数据交给 Learner，Learner 会计算损失、反向传播、更新网络
            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample # 释放内存

        # --- [第 3 步: 测试 (Testing)] ---
        # 检查是否达到了测试的间隔时间
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            # ... (打印预估剩余时间) ...
            last_time = time.time()

            last_test_T = runner.t_env # 重置测试计时器
            
            # [关键] Runner 切换到 test_mode=True，运行 N 个测试回合
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # --- [第 4 步: 保存模型 (Saving)] ---
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            # ... (创建保存路径) ...
            logger.console_logger.info("Saving models to {}".format(save_path))
            
            # [关键] 告诉 Learner 保存所有模型 (MAC, Mixer, 优化器状态等)
            learner.save_models(save_path)

        # 增加回合计数
        episode += args.batch_size_run

        # --- [第 5 步: 记录日志 (Logging)] ---
        if (runner.t_env - last_log_T) >= args.log_interval:
            # [关键] 告诉 Logger (Sacred / TensorBoard) 记录最新的统计数据
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats() # 打印到控制台 (你之前看到的日志)
            last_log_T = runner.t_env
    
    # [循环结束]
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    """
    一个辅助函数，用于在实验开始前检查和修正配置。
    """

    # 检查 CUDA 是否可用
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    # (确保测试回合数是并行运行数的整数倍，以便高效测试)
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config