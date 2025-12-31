import torch as th
import numpy as np
from types import SimpleNamespace as SN # 导入 SN，一个方便的数据结构，允许使用 . 语法访问字典键
from .segment_tree import SumSegmentTree, MinSegmentTree # 导入用于 PER 的线段树
import random

class EpisodeBatch:
    """
    [核心数据结构] - 回合批次容器
    
    这是一个高性能的数据容器，用于存储一批 (batch) 回合 (episode) 数据。
    它不使用列表追加 (append)，而是 *预先分配* (pre-allocate) 
    一个固定大小的 PyTorch 零张量，然后将数据“填充”进去。
    
    它也是 ReplayBuffer (经验回放) 的 *父类*。
    """
    def __init__(self,
                 scheme,        # scheme: "蓝图"，一个字典，定义了要存储哪些数据 (key) 及其形状/类型 (value)
                 groups,        # groups: "分组"，一个字典，定义了哪些数据是按智能体分组的 (例如 "agents": 8)
                 batch_size,    # batch_size: 此批次包含多少个回合
                 max_seq_length, # max_seq_length: 每个回合的最大时间步长度
                 data=None,       # data: (可选) 用于从现有数据创建"视图" (例如在切片时)
                 preprocess=None, # preprocess: (可选) 一个字典，定义了数据预处理管道 (例如 OneHot 编码)
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            # 如果 data *不是* None，意味着我们正在从现有数据创建一个"视图" (View)
            # (这主要发生在切片 `__getitem__` 操作时)
            self.data = data
        else:
            # [核心] 如果 data 是 None，意味着我们要 *初始化* 一个全新的、*空*的批次
            # (这是创建 ReplayBuffer 或 Runner 临时批次时的标准流程)
            self.data = SN() # SimpleNamespace()
            self.data.transition_data = {} # 存储“逐帧”数据 (s, a, r, o...)
            self.data.episode_data = {}    # 存储“全局”数据 (只在 t=0 存储一次的数据)
            # [关键] 调用 _setup_data 来 *预先分配* 全零张量
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        """
        [核心] 预分配 (Pre-allocate) 全零张量。
        这是这个类高性能的关键。
        """
        # --- 1. 处理预处理 (Preprocess) ---
        # 如果定义了 preprocess (例如 "actions" -> "actions_onehot")
        # 它会*更新* scheme 蓝图，为 "actions_onehot" 这样的新字段也预分配空间
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0] # 拿到新名字: "actions_onehot"
                transforms = preprocess[k][1] # 拿到转换器: [OneHot(n_actions=6)]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                # 推断转换后的形状和类型
                # 这里没有进行操作，就是在问，如果我输入了这个东西，我输出的是什么shape和什么type
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # 将新字段 (例如 "actions_onehot") 添加到蓝图中
                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        # --- 2. 添加 "filled" 掩码 ---
        # "filled" 是一个特殊的掩码 (mask)，用于标记哪些时间步是真实数据 (1)，哪些是空白填充 (0)
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        # --- 3. 遍历蓝图 (scheme)，创建张量 ---
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False) # 是否是“全局”数据？如果是，就只需要存储一次
            group = field_info.get("group", None) # 是否属于某个组 (例如 "agents")？如果是，就需要填充n_agents次
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,) # 确保 vshape 是一个元组 (tuple)

            if group:
                # 如果属于 "agents" 组 (n_agents=8)，则在形状中添加一个维度
                assert group in groups
                shape = (groups[group], *vshape) # e.g., (8, obs_shape)
            else:
                shape = vshape

            if episode_const:
                # 如果是“全局”数据 (只存一次)
                # 形状: (batch_size, *shape)
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                # 如果是“逐帧”数据 (每个时间步都存)
                # 形状: (batch_size, max_seq_length, *shape)
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        # 允许在创建后，向批次中添加新的字段
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        # [辅助函数] 将此批次中的*所有*张量移动到指定设备 (例如 "cuda")
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        """
        [核心] 将真实数据“填充”到预先分配好的全零张量中。
        """
        slices = self._parse_slices((bs, ts)) # 解析切片
        for k, v in data.items():
            # 如果是动态的数据
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    # [关键] 自动将 "filled" 掩码的对应位置设为 1
                    target["filled"][slices] = 1
                    mark_filled = False # 只在一个字段上标记一次
                _slices = slices
            # 如果是静态的数据
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device) # 将数据 (numpy/list) 转为 Tensor
            self._check_safe_view(v, target[k][_slices])
            
            # [关键] 将数据 v "填充" 到目标张量 `target` 的 `_slices` 位置
            target[k][_slices] = v.view_as(target[k][_slices]) # 保证维度匹配

            # [关键] 如果此字段 (k) 有预处理指令
            if k in self.preprocess:
                new_k = self.preprocess[k][0]       # e.g., "actions_onehot"
                v = target[k][_slices]             # e.g., 整数 5
                for transform in self.preprocess[k][1]: # e.g., OneHot()
                    v = transform.transform(v)     # e.g., [0,0,0,0,1,...]
                # 将转换后的数据存入新字段
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        # (辅助函数) 确保 reshape 是安全的，不会意外地混淆数据
        # 防止维度从[5,4]变成[4,5]
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        """
        [核心] 强大的切片功能。
        这使得 `EpisodeBatch` 可以像 NumPy 数组一样被索引。
        """
        if isinstance(item, str):
            # 1. 按键名获取: batch["obs"]
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            # 2. 按键元组获取: batch["obs", "actions"]
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            # 返回一个*新*的、只包含这几个键的 EpisodeBatch
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            # 3. 按索引或切片获取: batch[0:4] (采样 mini-batch) 或 batch[:, 0:10] (裁剪时间)
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            # 对*所有*张量应用这个切片
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size) # 计算新的 batch_size
            ret_max_t = self._get_num_items(item[1], self.max_seq_length) # 计算新的 max_seq_length

            # [关键] 返回一个*新*的、更小的 EpisodeBatch "视图"
            # (注意 `data=new_data`，这会触发 `__init__` 中的 `if data is not None:` 逻辑)
            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        # (辅助函数) 计算切片后的新维度大小
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        # (辅助函数) 创建一个新的空 SN() 对象
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        # (辅助函数) 解析切片索引
        parsed = []
        if (isinstance(items, slice)
            or isinstance(items, int)
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))
            ):
            items = (items, slice(None)) # (如果只给 batch 切片，自动添加完整的时间切片)

        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous") # 时间维度不支持非连续索引

        for item in items:
            if isinstance(item, int):
                parsed.append(slice(item, item+1)) # 将整数索引转为切片
            else:
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        """
        [核心优化] 计算此批次中，所有回合的*最大*实际长度 (max_t)。
        这是通过对 "filled" 掩码求和 (dim=1) 并取最大值 (max(0)) 来实现的。
        
        例如，一个批次:
        [1, 1, 1, 0, 0] -> 长度 3
        [1, 1, 1, 1, 1] -> 长度 5
        [1, 1, 0, 0, 0] -> 长度 2
        
        此函数将返回 5。
        
        这在 `Learner` 中用于将批次裁剪 (truncate) 到 `max_t`，
        以避免 RNN 在无用的空白填充 (0) 上浪费计算资源。
        """
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        # (辅助函数) 打印此批次的摘要
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                   self.max_seq_length,
                                                                                   self.scheme.keys(),
                                                                                   self.groups.keys())

# --- [标准经验回放缓冲区] ---
# 一般的循环队列 (Circular Queue / Ring Buffer)
# 在上面的基础上添加了循环队列机制，可以让旧的经验被替换。还有随机采样机制，可以从中random_sample
class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        # [关键] ReplayBuffer 继承了 EpisodeBatch
        # 它*就是*一个巨大的 EpisodeBatch，其 batch_size = buffer_size
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0           # 指针：下一个要插入数据的位置
        self.episodes_in_buffer = 0     # 计数器：缓冲区中当前有多少*有效*的回合

    def insert_episode_batch(self, ep_batch):
        """
        [核心] 将一个“采集车”(ep_batch) 的数据插入到这个巨大的“仓库”(ReplayBuffer) 中。
        """
        # 检查是否会超出缓冲区末尾
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            # --- [情况A: 空间足够] ---
            # 直接使用父类 (EpisodeBatch) 的 update 方法，
            # 将 `ep_batch` 的数据 "填充" 到 `self.buffer_index` 指针的位置
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size), # batch 切片
                        slice(0, ep_batch.max_seq_length),                                 # time 切片
                        mark_filled=False) # (ep_batch 已经有 "filled" 了，不需要重复标记)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            
            # [关键] 移动指针 (并实现循环)
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            # [关键] 取模 (Modulo) 运算，实现*循环*
            # (如果 buffer_index == buffer_size, 它会变回 0)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            # --- [情况B: 空间不足，需要"绕圈"] ---
            # (例如: buffer_size=100, index=98, ep_batch_size=8)
            buffer_left = self.buffer_size - self.buffer_index # 100 - 98 = 2
            # 1. 先填充剩下的 2 个位置
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            # 2. 递归调用，将剩下的 6 个数据从*开头* (index=0) 插入
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        # [核心] 检查缓冲区中的*有效*回合数是否*足够*一次训练采样
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        """
        [核心] 从缓冲区中*均匀随机*地采样一个批次的数据。
        """
        assert self.can_sample(batch_size)
        
        if self.episodes_in_buffer == batch_size:
            # (如果缓冲区大小 == 批次大小，例如 On-Policy 算法)
            return self[:batch_size]
        else:
            # [关键] 均匀采样 (Uniform Sampling)
            # 从 [0, episodes_in_buffer) 中，随机选择 `batch_size` 个 *不重复* 的索引
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            # [关键] 使用父类 (EpisodeBatch) 强大的切片功能
            # `self[ep_ids]` 会返回一个*新*的 EpisodeBatch，
            # 它只包含 `ep_ids` 索引对应的回合数据。
            return self[ep_ids]

    def uni_sample(self, batch_size):
        # (sample 的别名)
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        # (一个特殊的采样器，总是采样*最新*的数据，不常用)
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            return self.uni_sample(batch_size)
        else:
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        # (打印摘要)
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                       self.buffer_size,
                                                                       self.scheme.keys(),
                                                                       self.groups.keys())


# --- [优先经验回放 (Prioritized Experience Replay, PER)] ---
# (继承自标准 ReplayBuffer)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, beta, t_max, preprocess=None, device="cpu"):
        super(PrioritizedReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length,
                                                      preprocess=preprocess, device="cpu")
        # [PER 参数]
        self.alpha = alpha # (alpha) 控制优先级的程度 (0=均匀, 1=完全优先)
        self.beta_original = beta # (beta) 重要性采样(IS)权重的初始值
        self.beta = beta
        # (beta 会从初始值*退火* (anneal) 到 1.0)
        self.beta_increment = (1.0 - beta) / t_max 
        self.max_priority = 1.0 # 新数据的默认优先级

        # [PER 核心数据结构] - 线段树 (Segment Tree)
        # (线段树是一种高效的数据结构，允许 O(log N) 时间的采样和更新)
        
        # 1. 找到大于 buffer_size 的最小的 2 的幂 (例如 5000 -> 8192)
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        # 2. `SumSegmentTree` 用于存储*优先级* (P^alpha)，并从中进行*采样*
        self._it_sum = SumSegmentTree(it_capacity)
        # 3. `MinSegmentTree` 用于存储*最小*优先级，用于计算*重要性采样权重* (IS Weight)
        self._it_min = MinSegmentTree(it_capacity)

    def insert_episode_batch(self, ep_batch):
        """
        [核心] [重写] 插入新数据，并*更新*线段树。
        """
        pre_idx = self.buffer_index
        # 1. [关键] 首先，调用*父类*的 insert... 方法，将数据存入张量
        super().insert_episode_batch(ep_batch)
        idx = self.buffer_index # (获取更新后的 index)

        # 2. [关键] 为所有*新插入*的数据，在线段树中设置*最大优先级*
        #    (这确保了新数据 (模型还没见过) 会被*优先*采样)
        if idx >= pre_idx:
            # (情况A: 未绕圈)
            for i in range(idx - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
        else:
            # (情况B: 绕圈了)
            # e.g., pre_idx=98, idx=6, buffer_size=100
            # 2a. 更新 [98, 99]
            for i in range(self.buffer_size - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
            # 2b. 更新 [0, 1, 2, 3, 4, 5]
            for i in range(self.buffer_index):
                self._it_sum[i] = self.max_priority ** self.alpha
                self._it_min[i] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        """
        [核心] 按优先级*比例*采样 (Proportional Sampling)。
        """
        res = []
        # 1. 获取所有有效数据的优先级总和 (p_total)
        p_total = self._it_sum.sum(0, self.episodes_in_buffer - 1)
        # 2. 将总和 (p_total) 分成 `batch_size` 个*等长*的区间 (strata)
        every_range_len = p_total / batch_size
        # 3. [分层采样] (Stratified Sampling)
        for i in range(batch_size):
            # 4. 从第 i 个区间 [i*len, (i+1)*len] 中*均匀*采样一个值
            mass = random.random() * every_range_len + i * every_range_len
            # 5. [关键] 使用线段树的 `find_prefixsum_idx`
            #    (在 O(log N) 时间内) 找到这个采样值 `mass` 对应的*数据索引* (idx)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, t):
        """
        [核心] [重写] 执行*优先*采样，并返回*重要性采样权重*。
        """
        assert self.can_sample(batch_size)
        # 1. [退火 Beta] (Anneal Beta)
        #    t 是 t_env (全局环境步数)
        self.beta = self.beta_original + (t * self.beta_increment)

        # 2. [关键] 执行*比例*采样 (而不是均匀采样)
        idxes = self._sample_proportional(batch_size) # (获取采样到的索引)
        
        # --- [第 3 步: 计算重要性采样 (IS) 权重] ---
        # (IS 权重用于在 Learner 中*修正*损失，以抵消优先采样带来的*偏差*)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum() # (最小优先级 P_min)
        # (最大权重，用于归一化)
        max_weight = (p_min * self.episodes_in_buffer) ** (-self.beta)

        for idx in idxes:
            # (P_sample = 该样本被采样的概率)
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            # (IS 权重公式: (P(i) * N)^(-beta))
            weight = (p_sample * self.episodes_in_buffer) ** (-self.beta)
            # (归一化权重，使其 <= 1)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        # 4. [关键] 返回数据批次、它们的索引(idxes)、以及它们的权重(weights)
        return self[idxes], idxes, weights

    def update_priorities(self, idxes, priorities):
        """
        [核心] 在 Learner 训练*之后*，用新的 TD-Error (priorities) 来*更新*线段树。
        
        Parameters
        ----------
        idxes: [int]
            `sample` 函数返回的索引列表
        priorities: [float]
            Learner 计算出的、与 `idxes` 对应的*新* TD-Error (或优先级)
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.episodes_in_buffer
            # [关键] 更新线段树
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            # 更新已知的最大优先级
            self.max_priority = max(self.max_priority, priority)