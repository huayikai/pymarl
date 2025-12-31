import numpy as np


class DecayThenFlatSchedule():
    # 减少然后不变
    # liner就是线性下降，exp就是指数下降
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))

class LinearIncreaseSchedule():
    # 线性增加然后不变
    def __init__(self,
                 start,
                 finish,
                 time_length):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start - self.delta * T)
