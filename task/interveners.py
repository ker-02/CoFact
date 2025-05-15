import torch
import torch.nn as nn


def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)

    return wrapped


class Collector():
    collect_state = True
    collect_action = False
    collect_q = True  # 是否收集Q
    collect_k = True  # 是否收集K

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
        # self.q_values = []  # 用来保存 Q
        # self.k_values = []  # 用来保存 K

    def reset(self):
        self.states = []
        self.actions = []
        # self.q_values = []
        # self.k_values = []

    def __call__(self, b, s):
        if self.head == -1:
            self.states.append(
                b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[
                                   self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b


class Collector2():
    collect_q = True  # 是否收集Q
    collect_k = True  # 是否收集K
    collect_state = False
    collect_action = False

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.q_values = []  # 用来保存 Q
        self.k_values = []  # 用来保存 K

    def reset(self):
        self.states = []
        self.q_values = []
        self.k_values = []

    def __call__(self, b, q, k):
        # 捕获 Q 和 K
        if self.collect_q and q is not None:
            if self.head == -1:
                self.q_values.append(q[0, -1].detach().clone())  # 捕获 Q
            else:
                self.q_values.append(q[0, -1].reshape(32, -1)[self.head].detach().clone())  # 捕获指定 head 的 Q

        if self.collect_k and k is not None:
            if self.head == -1:
                self.k_values.append(k[0, -1].detach().clone())  # 捕获 K
            else:
                self.k_values.append(k[0, -1].reshape(32, -1)[self.head].detach().clone())  # 捕获指定 head 的 K

        # 返回 b，因为你可能在这个方法中修改了其他部分的值
        return b


class AttentionCollector:
    def __init__(self):
        self.q_values = []
        self.k_values = []

    def __call__(self, module, input, output):
        # 打印输出确认其格式
        print(f"Output of Attention Layer: {output}")

        # 假设 output 是一个包含 Q, K 和 V 的元组
        if isinstance(output, tuple) and len(output) == 3:
            q, k, v = output
            self.q_values.append(q.detach().cpu().numpy())  # 保存 Q
            self.k_values.append(k.detach().cpu().numpy())  # 保存 K
        else:
            print("Output is not in expected format (Q, K, V)")

    def reset(self):
        self.q_values = []
        self.k_values = []


class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1

    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b

class Intervener():

    def __init__(self, weight):
        self.weight = weight
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []


    def __call__(self, b, s):
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.weight.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = action * b[0, -1]
        return b