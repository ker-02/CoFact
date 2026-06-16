import torch
import torch.nn as nn


def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)

    return wrapped


class Collector():
    collect_state = True
    collect_action = False
    collect_q = True 
    collect_k = True  

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []



    def reset(self):
        self.states = []
        self.actions = []



    def __call__(self, b, s):
        if self.head == -1:
            self.states.append(
                b[0, -1].detach().clone())  
        else:
            self.states.append(b[0, -1].reshape(32, -1)[
                                   self.head].detach().clone())  
        return b


class Collector2():
    collect_q = True  
    collect_k = True 
    collect_state = False
    collect_action = False

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.q_values = [] 
        self.k_values = [] 

    def reset(self):
        self.states = []
        self.q_values = []
        self.k_values = []

    def __call__(self, b, q, k):

        if self.collect_q and q is not None:
            if self.head == -1:
                self.q_values.append(q[0, -1].detach().clone())
            else:
                self.q_values.append(q[0, -1].reshape(32, -1)[self.head].detach().clone()) 

        if self.collect_k and k is not None:
            if self.head == -1:
                self.k_values.append(k[0, -1].detach().clone()) 
            else:
                self.k_values.append(k[0, -1].reshape(32, -1)[self.head].detach().clone())  


        return b


class AttentionCollector:
    def __init__(self):
        self.q_values = []
        self.k_values = []

    def __call__(self, module, input, output):
    
        print(f"Output of Attention Layer: {output}")

     
        if isinstance(output, tuple) and len(output) == 3:
            q, k, v = output
            self.q_values.append(q.detach().cpu().numpy()) 
            self.k_values.append(k.detach().cpu().numpy())  
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
        self.states.append(b[0, -1].detach().clone()) 
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
        self.states.append(b[0, -1].detach().clone())
        action = self.weight.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = action * b[0, -1]
        return b
