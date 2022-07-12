import torch
import numpy as np
import random
from torch.utils.data import Dataset

class ArithmeticDataset(Dataset):
    def __init__(self, size=10000, num_digit=4, seed=0):
        self.equations = []
        self.digit = num_digit

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        upper_bound = 10**self.digit
        for i in range(size):
            op = random.randint(0, 3)
            a = random.randint(0, upper_bound)
            b = random.randint(0, upper_bound)
            if op == 0: # addition
                self.equations.append(f"{a}+{b}={a+b}")
            elif op == 1: # subtraction
                self.equations.append(f"{a}-{b}={a-b}")
            elif op == 2: # multiplication
                self.equations.append(f"{a}*{b}={a*b}")
            elif op == 3: # division
                while b == 0:
                    b = random.randint(0, upper_bound)
                self.equations.append(f"{a}/{b}={a//b}")

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, idx):
        return self.equations[idx]

    def num_digit(self):
        return self.digit




