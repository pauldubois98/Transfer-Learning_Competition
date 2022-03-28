import torch

t = torch.empty(3, 4, 5)  # returns a tensor filled with uninitialized (random and not to be used) data
print(t)
print(t.size())
print(t.size(0))
print(t.size(1))
print(t.size(2))
