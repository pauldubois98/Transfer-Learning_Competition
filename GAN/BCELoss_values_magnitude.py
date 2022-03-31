import torch
import torch.nn as nn

# La moitié est fail
loss = nn.BCELoss()
input_bce_loss = torch.tensor([0.01, 0.05, 0.5, 0.2])
print(input_bce_loss)
target = torch.tensor([0., 0., 1., 1.])
print(target)
output = loss(input_bce_loss, target)
print(output)

# Tout est globalement bon
input_bce_loss = torch.tensor([0.01, 0.05, 0.9, 0.9])
print(input_bce_loss)
target = torch.tensor([0., 0., 1., 1.])
print(target)
output = loss(input_bce_loss, target)
print(output)

