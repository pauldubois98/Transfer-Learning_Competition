import torch

conv_inp1 = torch.rand(1,1,7,7)
conv_inp2 = torch.rand(1,1,8,8)

conv1 = torch.nn.Conv2d(1, 1, kernel_size = 3, stride = 2)

out1 = conv1(conv_inp1)
out2 = conv1(conv_inp2)
print(out1.shape)         # torch.Size([1, 1, 3, 3])
print(out2.shape)         # torch.Size([1, 1, 3, 3])

conv_t1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)
conv_t2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, output_padding=1)
transposed1 = conv_t1(out1)
transposed2 = conv_t2(out2)

print(transposed1.shape)      # torch.Size([1, 1, 7, 7])
print(transposed2.shape)      # torch.Size([1, 1, 8, 8])