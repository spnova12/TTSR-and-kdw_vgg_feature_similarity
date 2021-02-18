import torch
import torch.nn as nn

"""
nn.Unfold 를 이해하고자 작성한 script 이다. 
print 된 값들을 하나씩 비교해 보면 unfold 의 방식이 보인다. 
"""

unfold = nn.Unfold(kernel_size=(2, 3))
input = torch.randn(1, 2, 3, 4)
print('-----------------------')
print(input.shape)
print(input)
output = unfold(input)
print('-----------------------')
# each patch contains 30 values (2x3=6 vectors, each of 5 channels)
# 4 blocks (2x3 kernels) in total in the 3x4 input
print(output.shape)
print(output)

B = torch.randn(1, 4, 6)

print('\n\nB.shape :', B.shape)
print('\nB :', B)
print('\n\noutput * B :\n', torch.bmm(output, B))
print('\n\noutput * B.shape :\n', torch.bmm(output, B).shape)




