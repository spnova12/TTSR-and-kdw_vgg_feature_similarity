"""
dim 햇갈리는거 한번 정리.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.tensor(
    [
        [
            [
                [1, 2],
                [1, 2],
                [4, 5]
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1]
            ],
        ],
        [
            [
                [1, 2],
                [1, 2],
                [4, 5]
            ],
            [
                [1, 1],
                [1, 1],
                [1, 2]
            ],
        ],
        [
            [
                [1, 2],
                [1, 2],
                [4, 5]
            ],
            [
                [1, 1],
                [1, 1],
                [1, 3]
            ],
        ]
    ]

)
print(a)
print(a.shape)
b = torch.sum(a, dim=(1, 2, 3))
print(b.shape)
print(b)

b = b.view(3, 1, 1, 1)
print('1. ', b.shape)
b = b.expand(3, 2, 3, 2)
print('2.', b.shape)
print(b)





