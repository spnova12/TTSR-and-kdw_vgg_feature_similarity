import torch
import torch.nn as nn

v = torch.tensor(
    [
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [-1, -2, -3],
        ],
    ]
)
print('\nv')
print(v.shape)
print(v)

index = torch.LongTensor(
    [[[1, 2, 2]]]
)
expanse = [-1, 4, -1]
index = index.expand(expanse)
print('\nindex')
print(index.shape)
print(index)

# index = torch.LongTensor(
#     [
#         [
#             [0, 1, 0],
#             [1, 0, 0],
#             [2, 1, 2]
#         ],
#         [
#             [0, 1, 1],
#             [1, 0, 1],
#             [2, 1, 1]
#         ]
#     ]
# )

v_new = torch.gather(v, 2, index)

print('v_new')
print(v_new.shape)
print(v_new)
