import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        print('\n-------------')
        print(input.shape)
        print(index.shape)
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1

        index = index.view(views)
        print(index.shape)
        print('expense :',expanse)
        index = index.expand(expanse)
        print(index.shape)

        print(dim)
        print('index', index)
        g = torch.gather(input, dim, index)
        print(g.shape)

        return g

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):

        ### search
        print('lrsr_lv3.shape :', lrsr_lv3.shape)
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)

        print('refsr_lv3.shape :', refsr_lv3.shape)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)


        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        print('refsr_lv3_unfold.shape :', refsr_lv3_unfold.shape)
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
        print('lrsr_lv3_unfold.shape :', lrsr_lv3_unfold.shape)

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) # [N, Hr*Wr, C*k*k] * [N, C*k*k, H*W] = [N, Hr*Wr, H*W]
        print('R_lv3.shape :', R_lv3.shape)
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]
        print('R_lv3_star_arg.shape :', R_lv3_star_arg.shape)

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        print('ref_lv3_unfold.shape :', ref_lv3_unfold.shape)
        print('ref_lv2_unfold.shape :', ref_lv2_unfold.shape)
        print('ref_lv1_unfold.shape :', ref_lv1_unfold.shape)

        print('\n=====================bis')
        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        print('\n\noutput_size=lrsr_lv3.size()[-2:] :', lrsr_lv3.size()[-2:])
        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        print('T_lv3.shape : ', T_lv3.shape)
        print('\n')
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1