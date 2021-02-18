from utils import MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torchvision.transforms as transforms

import numpy as np
import cv2
from sklearn.feature_extraction import image


def tensor2cv2(tensor):
    # 3 channel 영상을 RGB 영상으로 만들어준다.
    # 또는 1 channel 영상을 grayscale 영상으로 만들어준다.

    # 0,1 -> 0,255
    npimg = tensor.numpy() * 255

    # 영상 후처리.
    npimg = np.around(npimg)
    npimg = npimg.clip(0, 255)
    npimg = npimg.astype(np.uint8)

    # 3 channel 일 경우
    if len(npimg.shape) == 3:
        # tensor 를 cv2 로 바꾸기 위해 channel 순서를 바꿔줌 (channel, w, h) -> (w, h, channel)
        npimg = np.transpose(npimg, (1, 2, 0))
    # 1 channel 영상일 경우

    return npimg


def tensor_bound_box(t, top_left, bottom_right, bgr):
    # 색 정보를 담고있는 tensor 를 만들어준다.
    bgr_tensor = torch.tensor([[bgr[0]], [bgr[1]], [bgr[2]]])  # [3, 1]
    bgr_tensor = bgr_tensor.expand(3, bottom_right[1]-top_left[1])  # right_bottom[1]-left_top[1] 는 윈도우 사이즈.

    # 박스를 그려줄 새 img 를 준비한다.
    img_tensor_marked = t.clone().detach()

    # 새 img 에 박스를 그려준다.
    img_tensor_marked[..., top_left[0], top_left[1]:bottom_right[1]] = bgr_tensor
    img_tensor_marked[..., bottom_right[0], top_left[1]:bottom_right[1]] = bgr_tensor
    img_tensor_marked[..., top_left[0]:bottom_right[0], top_left[1]] = bgr_tensor
    img_tensor_marked[..., top_left[0]:bottom_right[0], bottom_right[1]] = bgr_tensor

    return img_tensor_marked


def RGB_patches_extractor(img, window_size): # must color image
    [M, N, _] = img.shape
    img_B = np.pad(img[:,:,0], [int(window_size/2), int(window_size/2)], mode='constant', constant_values=0)
    img_G = np.pad(img[:,:,1], [int(window_size/2), int(window_size/2)], mode='constant', constant_values=0)
    img_R = np.pad(img[:,:,2], [int(window_size/2), int(window_size/2)], mode='constant', constant_values=0)
    img_patches_B = image.extract_patches_2d(img_B, (window_size, window_size))
    img_patches_G = image.extract_patches_2d(img_G, (window_size, window_size))
    img_patches_R = image.extract_patches_2d(img_R, (window_size, window_size))
    bbb = img_patches_B.reshape(M*N, window_size,window_size, 1)
    ggg = img_patches_G.reshape(M*N, window_size,window_size, 1)
    rrr = img_patches_R.reshape(M*N, window_size,window_size, 1)

    # 이 코드에서는 cv2 형태로 진행할 것이기 때문에 bgr 로 순서 통일해줌.
    rgb_patches = np.concatenate((bbb, ggg, rrr), axis=3)
    return rgb_patches


############################################################################################################

if __name__ == "__main__":
    # scale factor
    scale_factor = 2

    # 사용할 영상을 읽어준다.
    img = cv2.imread("img_067.png", cv2.IMREAD_COLOR)
    print('입력 영상 사이즈 :', img.shape)

    # scale factor 에 맞게 ref 영상을 만들어준다.
    img_ref = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LANCZOS4)
    print('ref 영상 사이즈 :', img_ref.shape)
    # cv2.imwrite('img_ref.png', img_ref)

    # sliding window size. 홀수여여 됨.
    win_size = 63

    # img_ref 를 win size 로 잘라서 Pool 을 만들어 준다.
    pool_patches = RGB_patches_extractor(img_ref, win_size)
    print('pool_patches.shape :', pool_patches.shape)

    # pool 의 patch 한장을 저장해본다.
    # cv2.imwrite('pool_patches_sample.png', pool_patches[0])

    # pool_patches 를 tensor 로 만들어준다.
    pool_patches = torch.from_numpy(np.array(pool_patches, np.float32, copy=False))
    pool_patches = pool_patches.permute((0, 3, 1, 2)).contiguous()
    pool_patches_tensor = pool_patches.float().div(255)
    print('pool_patches_tensor.shape :', pool_patches_tensor.shape)

    # cv2 image to tensor
    totensor = transforms.ToTensor()  # 이 함수를 통해 영상이 0~1 사이로 normalized 됨.
    img_tensor = totensor(img)
    print('img_tensor.shape :', img_tensor.shape)

    # img_tensor 에서 window 를 하나 뽑아본다.
    h_position = 100
    w_position = 110
    img_tensor_patch = img_tensor[..., h_position:h_position+win_size, w_position:w_position+win_size]
    print('img_tensor_patch.shape :', img_tensor_patch.shape)

    # 원본 영상에 이 위치를 표시해보자.
    img_tensor_marked = tensor_bound_box(img_tensor,
                                         top_left=(h_position, w_position),
                                         bottom_right=(h_position+win_size, w_position+win_size),
                                         bgr=(0, 0, 1)
                                         )
    cv2.imwrite('img_tensor_marked.png', tensor2cv2(img_tensor_marked))

    # img_tensor_patch 한번 저장해본다.
    cv2.imwrite('img_tensor_patch.png', tensor2cv2(img_tensor_patch))

    ############################################################################################################

    # img_tensor_patch 를 길이가 1인 patch 로 만들어준다.
    img_tensor_patch_norm = img_tensor_patch / torch.sqrt(torch.sum(img_tensor_patch**2))

    # pool_patches_tensor 를 길이가 1인 patch 들로 만들어준다.
    _pool_patches_tensor = torch.sqrt(torch.sum(pool_patches_tensor**2, dim=(1, 2, 3)))
    pool_size = pool_patches_tensor.shape[0]
    _pool_patches_tensor = _pool_patches_tensor.view(pool_size, 1, 1, 1)
    pool_patches_tensor_norm = pool_patches_tensor / _pool_patches_tensor.expand(pool_patches_tensor.shape)

    # 잘 노말라이즈 되었나 확인해보자. patch 들 중 아무거나 하나 골라서 길이를 측정해보자. 1 이 나오면 잘 된 것!
    idx_temp = 10  # 측정해볼 patch 의 인덱스.
    print('norm 잘 됐는지 검증 :', torch.sqrt(torch.sum(pool_patches_tensor_norm[idx_temp]**2)))

    ############################################################################################################

    # conv 연산을 통해 유사도를 측정해보자.
    # https://pytorch.org/docs/stable/nn.functional.html#conv2d
    # input -> input tensor of shape : (minibatch, in_channels, iH, iW)
    # weight -> filters of shape : (out_channels, in_channels, kH, kW))
    _input = img_tensor_patch_norm.unsqueeze(0)  # [1, 3, win_size, win_size]
    _weight = pool_patches_tensor_norm  # [pool_size, 3, win_size, win_size]

    gpu_on = False
    # cpu 로 해도 빠른듯.
    if gpu_on:
        _input.cuda()
        _weight.cuda()

    with torch.no_grad():  # grad 는 여기서 필요 없기때문에 빠른 속도를 위해 꺼준다.
        similarity_score = F.conv2d(_input, _weight)

    # 각 pool 의 영상이 img 와 얼마나 비슷한지에 대한 점수 리스트.
    similarity_score = torch.squeeze(similarity_score)

    # 가장 conv 연산 결과가 큰놈 의 index (준상이가 필요한 부분)
    similarity_score_argmax = torch.argmax(similarity_score)  # (1, pool_size, 1, 1)
    print('similarity_score_argmax :', similarity_score_argmax)
    similar_patch = pool_patches_tensor[similarity_score_argmax]
    cv2.imwrite('ref_tensor_patch.png', tensor2cv2(similar_patch))

    # ref 영상에 이 위치를 표시해보자.
    h_position = similarity_score_argmax // img_ref.shape[1]
    w_position = similarity_score_argmax % img_ref.shape[1]
    padding = transforms.Pad(int(win_size / 2), fill=0)
    img_ref_tensor_marked = tensor_bound_box(padding(totensor(img_ref)),
                                             top_left=(h_position, w_position),
                                             bottom_right=(h_position+win_size, w_position+win_size),
                                             bgr=(0, 0, 1)
                                             )
    cv2.imwrite('img_ref_tensor_marked.png', tensor2cv2(img_ref_tensor_marked))





class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        print('================LTE')
        print(x.shape)
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        print(x.shape)
        x = self.slice2(x)
        x_lv2 = x
        print(x.shape)
        x = self.slice3(x)
        x_lv3 = x
        print(x.shape)
        return x_lv1, x_lv2, x_lv3