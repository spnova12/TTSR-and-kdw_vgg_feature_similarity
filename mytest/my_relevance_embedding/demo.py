from utils import MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torchvision.transforms as transforms

import numpy as np
import cv2
# from PIL import Image
import matplotlib.pyplot as plt


def tensor2pil(tensor):
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
        # tensor 를 pil 로 바꾸기 위해 channel 순서를 바꿔줌 (channel, w, h) -> (w, h, channel)
        npimg = np.transpose(npimg, (1, 2, 0))
        # numpy to pil
        img_recon = Image.fromarray(np.uint8(npimg))

    # 1 channel 영상일 경우
    else:
        # numpy to pil
        img_recon = Image.fromarray(np.uint8(npimg), 'L')

    return img_recon


from sklearn.feature_extraction import image


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
    rgb_patches = np.concatenate((rrr,ggg,bbb),axis=3)
    return rgb_patches


###################################
if __name__ == "__main__":
    # scale factor
    scale_factor = 2

    # 사용할 영상을 읽어준다.
    img = cv2.imread("img_002.png", cv2.IMREAD_COLOR)
    print('입력 영상 사이즈 :', img.shape)

    # scale factor 에 맞게 ref 영상을 만들어준다.
    img_ref = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LANCZOS4)
    print('ref 영상 사이즈 :', img_ref.shape)

    # sliding window size.
    win_size = 63

    # img_ref 를 win size 로 잘라서 Pool 을 만들어 준다.
    pool_patches = RGB_patches_extractor(img_ref, win_size)
    print('pool_patches.shape :', pool_patches.shape)

    # pool 의 patch 한장을 저장해본다.
    cv2.imwrite('patch_sample.png', pool_patches[0])

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
    x_position = 0
    y_position = 0
    img_tensor_patch = img_tensor[..., x_position:x_position+win_size, 0:63]
    print('img_tensor_patch.shape :', img_tensor_patch.shape)



###
    class inner_product_by_conv(nn.Module):
        def __init__(self, win_size, img_tensor_patch):
            super(inner_product_by_conv, self).__init__()
            self.conv1 = nn.Conv2d(3, win_size, win_size)

            K = img_tensor_patch
            print(K.shape)
            K = torch.unsqueeze(torch.unsqueeze(K, 0), 0)
            print(k.shape)
            self.conv1.weight.data = K

        def forward(self, x):
            x = self.conv1(x)
            return x


    net = inner_product_by_conv(win_size=win_size, img_tensor_patch=img_tensor_patch)
    with torch.no_grad():
        corespondence_tensor = net(pool_patches_tensor)

    print(corespondence_tensor.shape)



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