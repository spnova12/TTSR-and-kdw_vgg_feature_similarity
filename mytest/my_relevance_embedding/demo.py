import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torchvision.transforms as transforms

import numpy as np
import cv2
from sklearn.feature_extraction import image


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


# pretrained vgg19 사용하는 방식은 https://github.com/researchmm/TTSR 에서 참고함.
class Vgg19FeatureExtractor(torch.nn.Module):
    def __init__(self, vgg_range=12, rgb_range=1):
        super(Vgg19FeatureExtractor, self).__init__()

        # use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()

        for x in range(vgg_range):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # 학습을 시켜주지 않기때문에 grad 를 모두 없앤다.
        for param in self.slice1.parameters():
            param.requires_grad = False

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        return x_lv1


def tensor2cv2(tensor):
    # 3 channel 영상을 RGB 영상으로 만들어준다.
    # 또는 1 channel 영상을 grayscale 영상으로 만들어준다.

    # 0,1 -> 0,255
    npimg = tensor.cpu().numpy() * 255

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


def img_tensor_bound_box(t, top_left, bottom_right, bgr):
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


def RGB_patch_tensors_extractor(img_tensor, window_size): # must color image
    print("RGB_patch_tensors_extractor===============================")
    print(img_tensor.shape)
    channel = img_tensor.shape[1]
    img_tensor_unfold = F.unfold(img_tensor, kernel_size=(window_size, window_size), padding=window_size//2).squeeze()
    print(img_tensor_unfold.shape)
    img_tensor_unfold = img_tensor_unfold.permute((1, 0))
    print(img_tensor_unfold.shape)
    patches = img_tensor_unfold.view(img_tensor_unfold.shape[0],
                                     channel,
                                     window_size,
                                     window_size)
    print(patches.shape)
    print("===========================================================")
    return patches


if __name__ == "__main__":
    ############################################################################################################
    # scale factor
    scale_factor = 2

    # 사용할 영상을 읽어준다.
    img = cv2.imread("img_098.png", cv2.IMREAD_COLOR)
    print('입력 영상 사이즈 :', img.shape)

    # scale factor 에 맞게 ref 영상을 만들어준다.
    img_ref = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LANCZOS4)
    print('ref 영상 사이즈 :', img_ref.shape)
    cv2.imwrite('img_ref.png', img_ref)

    # sliding window size. 홀수 사용.
    win_size = 63

    totensor = transforms.ToTensor()  # 이 함수를 통해 영상이 0~1 사이로 normalized 됨.

    img_tensor = totensor(img)
    print('img_tensor.shape :', img_tensor.shape)

    img_ref_tensor = totensor(img_ref)
    print('img_ref_tensor.shape :', img_ref_tensor.shape)

    ############################################################################################################

    # vgg19 feature extractor

    # vgg_range : 0 ~ 37 까지 설정 가능. TTSR 에서는 12 를 기본으로 사용, 0 으로 하면 영상이 모델을 거치지 않고 그대로 나온다.
    vgg_range = 12

    if vgg_range > 0:  # -> vgg feature 를 뽑는다.
        # Vgg19FeatureExtractor 를 선언하는 것 만으로도 gpu 용량을 차지하는 듯 하다?
        Vgg19FeatureExtractor = Vgg19FeatureExtractor(vgg_range=vgg_range, rgb_range=1).cuda()

        # vgg feature 에서 win size 를 즉정한다. 측정을 위해 win_size 크기의 임의의 패치를 모델에 통과시켜서 output size 를 관찰한다.
        temp_patch = torch.zeros(3, win_size, win_size)  # vgg19 의 입력은 3 channel 이다.
        print('temp_patch.shape :', temp_patch.shape)
        vgg_win_size = Vgg19FeatureExtractor(temp_patch.unsqueeze(0).cuda()).shape[-1]
        if vgg_win_size % 2 == 0:  # win size 가 홀수인게 편한 듯 하다.-> 짝수면 홀수로 바꿔줌.
            vgg_win_size += 1
        print('vgg_win_size :', vgg_win_size)

        # img_tensor 의 feature 추출. conv 연산을 하려면 [batch, channel, h, w] 형태여야 하기때문에 unsqueeze 사용.
        img_tensor_feature = Vgg19FeatureExtractor(img_tensor.unsqueeze(0).cuda()).detach()
        print('img_tensor_feature.shape :', img_tensor_feature.shape)

        # img_ref_tensor 의 feature 추출.
        img_ref_tensor_feature = Vgg19FeatureExtractor(img_ref_tensor.unsqueeze(0).cuda()).detach()
        print('img_ref_tensor_feature.shape :', img_ref_tensor_feature.shape)
    else:  # -> 이미지 그대로 사용한다.
        vgg_win_size = win_size
        img_tensor_feature = img_tensor.unsqueeze(0)
        img_ref_tensor_feature = img_ref_tensor.unsqueeze(0)

    ############################################################################################################

    # img_ref_tensor_feature 를 vgg_win_size 로 잘라서 Pool 을 만들어 준다.
    pool_patches_tensor = RGB_patch_tensors_extractor(img_ref_tensor_feature, vgg_win_size)
    print('pool_patches_tensor.shape :', pool_patches_tensor.shape)

    # pool 의 patch 한장을 저장해본다. pool 이 생각대로 잘 만들어지고 있음을 확인할 수 있음.
    if vgg_range == 0:
        cv2.imwrite('pool_patches_sample.png', tensor2cv2(pool_patches_tensor[10000]))

    ############################################################################################################

    # img_tensor 에서 window 를 하나 뽑아본다.
    h_position = 500
    w_position = 800
    img_tensor_patch = img_tensor[..., h_position:h_position+win_size, w_position:w_position+win_size]
    print('img_tensor_patch.shape :', img_tensor_patch.shape)

    # 원본 영상에 이 위치를 표시해보자.
    img_tensor_marked = img_tensor_bound_box(img_tensor,
                                             top_left=(h_position, w_position),
                                             bottom_right=(h_position+win_size, w_position+win_size),
                                             bgr=(0, 0, 1)
                                             )
    cv2.imwrite('img_tensor_marked.png', tensor2cv2(img_tensor_marked))

    # img_tensor_patch 한번 저장해본다.
    cv2.imwrite('img_tensor_patch.png', tensor2cv2(img_tensor_patch))

    # 실제 feature 단에서 crop 해야되는 위치를 계산하고, feature 단에서 patch 를 만들어준다.
    h_feature_position = int(h_position * (vgg_win_size/win_size))
    w_feature_position = int(w_position * (vgg_win_size/win_size))
    img_tensor_feature_patch = img_tensor_feature[
                               ...,
                               h_feature_position:h_feature_position+vgg_win_size,
                               w_feature_position:w_feature_position+vgg_win_size]
    print('img_tensor_feature_patch.shape :', img_tensor_feature_patch.shape)

    ############################################################################################################

    # img_tensor_patch 를 길이가 1인 patch 로 만들어준다.
    img_tensor_feature_patch_norm = img_tensor_feature_patch / torch.sqrt(torch.sum(img_tensor_feature_patch**2))

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
    _input = img_tensor_feature_patch_norm  # [1, C, win_size, win_size]
    _weight = pool_patches_tensor_norm  # [pool_size, C, win_size, win_size]

    with torch.no_grad():  # grad 는 여기서 필요 없기때문에 빠른 속도를 위해 꺼준다.
        similarity_score = F.conv2d(_input.cuda(), _weight.cuda())

    # 각 pool 의 영상이 img 와 얼마나 비슷한지에 대한 점수 리스트.
    similarity_score = torch.squeeze(similarity_score)

    # 가장 conv 연산 결과가 큰놈 의 index (준상이가 필요한 부분)
    similarity_score_argmax = torch.argmax(similarity_score)  # (1, pool_size, 1, 1)
    print('similarity_score_argmax :', similarity_score_argmax)

    ############################################################################################################

    # ref 영상에 이 위치를 표시해보자.

    # img_ref_tensor_feature.shape -> [1, C, h, w]
    h_feature_position = similarity_score_argmax // img_ref_tensor_feature.shape[-1]
    w_feature_position = similarity_score_argmax % img_ref_tensor_feature.shape[-1]

    h_ref_position = int(h_feature_position * (win_size/vgg_win_size))
    w_ref_position = int(w_feature_position * (win_size/vgg_win_size))

    # unfold 하는 과정에서 padding 이 되었기 때문에 여기서도 padding 해줘야 된다.
    padding = transforms.Pad(int(win_size / 2), fill=0)
    img_ref_tensor_padded = padding(img_ref_tensor)

    # 가장 비슷한 ref 영역 잘라주기.
    ref_tensor_similar_patch = img_ref_tensor_padded[
                           ...,
                           h_ref_position:h_ref_position + win_size,
                           w_ref_position:w_ref_position + win_size]

    # 잘라준 patch 만 저장해보자.
    cv2.imwrite('ref_tensor_patch.png', tensor2cv2(ref_tensor_similar_patch))

    # 잘라준 patch 가 ref 영상 어디에 있는지 표시해보기.
    img_ref_tensor_marked = img_tensor_bound_box(img_ref_tensor_padded,
                                                 top_left=(h_ref_position, w_ref_position),
                                                 bottom_right=(h_ref_position+win_size, w_ref_position+win_size),
                                                 bgr=(0, 0, 1)
                                                 )
    cv2.imwrite('img_ref_tensor_marked.png', tensor2cv2(img_ref_tensor_marked))
