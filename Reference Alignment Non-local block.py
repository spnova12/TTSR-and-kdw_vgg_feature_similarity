import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from PIL import ImageFilter
import torchvision.transforms as transforms

"""
Non local 방식에 대한 실험.
"""


def min_max_norm(tensor):
    # tensor 를 min max 로 normalize 해준다.
    normalized = torch.clone(tensor)
    for c in range(3):
        min_v = torch.min(tensor[c])
        range_v = torch.max(tensor[c]) - min_v
        normalized[c] = (tensor[c] - min_v) / range_v
    return normalized


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


def norm_to_one(t1):
    # softmax 대신에 사용해보려고 만듬. (일반적인 평균 norm)
    t1_s = torch.sum(t1, dim=1)
    t1_s = t1_s.unsqueeze(1).repeat(1, t1.shape[0])
    t1_norm = t1 / t1_s
    return t1_norm


def max_one_hot(t1):
    # 가장 큰 값만 골라라.
    t1_max = (t1 == torch.max(t1, 1, keepdim=True)[0]).float()
    t1_max = norm_to_one(t1_max)
    return t1_max




def get_similarity_map(t1, t2, mode=0):
    similarity_map = None

    if mode == 0:
        similarity_map = torch.matmul(t1, t2)

    # cosine similarity.
    elif mode == 1:
        t1_s = torch.sqrt(torch.sum(t1*t1, dim=1))
        t1_s = t1_s.unsqueeze(1).repeat(1, 3)
        t1_norm = t1 / t1_s
        # print('t1norm :', torch.sum(t1_norm * t1_norm, dim=1))

        t2_s = torch.sqrt(torch.sum(t2*t2, dim=0))
        t2_s = t2_s.repeat(3, 1)
        t2_norm = t2 / t2_s
        # print('t2norm :', torch.sum(t2_norm * t2_norm, dim=0))

        similarity_map = torch.matmul(t1_norm, t2_norm)

    return similarity_map


"""
dim 에 대한 연습 예제.
"""
# test = torch.tensor(np.array([
#     [1., 2., 3.],
#     [1., 5., 6.],
#     [1., 5., 6.],
#     [1., 5., 6.]
# ]))
# print(test.shape)
# print(test)
# test_norm = F.softmax(test, dim=-1)
# print(test_norm)


def reference_alignment_nl(img_LRup, img_ref):
    # pil to tensor
    totensor = transforms.ToTensor()
    LRup = totensor(img_LRup)
    ref = totensor(img_ref)

    # r, g, b 별로 시각화해본다.
    w = LRup.shape[-1]
    LRup_gray = torch.zeros(w, 3 * w)
    ref_gray = torch.zeros(w, 3 * w)
    for i in range(3):
        LRup_gray[:, i * w:(i + 1) * w] = LRup[i]
        ref_gray[:, i * w:(i + 1) * w] = ref[i]
    tensor2pil(LRup_gray).save(f'{out_folder}/0_LRup_.png')
    tensor2pil(ref_gray).save(f'{out_folder}/1_ref_.png')

    # theta 를 만들어준다.
    theta = LRup.view(3, -1).permute(1, 0)
    print(f'theta : {LRup.shape} -> {theta.shape}')
    tensor2pil(theta).save(f'{out_folder}/2_theta.png')

    # pi 를 만들어준다.
    pi = ref.view(3, -1)
    print(f'pi : {ref.shape} -> {pi.shape}')
    tensor2pil(pi).save(f'{out_folder}/3_pi.png')

    # g 를 만들어준다.
    g = ref.view(3, -1).permute(1, 0)
    print(f'g : {ref.shape} -> {g.shape}')
    tensor2pil(g).save(f'{out_folder}/4_g.png')
    tensor2pil(g.permute(1, 0).reshape(3, w, w)).save(f'{out_folder}/4_g_back.png')

    # similarity map 추출.
    similarity_map = get_similarity_map(theta, pi, mode=1)
    print(f'similarity_map : {similarity_map.shape}')
    tensor2pil(similarity_map).save(f'{out_folder}/5_similarity_map.png')

    # normalize 해주기.
    # similarity_map_norm = F.softmax(similarity_map, dim=-1)
    # similarity_map_norm = norm_to_one(similarity_map)
    similarity_map_norm = max_one_hot(similarity_map)
    tensor2pil(similarity_map_norm).save(f'{out_folder}/6_similarity_map_softmax.png')

    aligned = torch.matmul(similarity_map_norm, g)
    tensor2pil(aligned).save(f'{out_folder}/7_aligned.png')
    print('aligned : ', aligned.shape)

    aligned_reshaped = aligned.permute(1, 0).reshape(3, w, w)
    # k_reshaped_minmaxnorm = min_max_norm(aligned_reshaped)
    tensor2pil(aligned_reshaped).save(f'{out_folder}/8_aligned_reshaped.png')

    aligned_reshaped_gray = torch.zeros(w, 3 * w)
    for i in range(3):
        aligned_reshaped_gray[:, i * w:(i + 1) * w] = aligned_reshaped[i]
    tensor2pil(aligned_reshaped_gray).save(f'{out_folder}/9_aligned_reshaped_gray_.png')

    # img_kre_minmaxnorm = tensor2pil(kre_minmaxnorm)

    # img_kre_minmaxnorm.save('aligned_minmaxnorm.png')

    #
    #
    # k = torch.matmul(f_nor, g)
    #
    # kre = k.reshape(128,128,3)
    #
    # z = kre + torch_img
    #


"""
input(LRup) 이미지와 reference 이미지 읽어오기
"""
# 사용할 영상을 읽어준다.
img = Image.open("ori_imgs/LR_up.png")
print("pil image shape :", np.array(img).shape)

out_folder = "new_outs_verynext"
#
# img = img.resize((100, 100), Image.LANCZOS)
# # LR 로 사용할 영상 만들기.
# left = 20
# top = 20
# size = 35
# img_LRup = img.crop((left, top, left + size, top + size))
#
# gaussianBlur = ImageFilter.GaussianBlur(1.5)
# img_LRup = img_LRup.filter(gaussianBlur)
#
#
# # reference 로 사용할 영상 만들기.
# left = 40
# top = 40
# img_ref = img.crop((left, top, left + size, top + size))
#
#
# img_LRup.save('img_LR.png')
# img_ref.save('img_ref.png')

img_LRup = Image.open("ori_imgs/LR_up.png").convert("RGB")
img_ref = Image.open("ori_imgs/REF_VERY_NEXT.png").convert("RGB")

# non-local 연산 및 attention 가중치 연산.
reference_alignment_nl(img_LRup, img_ref)
"""

x = torch.tensor([4])
k = x.unsqueeze(1)
print(k)
r = k.repeat(1,4)
print(r)
"""




