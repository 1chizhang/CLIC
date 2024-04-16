import cv2
import math
import numpy as np
import os
import natsort as natsort
from pytorch_msssim import ms_ssim
from torchvision import transforms

from scipy.ndimage import gaussian_filter
def psnr(img1, img2):
    mse = np.mean((img1 / 255 - img2 / 255) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def mpsnr(x_true, x_pred):
     n_bands = x_true.shape[2]
     p = [psnr(x_true[:, :, k], x_pred[:, :, k]) for k in range(n_bands)]
     return np.mean(p)
def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim==2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)
def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D
    mssim = SSIM.mean()

    return mssim

## 计算单张图像
# img1 = cv2.imread('15_CLAHE.png')
# img2 = cv2.imread('15.png')
# e2 = (mpsnr(img1,img2))
# print(e2)

### 批量计算文件夹中的图像
phase = "msssim"
# GEN_PATH
path1 = r""
files1 = os.listdir(path1)
files1 = natsort.natsorted(files1)
print(files1)
print(len(files1))
# GT_PATH
from PIL import Image
path = r""
files = os.listdir(path)
files = natsort.natsorted(files)
print(files)
print(len(files))
# scores = []
sum = 0
for i in range(len(files)):

    file = files[i]  # GT
    file1 = files1[i]  # gen

    filepath = path + "/" + file
    if os.path.isfile(filepath):

        image1 = Image.open(path1 + '/' + file1).convert("RGB")
        # print(image1.shape)
        image = Image.open(path + '/' + file).convert("RGB")

        if phase == "psnr":
            score = psnr(image1, image)
            print(score)
        elif phase == "mpsnr":
            score = (mpsnr(image1, image))
            print('mpsnr:', (score))
        elif phase == "ssim":
            score = getSSIM(image1,image)
            print('ssim:', (score))
        elif phase=='msssim':
            image1  = transforms.ToTensor()(image1).unsqueeze(0)
            image  = transforms.ToTensor()(image).unsqueeze(0)
            ssim = ms_ssim(image1, image, data_range=1.0).item()
            ssim = -10 * math.log10(1 - ssim)
            score = ssim
            print(score)


        sum += score
avg = sum / len(files)
print(avg)

        # scores.append(score)
# sum = 0
# for j in range(len(scores)):
#     if phase == "psnr":
#         print('psnr:', (scores[j]))
#     elif phase == "mpsnr":
#         print('mpsnr:',(scores[j]))
#     elif phase == "ssim":
#         print('ssim:', (scores[j]))
#     sum += scores[j]
# avg = sum / len(scores)


# for j in range(len(scores)):
#     print(scores[j])
