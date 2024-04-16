import numpy as np
from skimage.io import imread,imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

## 读取一张图像并可视化
im = imread("kodim23.png") / 255.0
# im = rgb2gray(im)
print(im.shape)


def image2cols(image, patch_size, stride):
    """
    image:需要切分为图像块的图像
    patch_size:图像块的尺寸，如:(10,10)
    stride:切分图像块时移动过得步长，如:5
    """
    import numpy as np
    if len(image.shape) == 2:
        # 灰度图像
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:
        # RGB图像
        imhigh, imwidth, imch = image.shape
    ## 构建图像块的索引
    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    sz = len(range_y) * len(range_x)  ## 图像块的数量
    if len(image.shape) == 2:
        ## 初始化灰度图像
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:
        ## 初始化RGB图像
        res = np.zeros((sz, patch_size[0], patch_size[1], imch))
    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            res[index] = patch
            index = index + 1

    return res

im2col = image2cols(image=im,patch_size=(128,128),stride=128)
print(im2col.shape)
plt.figure(figsize=(12,12))
for ii in np.arange(96):
    imsave(str(ii)+'.png',im2col[ii])

    # plt.subplot(10,10,ii+1)
    # plt.imshow(im2col[ii])
    # plt.axis("off")
# plt.subplots_adjust(wspace = 0.05,hspace = 0.05)
# plt.show()
