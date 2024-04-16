import os
import numpy as np
# import cv2
from PIL import Image
import math
def load_file_list(directory):
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        list.append(os.path.join(directory,filename))
        #list.sort()
    return list

def iter_files(rootDir):
    #遍历根目录
    list=[]
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            list.append(file_name)
    return list

def sort_list(l):
    s_list=[]
    l = list(map(lambda x:x.split("\\"),l))
    l.sort(key=lambda ele:(ele[-1],ele[-2]))
    #print(l)

    for i in range(len(l)):
        head = ""
        #print(l[i])
        for j in range(len(l[i])):
            head =os.path.join(head,l[i][j])
            #print(head)
        s_list.append(head.replace(":",":\\"))
    return s_list



def isNum(value):
    try:
        x = float(value)
    except TypeError:
        return False
    except ValueError:
        return False
    except Exception:
        return False
    else:
        return True



def getYdata(path, size):
    w= size[0]
    h=size[1]
    #print(w,h)
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)

        Yt = np.asarray(tem, dtype='float32')

        # for n in range(h):
        #     for m in range(w):
        #         Yt[n, m] = ord(fp.read(1))

    return Yt

def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if((type(hr_image)==type(np.array([]))) or (type(hr_image)==type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff*diff)
        mse = np.maximum(eps, mse)
        return float(10*math.log10(max_value*max_value/mse))
    else:
        assert len(hr_image.shape)==4 and len(sr_image.shape)==4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10*tf.log(max_value*max_value/mse)/math.log(10)

def read_yuv420_file(r_file):
    w = int(r_file.split(".")[0].split("_")[-1].split("x")[0])
    h = int(r_file.split(".")[0].split("_")[-1].split("x")[1])
    my_file = open(r_file, 'rb')
    yuv = np.zeros((h, w, 3), np.uint8)
    rgb = np.zeros((h, w, 3), np.uint8)
    for num in range(0, 1):
        for i in range(0, h):
            for j in range(0, w):
                data = my_file.read(1)
                data = ord(data)
                yuv[i, j, 0] = data
        for y in range(0, int(h / 2)):
            for x in range(0, int(w / 2)):
                data = my_file.read(1)
                data = ord(data)
                yuv[2 * y, 2 * x, 1] = data
                yuv[2 * y + 1, 2 * x, 1] = data
                yuv[2 * y, 2 * x + 1, 1] = data
                yuv[2 * y + 1, 2 * x + 1, 1] = data
        for y in range(0, int(h / 2)):
            for x in range(0, int(w / 2)):
                data = my_file.read(1)
                data = ord(data)
                yuv[2 * y, 2 * x, 2] = data
                yuv[2 * y + 1, 2 * x, 2] = data
                yuv[2 * y, 2 * x + 1, 2] = data
                yuv[2 * y + 1, 2 * x + 1, 2] = data

    # rgb[:,:,0]=yuv[:,:,0]
    # rgb[:, :, 1] = yuv[:, :, 2]
    # rgb[:, :, 2] = yuv[:, :, 1]
    # cv2.imshow("yuv",yuv[:,:,0])
    # cv2.waitKey()
    # rgb=cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    # yuv2=cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    # yuv2=0.299*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
    # cv2.imshow("yuv2",yuv2[:,:,0])
    # # cv2.waitKey()
    # org_y=getYdata("BasketballDrill_832x480.yuv",(w,h))
    # print(psnr(org_y[:,:],yuv[:,:,0]))
    # print(psnr(org_y[:, :], yuv2[:, :, 0]))
    # print(psnr(yuv2[:, :, 0],yuv[:,:,0]))


    return yuv

def write_RGB2Y_file(r_file,mat):
    yuv=cv2.cvtColor(mat,cv2.COLOR_RGB2YCrCb)
    y = np.asarray(yuv, "int8")
    np.save(r"./tmp", yuv[:,:,0])
    src = open(r"./tmp.npy", "rb")
    src.seek(128, 1)
    data = src.read()
    dst = open(r_file, "wb")
    dst.write(data)
    #os.remove("./tmp.npy")