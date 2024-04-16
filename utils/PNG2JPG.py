import os
import cv2
from PIL import Image

def load_file_list(directory):
    list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.split(".")[-1] == "png":
                list.append(file_name)
    return sorted(list)

def getName(yuvfileName):  # Test
    name = os.path.splitext(os.path.basename(yuvfileName))[0]
    return name




if __name__ == '__main__':
    file_path = r''
    save_path = r''
    file_name = load_file_list(file_path)
    for i in file_name:
        img = Image.open(i).convert("RGB")
        print(i)
        # print()
        W,H = img.size
        # if W>=2000 and H>=1500:
        #     img = img.resize((W//2, H//2), Image.ANTIALIAS)
        img.save(save_path+'/'+getName(i)+".jpg", quality=100)
