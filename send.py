import os
import time
import requests
import configparser
from utils.utils import load_image
import numpy as np


config = configparser.ConfigParser()
config.read('config.ini')


# 1. 设定文本提示词（整个推理过程仅需输入一次，之后也可以修改）
# 获取检测类别（文本提示词）
caption = config.get('Paths', 'caption')
# 将文本提示词送入GPU内存（整个推理过程中仅需送入一次）
response = requests.post("http://127.0.0.1:8011/upload_caption/", json={'message': caption})
print(f"Load caption {caption} successfully\n")


# 2. 基于先前的文本提示词检测图片
# 指定图片目录
image_folder = config.get('Paths', 'input_path')

print(f"Inferencing data in {image_folder}")
# 收集所有图片文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅限图片格式
        file_path = os.path.join(image_folder, filename)
        print(f"Processing {file_path}")
        t0 = time.time()
        image, image_transformed = load_image(file_path)
        input_img = np.expand_dims(np.asarray(image_transformed), axis=0)
        print(f"Preprocessing time: {(time.time() - t0):.3f}s")

        # 发送请求
        response = requests.post("http://127.0.0.1:8011/upload_image/", json={'img_name': filename, 'preprocessed_img':input_img.tolist(), 'ori_img':image.tolist()})

        # 打印服务器返回的宽高信息
        print(response.json())
