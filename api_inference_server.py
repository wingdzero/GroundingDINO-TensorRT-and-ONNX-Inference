import os
import time
import configparser

from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from engine_inference import TensorRTInfer
from utils.utils import *

app = FastAPI()

config = configparser.ConfigParser()

config.read('config.ini')

output_path = config.get('Paths', 'output_path')

# 全局初始化TensorRT引擎，使其只加载一次
trt_infer = None

@app.on_event("startup")
async def startup_event():
    global trt_infer
    engine_path = config.get('Paths', 'model_path')
    # engine_path = './yolov8n.engine'
    trt_infer = TensorRTInfer(engine_path)
    print("\nTensorRT 引擎已加载！\n")

# 定义接收请求的模型
class Message(BaseModel):
    message: str

# 定义 Pydantic 模型，用于接收请求数据
class DataModel(BaseModel):
    img_name: str  # 字符串字段
    preprocessed_img: List  # 第一个 NumPy 数组
    ori_img: List  # 第二个 NumPy 数组

@app.post("/upload_caption/")
async def upload_caption(message: Message):
    caption = message.message
    print('\n------------------------------------------Start Tokenize captions-----------------------------------------\n')
    t0 = time.time()
    trt_infer.caption_tokenize(caption)
    print(f'Finish tokenize, cost: {time.time() - t0}s')

@app.post("/upload_image/")
async def run_inference(data: DataModel):

    filename = data.img_name
    input_img_data = np.array(data.preprocessed_img, np.float32)  # 需要float32图像数据
    ori_img_data = np.array(data.ori_img, np.uint8)

    print('\n-----------------------------------------Start Inference-----------------------------------------\n')

    # 执行推理
    outputs = trt_infer.infer(input_img_data)

    # 输出结果处理
    boxes, confs, phrases = trt_infer.post_process(outputs, float(config.get('Settings', 'box_threshold')), float(config.get('Settings', 'text_threshold')))
    print(f"{filename} Detect {len(boxes)} target")

    # 画图
    file_name, file_extension = os.path.splitext(filename)
    save_file_name = f"{file_name}_result{file_extension}"
    draw_results(cv2.cvtColor(ori_img_data, cv2.COLOR_RGB2BGR), boxes, confs, phrases, img_save_path=os.path.join(config.get('Paths', 'output_path'), save_file_name))
    print(f"Saved {filename} results in {os.path.join(config.get('Paths', 'output_path'), save_file_name)}")

    print('\n-----------------------------------------Done-----------------------------------------\n')
    # 后处理，返回JSON格式的结果
    return {"box": boxes.tolist(), "confs":confs.tolist(), "cls":phrases}


@app.on_event("shutdown")
async def shutdown_event():
    trt_infer.cleanup()
    print("\nTensorRT 引擎清理完成！")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8011)