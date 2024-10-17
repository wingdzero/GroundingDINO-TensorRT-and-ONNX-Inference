import cv2
import torch
import numpy as np
from PIL import Image

from typing import Dict
from cuda import cuda, cudart
from transformers import AutoTokenizer

import groundingdino.datasets.transforms as T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# caption前处理
def preprocess_caption(caption: str):
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

# 加载模型并返回前处理后的图像
def load_image(image_path: str):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

# 根据匹配的概率图从文本中提取短语
def get_phrases_from_posmap(
    posmap: np.ndarray, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
):
    assert isinstance(posmap, np.ndarray), "posmap must be np.ndarray"
    if posmap.ndim == 1:
        # 将指定范围内的元素设为 False
        posmap[:left_idx + 1] = False
        posmap[right_idx:] = False

        # 获取非零元素的索引
        non_zero_idx = np.nonzero(posmap)[0]
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def check_cuda_err(err):
    # 检查是否为 CUDA Driver API 的错误类型 CUresult
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    # 检查是否为 CUDA Runtime API 的错误类型 cudaError_t
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    # 未知错误
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    # 调用 CUDA 函数，并检查其返回的错误码，确保调用成功
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

# 将数据从 CPU 内存复制到 GPU 内存
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# 将数据从 GPU 内存复制到 CPU 内存
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

# 递归求list的维度
def get_shape(lst):
    if isinstance(lst, list) and lst:
        return (len(lst),) + get_shape(lst[0])
    return ()

# 画图并保存
def draw_results(ori_img, boxes, confs, phrases, img_save_path='./result.jpg'):
    img_h = ori_img.shape[0]
    img_w = ori_img.shape[1]
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_conf = confs[i]
        one_cls = phrases[i]
        x1 = int((one_box[0] - one_box[2] / 2) * img_w)
        y1 = int((one_box[1] - one_box[3] / 2) * img_h)
        x2 = int((one_box[0] + one_box[2] / 2) * img_w)
        y2 = int((one_box[1] + one_box[3] / 2) * img_h)
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(ori_img, f'{one_cls} {one_conf:.2f}', (x1-15, y1-15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (255, 255, 255), fontScale=1.5, thickness=3)
    cv2.imwrite(img_save_path, ori_img)

# 来自 groundingdino.models.GroundingDINO.bertwarper，用于生成包含特殊 token 的注意力掩码、位置编码以及类别到 token 的映射掩码列表
def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list