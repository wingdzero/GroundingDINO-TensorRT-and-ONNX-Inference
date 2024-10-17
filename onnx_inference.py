from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
import onnxruntime as ort
from transformers import BertTokenizer, AutoTokenizer
import bisect
import time

from groundingdino.util.inference import load_image
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def predict(
        ort_session,
        image: np.array,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cpu",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # 1. 文本预处理
    t0 = time.time()
    caption = preprocess_caption(caption=caption)  # 对输入的 caption 进行预处理，去除多余的空格或无效字符
    print(f"Caption processing took {(time.time() - t0):.3f}s")

    # # 2. 模型与数据加载到设备
    # model = model.to(device)
    # image = image.to(device)

    captions = [caption]
    # 3. 编码文本
    # 使用模型的 tokenizer 对 caption 进行分词，并将其转换为张量格式
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Loaded BERT tokenizer took {(time.time() - t0):.3f}s")
    t0 = time.time()
    tokenized = tokenizer(captions, padding="longest", return_tensors="pt").to(device)  # padding="longest" 确保在批处理中对齐较短的句子
    specical_tokens = tokenizer.convert_tokens_to_ids (["[CLS]", "[SEP]", ".", "?"])  # 将特殊字符（如 [CLS]、[SEP] 等）转换为它们在词汇表中的对应 ID
    print(f"Word embedding took {(time.time() - t0):.3f}s")

    # 4. 生成注意力掩码和位置信息
    # 生成自注意力掩码，位置信息和类别到 token 的映射。这些掩码用于在 Transformer 中对注意力进行控制
    t0 = time.time()
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer)
    print(f"Generate attention masks took {(time.time() - t0):.3f}s")

    # 5. 处理超长文本
    max_text_len = 256
    # 如果 caption 的长度超过模型的最大长度 max_text_len，则进行裁剪处理，包括裁剪输入 ID、注意力掩码和 token 类型 ID
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
                                    :, : max_text_len, : max_text_len]

        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    # 6. 执行模型推理
    attention_mask = np.asarray(tokenized["attention_mask"]).astype(bool)

    input_dict = {"img": np.expand_dims(np.asarray(image), axis=0),"input_ids": np.asarray(tokenized["input_ids"]), "attention_mask": attention_mask,
             "position_ids": np.asarray(position_ids), "token_type_ids": np.asarray(tokenized["token_type_ids"]), "text_token_mask": np.asarray(text_self_attention_masks)}
    t0 = time.time()
    outputs = ort_session.run(['logits', 'boxes'], input_dict)
    print(f"Inference time: {(time.time() - t0):.3f}s")

    # 7. 获取预测结果
    prediction_logits = np.apply_along_axis(sigmoid, -1, outputs[0][0])
    # prediction_logits = outputs[0].sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs[1][0] # prediction_boxes.shape = (nq, 4)

    # 8. 应用过滤条件
    # 获取每一行的最大值
    max_values = np.max(prediction_logits, axis=1)
    # 与阈值比较
    mask = max_values > box_threshold
    # mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # 9. 处理文本匹配
    # tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    # 10. 处理特殊标记
    # 如果 remove_combined 为 True，则根据 [SEP] 等特殊标记对文本进行分段处理，否则直接从预测的文本概率图中提取匹配的短语
    # get_phrases_from_posmap: 根据匹配的概率图从文本中提取短语
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append \
                (get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, np.max(logits, axis=1), phrases

if __name__ == '__main__':

    model_path = 'weights/grounded_opset17.onnx'
    img_path = 'images/in/car_1.jpg'
    TEXT_PROMPT = "car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(img_path)

    # 加载 ONNX 模型，创建 InferenceSession
    print("Loading ONNX model")
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 查看当前正在使用的 ExecutionProvider (第一个 provider)
    current_provider = ort_session.get_providers()[0]
    print("Loaded ONNX model, Current Execution Provider:", current_provider)

    boxes, confs, phrases = predict(ort_session, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)

    ori_img = cv2.imread(img_path)
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
        image = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(ori_img, f'{one_cls} {one_conf:.2f}', (x1-15, y1-15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (255, 255, 255), fontScale=1.5, thickness=3)


    cv2.imwrite('./result.jpg', ori_img)






