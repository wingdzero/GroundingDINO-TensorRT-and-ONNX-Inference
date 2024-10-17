#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time
import bisect
import argparse
import tensorrt as trt

# 将当前脚本目录添加到系统路径中
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from transformers import BertTokenizer
from utils.utils import *


class TensorRTInfer:

    def __init__(self, engine_path):
        t0 = time.time()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(f"Loaded BERT tokenizer took {(time.time() - t0):.3f}s")
        self.tokenized = None
        self.captions = None
        self.text_self_attention_masks = None
        self.position_ids = None
        self.cate_to_token_mask_list = None
        self.attention_mask = None


        # 加载 TensorRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)  # 记录推理过程中的错误和警告日志
        # self.logger = trt.Logger(trt.Logger.VERBOSE)  # 记录更详细信息
        trt.init_libnvinfer_plugins(self.logger, namespace="")  # 初始化 TensorRT 的所有插件库
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime  # 确保 runtime 正常加载
            self.engine = runtime.deserialize_cuda_engine(f.read())  # 反序列化引擎
        assert self.engine
        self.context = self.engine.create_execution_context()  # 创建推理执行上下文，用于管理推理过程中实际的计算和资源调度
        assert self.context

        # # 用于获取每层名字及运行时间
        # profiler = LayerProfiler()
        # self.context.profiler = profiler

        # 为输入和输出的张量分配 GPU 内存，并将其与实际的数据绑定
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):  # 逐个获取模型的 tensor，并判断是输入 tensor 还是输出 tensor
            name = self.engine.get_tensor_name(i)  # 获取 tensor 的名称
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # 判断是否为输入张量
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))  # 获取 tensor 的数据类型
            shape = self.context.get_tensor_shape(name)  # 获取 tensor 的形状
            if is_input and shape[0] < 0:  # 动态输入时调整为最大形状
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # profile 包含 min、opt、max
                self.context.set_input_shape(name, profile_shape[2])  # 设置最大输入 shape
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]  # 设置 batch 大小
            size = dtype.itemsize
            for s in shape:
                size *= s  # 计算 tensor 所需的内存大小
            # print(f"{name} size: {size}")
            allocation = cuda_call(cudart.cudaMalloc(size))  # 使用 pycuda 为 tensor 分配 GPU 内存
            host_allocation = None if is_input else np.zeros(shape, dtype)  # 如果是输出 tensor，则在 CPU 上分配内存
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)  # 存储输入 tensor 信息
            else:
                self.outputs.append(binding)  # 存储输出 tensor 信息
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        # 确保有有效的 batch_size、输入和输出张量
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        获取模型输入 tensor 的规格（形状和数据类型）
        return: 输入 tensor 的 shape 和 numpy 数据类型
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def caption_tokenize(self, captions, max_text_len=256, device="cpu"):

        t0 = time.time()
        print(f"Start tokenizing captions: {captions}")
        self.captions = captions
        # 1. 文本预处理
        captions = preprocess_caption(caption=captions)  # 对输入的 caption 进行预处理，去除多余的空格或无效字符
        # 2. 编码文本
        self.tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            device)  # padding="longest" 确保在批处理中对齐较短的句子
        specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"])  # 将特殊字符（如 [CLS]、[SEP] 等）转换为它们在词汇表中的对应 ID
        # 3. 生成注意力掩码和位置信息
        # 生成自注意力掩码，位置信息和类别到 token 的映射。这些掩码用于在 Transformer 中对注意力进行控制
        (
            self.text_self_attention_masks,
            self.position_ids,
            self.cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            self.tokenized, specical_tokens, self.tokenizer)
        # 4. 处理超长文本
        max_text_len = max_text_len
        # 如果 caption 的长度超过模型的最大长度 max_text_len，则进行裁剪处理，包括裁剪输入 ID、注意力掩码和 token 类型 ID
        if self.text_self_attention_masks.shape[1] > max_text_len:
            self.text_self_attention_masks = self.text_self_attention_masks[:, : max_text_len, : max_text_len]
            self.position_ids = self.position_ids[:, : max_text_len]
            self.tokenized["input_ids"] = self.tokenized["input_ids"][:, : max_text_len]
            self.tokenized["attention_mask"] = self.tokenized["attention_mask"][:, : max_text_len]
            self.tokenized["token_type_ids"] = self.tokenized["token_type_ids"][:, : max_text_len]
        attention_mask = np.asarray(self.tokenized["attention_mask"]).astype(bool)

        memcpy_host_to_device(self.inputs[1]["allocation"], np.asarray(self.tokenized["input_ids"]))
        memcpy_host_to_device(self.inputs[2]["allocation"], attention_mask)
        memcpy_host_to_device(self.inputs[3]["allocation"], np.asarray(self.position_ids))
        memcpy_host_to_device(self.inputs[4]["allocation"], np.asarray(self.tokenized["token_type_ids"]))
        memcpy_host_to_device(self.inputs[5]["allocation"], np.asarray(self.text_self_attention_masks))
        print(f"\nProcess caption successfully, took {(time.time() - t0):.3f}s")


    def infer(self, img_data):
        """
            执行 batch 推理。图片应当已经进行 batch 打包与前处理。
            param img_data: 图片数据 (ndarray).
            param scales: 图片经过 letterbox 后的压缩比 r 与原图在新图上的左上角坐标 left top
            param nc: 模型识别的类别总数
            param nms_threshold: NMS 时的 IoU 阈值
            return: list, 由每个 batch 的框的 boxes, scores, classes 组成
        """

        # 图像前处理
        memcpy_host_to_device(self.inputs[0]["allocation"], img_data)

        # 执行推理
        t0 = time.time()
        self.context.execute_v2(self.allocations)
        print(f'Inference time: {(time.time() - t0):.3f}s')

        # 将输出数据从 GPU 内存复制到 CPU 内存
        for o in range(len(self.outputs)):
            memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]

    def post_process(self, inference_result, box_threshold, text_threshold, remove_combined: bool = False):

        t0 = time.time()
        # 1. 解析输出结果
        prediction_logits = np.apply_along_axis(sigmoid, -1, inference_result[0][0])
        prediction_boxes = inference_result[1][0]  # prediction_boxes.shape = (nq, 4)

        # 2. 应用过滤条件
        # 获取每一行的最大值
        max_values = np.max(prediction_logits, axis=1)
        # 与阈值比较
        mask = max_values > box_threshold
        # mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        # 3. 处理文本匹配
        tokenized = self.tokenizer(self.captions)

        # 4. 处理特殊标记
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
                    (get_phrases_from_posmap(logit > text_threshold, tokenized, self.tokenizer, left_idx, right_idx).replace(
                        '.', ''))
        else:
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, self.tokenizer).replace('.', '')
                for logit
                in logits
            ]
        print(f"Post processing time: {(time.time() - t0):.3f}s")
        return boxes, np.max(logits, axis=1), phrases

    def cleanup(self):
        # 释放 GPU 内存
        for i in range(len(self.inputs)):
            cuda_call(cudart.cudaFree(self.inputs[i]["allocation"]))
        for o in range(len(self.outputs)):
            cuda_call(cudart.cudaFree(self.outputs[o]["allocation"]))


def main(args):

    # 定义支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png')

    img_input_path = args.input
    TEXT_PROMPT = args.text_prompt
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold
    print(f"Input args: inference engine: {args.engine}, input img path: {img_input_path}, input text prompt: {TEXT_PROMPT}, box threshold: {BOX_TRESHOLD}, text threshold: {TEXT_TRESHOLD}")

    if args.output:
        output_dir = os.path.realpath(args.output)
        os.makedirs(output_dir, exist_ok=True)

    print("Building Engine")
    trt_infer = TensorRTInfer(args.engine)
    print("Build Engine successfully")

    # 如果给定输入，则进行推理
    if args.input:

        # 对文本提示词进行tokenize，并写入GPU内存中
        trt_infer.caption_tokenize(TEXT_PROMPT)

        # 文件夹推理
        if os.path.isdir(args.input):

            print(f"\nInferring data in {img_input_path}\n")

            for file_name in os.listdir(img_input_path):
                file_path = os.path.join(img_input_path, file_name)
                print(f"Inferring {file_path}")

                if os.path.isfile(file_path) and file_name.lower().endswith(image_extensions):

                    t0 = time.time()
                    image, image_transformed = load_image(file_path)
                    input_img = np.expand_dims(np.asarray(image_transformed), axis=0)
                    print(f"Preprocessing time: {(time.time() - t0):.3f}s")

                    # 执行推理
                    outputs = trt_infer.infer(input_img)

                    # 输出结果处理
                    boxes, confs, phrases = trt_infer.post_process(outputs, BOX_TRESHOLD, TEXT_TRESHOLD)

                    # 画图
                    file_name, file_extension = os.path.splitext(file_name)
                    save_file_name = f"{file_name}_result{file_extension}"
                    draw_results(image, boxes, confs, phrases, img_save_path=os.path.join(args.output, save_file_name))
                    print(f"Saved {file_path} results in {os.path.join(args.output, save_file_name)}")

            # 推理完成后清理GPU内存
            trt_infer.cleanup()
            print("\nMemory cleanup finished")

        # 单张图片推理
        else:
            if os.path.isfile(img_input_path) and img_input_path.lower().endswith(image_extensions):
                print(f"Inferring image {img_input_path}")
                file_name = os.path.basename(img_input_path)

                t0 = time.time()
                image, image_transformed = load_image(img_input_path)
                input_img = np.expand_dims(np.asarray(image_transformed), axis=0)
                print(f"Preprocessing time: {(time.time() - t0):.3f}s")

                # 执行推理
                outputs = trt_infer.infer(input_img)

                # 输出结果处理
                boxes, confs, phrases = trt_infer.post_process(outputs, BOX_TRESHOLD, TEXT_TRESHOLD)

                # 画图
                file_name, file_extension = os.path.splitext(file_name)
                save_file_name = f"{file_name}_result{file_extension}"
                draw_results(img_input_path, boxes, confs, phrases, img_save_path=os.path.join(args.output, save_file_name))
                print(f"Saved {img_input_path} results in {os.path.join(args.output, save_file_name)}")

            else:
                print("Please give an image path or image folder path")


    # 未给定输入，执行推理测试
    else:
        print("No input provided, running in benchmark mode")
        spec = trt_infer.input_spec()
        batch = 255 * np.random.rand(*spec[0]).astype(spec[1])
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(1000 * np.average(times)))
        print(
            "Average Throughput: {:.1f} ips".format(
                trt_infer.batch_size / np.average(times)
            )
        )
    print("Finished Processing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default='images/in',
        help="Path to the image or directory to process",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default='weights/grounded_opset17_fixed_fp16_4.engine',
        help="The serialized TensorRT engine",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='images/out',
        help="Directory where to save the visualization results",
    )
    parser.add_argument(
        "-t",
        "--text_prompt",
        default='car .',
        help="Target to detect(one word)",
    )
    parser.add_argument(
        "-bt",
        "--box_threshold",
        default=0.35,
        help="Box threshold",
    )
    parser.add_argument(
        "-tt",
        "--text_threshold",
        default=0.25,
        help="Text threshold",
    )
    args = parser.parse_args()
    main(args)