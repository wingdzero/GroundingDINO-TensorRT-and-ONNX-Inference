# GroundingDINO TensorRT&ONNX Inference

## 介绍
本项目基于TensorRT与ONNXRuntime实现高性能推理。

推理ONNX模型时支持动态输入，推理engine模型时目前仅支持固定输入，需要在engine模型转换前提前配置好文本提示词与输入图像大小。
其中ONNX模型转换部分参考了(https://github.com/wenyi5608/GroundingDINO)以及(https://github.com/hpc203/GroundingDINO-onnxrun?tab=readme-ov-file)

项目实现了基于Fastapi的访问接口。

## 快速上手

### 1. 环境配置
- TensorRT版本：

    本项目使用的TensorRT版本为10.1
- 安装项目依赖：

```angular2html
pip install -r requirements.txt
```

### 2. pt模型推理
```angular2html
$ python pt_inference.py
```

### 3. ONNX模型导出及推理

导出ONNX模型主要在groundingdino\models\GroundingDINO\groundingdino.py里的forward函数中将attention mask、position id等放在模型外部先提前提取好

- 导出onnx模型（文本部分可以支持动态输入，如果后续需要转换成engine模型则需要固定好你的文本输入shape，取值可以使用get_caption_mask.py得到）：
```angular2html
$ python export_onnx.py
```
- 推理代码中有详细的注释解释如何运行
```angular2html
$ python onnx_inference.py
```


### 3. Engine模型导出及推理

- engine模型由ONNX模型转换得到，使用如下脚本进行转换（目前仅支持固定shape输入）：

```angular2html
$ python export_engine.py
```
- engine模型推理：
```angular2html
$ python engine_inference.py
```

### 4. api推理
- 配置推理权重及相关参数：

  模型路径、置信度阈值、文本提示词等参数存放在 config.ini 中

- 首先启动engine推理 api服务：
```angular2html
$ python api_inference_server.py
```

- 之后发送图片并返回结果（文本提示词只需要在推理前输入一次，也可以在推理过程中修改）：
```angular2html
$ python send.py
```

# Acknowledgments

Provided codes were adapted from:


- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [wenyi5608/GroundingDINO](https://github.com/wenyi5608/GroundingDINO)
- [hpc203/GroundingDINO](https://github.com/hpc203/GroundingDINO-onnxrun?tab=readme-ov-file)