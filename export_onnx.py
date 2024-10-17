import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

config_file = './groundingdino/config/GroundingDINO_SwinT_OGC.py'
checkpoint_path = './weights/groundingdino_swint_ogc.pth'

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"

    # modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

model = load_model(config_file, checkpoint_path, cpu_only=True)


# 正式推理时使用的提示词，以及相关的mask，具体值可以使用get_caption_mask.py获取。必须在转换onnx前固定好！否则onnx转engine会推理异常
caption = "car ."
input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"]
position_ids = torch.tensor([[0, 0, 1, 0]])
token_type_ids = torch.tensor([[0, 0, 0, 0]])
attention_mask = torch.tensor([[True, True, True, True]])
text_token_mask = torch.tensor([[[True, False, False, False],
                                 [False,  True,  True,  False],
                                 [False,  True,  True,  False],
                                 [False,  False, False, True]]])
# 固定输入分辨率
img = torch.randn(1, 3, 800, 1200)

# onnx模型可以支持动态输入，在转换engine时建议注销
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "position_ids": {0: "batch_size", 1: "seq_len"},
    "token_type_ids": {0: "batch_size", 1: "seq_len"},
    "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
    "img": {0: "batch_size", 2: "height", 3: "width"},
    "logits": {0: "batch_size"},
    "boxes": {0: "batch_size"}
}

torch.onnx.export(
    model,
    f="weights/grounded_opset17_dynamic_4.onnx",
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask), # , zeros, ones),
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    dynamic_axes=dynamic_axes,
    opset_version=17)