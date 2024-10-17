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

caption = "the running dog ."
input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"]
position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
attention_mask = torch.tensor([[True, True, True, True, True, True]])
text_token_mask = torch.tensor([[[True, False, False, False, False, False],
                                 [False,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True, False],
                                 [False,  True,  True,  True,  True, False],
                                 [False, False, False, False, False,  True]]])

img = torch.randn(1, 3, 800, 1200)  # Fixing img shape to (1, 3, 800, 1200)

# Remove dynamic_axes for fixed input shapes
# export onnx model

scripted_model = torch.jit.script(model)

scripted_model.save("./grounded_torchscript_fixed.pt")

print("TorchScript 模型已成功导出。")

torch.onnx.export(
    scripted_model,
    f="./grounded_torchscript_opset17_fixed.onnx",
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
    input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    opset_version=17
)
print("ONNX 模型已成功导出。")