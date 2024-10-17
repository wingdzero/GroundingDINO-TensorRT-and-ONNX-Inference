import argparse
import os
import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from openvino.tools.mo import convert_model
from openvino.runtime import serialize

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    
    #modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def export_openvino(model, output_dir):
    caption = "the running dog ."  # ". ".join(input_text)
    input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"]
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    text_token_mask = torch.tensor([[[ True, False, False, False, False, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False, False, False, False, False,  True]]])
    
    img = torch.randn(1, 3, 800, 1200)

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

    # export onnx model
    torch.onnx.export(
        model,
        f="./grounded.onnx",
        args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask), #, zeros, ones),
        input_names=["img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16)
    
    #convert_model returns an openvino.runtime.Model object
    ov_model = convert_model("./grounded.onnx")
    
    # then model can be serialized to *.xml & *.bin files
    serialize(ov_model, xml_path=output_dir + '/groundingdino.xml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to IR", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    
    #export openvino
    export_openvino(model, output_dir)



    
