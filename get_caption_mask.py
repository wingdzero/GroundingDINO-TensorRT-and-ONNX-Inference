import time
import numpy as np

from transformers import BertTokenizer
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

# 输入你的文本提示词
caption = ''

captions = [caption]
# 编码文本
# 使用模型的 tokenizer 对 caption 进行分词，并将其转换为张量格式
t0 = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"Loaded BERT tokenizer took {(time.time() - t0):.3f}s")
t0 = time.time()
tokenized = tokenizer(captions, padding="longest", return_tensors="pt").to('cpu')  # padding="longest" 确保在批处理中对齐较短的句子
specical_tokens = tokenizer.convert_tokens_to_ids (["[CLS]", "[SEP]", ".", "?"])  # 将特殊字符（如 [CLS]、[SEP] 等）转换为它们在词汇表中的对应 ID
print(f"Word embedding took {(time.time() - t0):.3f}s")

# 生成注意力掩码和位置信息
# 生成自注意力掩码，位置信息和类别到 token 的映射。这些掩码用于在 Transformer 中对注意力进行控制
t0 = time.time()
(
    text_self_attention_masks,
    position_ids,
    cate_to_token_mask_list,
) = generate_masks_with_special_tokens_and_transfer_map(
    tokenized, specical_tokens, tokenizer)
print(f"Generate attention masks took {(time.time() - t0):.3f}s")

# 处理超长文本
max_text_len = 256
# 如果 caption 的长度超过模型的最大长度 max_text_len，则进行裁剪处理，包括裁剪输入 ID、注意力掩码和 token 类型 ID
if text_self_attention_masks.shape[1] > max_text_len:
    text_self_attention_masks = text_self_attention_masks[
                                :, : max_text_len, : max_text_len]

    position_ids = position_ids[:, : max_text_len]
    tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
    tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]


attention_mask = np.asarray(tokenized["attention_mask"]).astype(bool)
input_dict = {"input_ids": np.asarray(tokenized["input_ids"]), "attention_mask": attention_mask,
         "position_ids": np.asarray(position_ids), "token_type_ids": np.asarray(tokenized["token_type_ids"]), "text_token_mask": np.asarray(text_self_attention_masks)}

print(input_dict)