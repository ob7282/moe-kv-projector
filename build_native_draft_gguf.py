"""
Build native standalone MTP draft GGUF for Qwen 3.6 35B A3B.
Exact hyperparameter matching for llama.cpp qwen35moe architecture.
"""

import torch
import numpy as np
import gguf
from pathlib import Path

BASE_GGUF = r"C:\LocalAI\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
CKPT_PATH = r"C:\Projects\moe-kv-projector\models\hybrid_moe_mtp_advanced.pt"
OUT_GGUF = r"C:\LocalAI\models\qwen35b_mtp_drafter.gguf"

print(f"[1/4] Reading target model metadata from {BASE_GGUF}...")
reader = gguf.GGUFReader(BASE_GGUF)

print(f"[2/4] Loading trained PyTorch weights from {CKPT_PATH}...")
state = torch.load(CKPT_PATH, map_location="cpu")

print(f"[3/4] Initializing GGUFWriter with qwen35moe architecture...")
writer = gguf.GGUFWriter(OUT_GGUF, "qwen35moe")

# Copy tokenizer
for k, v in reader.fields.items():
    if k.startswith("tokenizer."):
        part = v.parts[-1]
        vtype = v.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            items = part.tolist()
            if len(items) > 0 and isinstance(items[0], (bytes, bytearray)):
                items = [x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x) for x in items]
            writer.add_array(k, items)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(k, part.tobytes().decode("utf-8", errors="ignore") if hasattr(part, "tobytes") else str(part))
        elif vtype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(k, int(part[0]))
        elif vtype == gguf.GGUFValueType.INT32:
            writer.add_int32(k, int(part[0]))
        elif vtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(k, float(part[0]))

# Exact hyperparameters required by qwen35moe loader in llama.cpp
writer.add_name("Qwen3.6-35B-A3B-MTP-Custom-Drafter")
writer.add_type("draft")
writer.add_uint32("general.file_type", 0)
writer.add_uint32("general.quantization_version", 2)
writer.add_uint32("qwen35moe.block_count", 2)
writer.add_uint32("qwen35moe.context_length", 32768)
writer.add_uint32("qwen35moe.embedding_length", 2048)
writer.add_uint32("qwen35moe.attention.head_count", 16)
writer.add_uint32("qwen35moe.attention.head_count_kv", 2)
writer.add_array("qwen35moe.rope.dimension_sections", [11, 11, 10, 0])
writer.add_float32("qwen35moe.rope.freq_base", 10000000.0)
writer.add_float32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-06)
writer.add_uint32("qwen35moe.expert_count", 64)
writer.add_uint32("qwen35moe.expert_used_count", 8)
writer.add_uint32("qwen35moe.attention.key_length", 256)
writer.add_uint32("qwen35moe.attention.value_length", 256)
writer.add_uint32("qwen35moe.expert_feed_forward_length", 512)
writer.add_uint32("qwen35moe.expert_shared_feed_forward_length", 512)
writer.add_uint32("qwen35moe.ssm.conv_kernel", 4)
writer.add_uint32("qwen35moe.ssm.state_size", 128)
writer.add_uint32("qwen35moe.ssm.group_count", 16)
writer.add_uint32("qwen35moe.ssm.time_step_rank", 32)
writer.add_uint32("qwen35moe.ssm.inner_size", 4096)
writer.add_uint32("qwen35moe.full_attention_interval", 4)
writer.add_uint32("qwen35moe.rope.dimension_count", 64)
writer.add_uint32("qwen35moe.nextn_predict_layers", 1)

print(f"[4/4] Writing tensors...")
for name, tensor in state.items():
    arr = tensor.detach().cpu().float().numpy()
    if name == "eh_proj.weight":
        writer.add_tensor("blk.0.nextn.eh_proj.weight", arr)
    elif name == "enorm.weight":
        writer.add_tensor("blk.0.nextn.enorm.weight", arr)
    elif name == "hnorm.weight":
        writer.add_tensor("blk.0.nextn.hnorm.weight", arr)
    elif name == "head_norm.weight":
        writer.add_tensor("blk.0.nextn.shared_head_norm.weight", arr)
    elif name == "lm_head.weight":
        writer.add_tensor("output.weight", arr)
    else:
        clean_name = f"blk.0.{name}"
        writer.add_tensor(clean_name, arr)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

size_mb = Path(OUT_GGUF).stat().st_size / (1024 * 1024)
print(f"[SUCCESS] Complete standalone MTP GGUF draft model built: {OUT_GGUF} ({size_mb:.2f} MB)")
