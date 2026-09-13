"""
GGUF Exporter for Hybrid MoE MTP Drafter
Serializes PyTorch checkpoint to GGUF format.
"""

import torch
import numpy as np
import gguf
from pathlib import Path

from config import MODELS_DIR, D_MODEL
from model_mtp_drafter import MTP_VOCAB_SIZE

def export_drafter_to_gguf():
    ckpt_path = MODELS_DIR / "hybrid_moe_mtp_advanced.pt"
    out_gguf_path = MODELS_DIR / "qwen35b_hybrid_moe_drafter.gguf"

    print(f"[EXPORT] Loading PyTorch weights from {ckpt_path.name}...")
    state = torch.load(ckpt_path, map_location="cpu")

    writer = gguf.GGUFWriter(str(out_gguf_path), "qwen35")
    writer.add_name("Qwen3.6-35B-Hybrid-MoE-MTP-Drafter")
    writer.add_type("draft")
    writer.add_uint32("general.file_type", 0)  # FP32/FP16 unquantized
    writer.add_uint32("qwen35.embedding_length", D_MODEL)
    writer.add_uint32("qwen35.vocab_size", MTP_VOCAB_SIZE)
    writer.add_uint32("qwen35.expert_count", 64)
    writer.add_uint32("qwen35.expert_used_count", 8)

    print(f"[EXPORT] Writing {len(state)} tensors into {out_gguf_path.name}...")
    for name, tensor in state.items():
        arr = tensor.detach().cpu().float().numpy()
        # Clean GGUF tensor naming convention
        clean_name = name.replace("eh_proj", "nextn.eh_proj")
        clean_name = clean_name.replace("enorm", "nextn.enorm")
        clean_name = clean_name.replace("hnorm", "nextn.hnorm")
        clean_name = clean_name.replace("head_norm", "nextn.shared_head_norm")
        writer.add_tensor(clean_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_gguf_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Exported GGUF drafter to {out_gguf_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_drafter_to_gguf()
