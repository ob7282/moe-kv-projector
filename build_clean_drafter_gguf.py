# Build clean standalone MTP draft GGUF
import torch
import numpy as np
import gguf
from pathlib import Path

BASE_GGUF = r'C:\LocalAI\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf'
CKPT_PATH = r'C:\Projects\moe-kv-projector\models\hybrid_moe_mtp_advanced.pt'
OUT_GGUF = r'C:\LocalAI\models\qwen35b_hybrid_moe_advanced.gguf'

print('[1/4] Reading base GGUF metadata...')
reader = gguf.GGUFReader(BASE_GGUF)

print('[2/4] Loading trained PyTorch weights...')
state = torch.load(CKPT_PATH, map_location='cpu')

print('[3/4] Initializing GGUFWriter with qwen35moe...')
writer = gguf.GGUFWriter(OUT_GGUF, 'qwen35moe')

writer.add_name('Qwen3.6-35B-Hybrid-MoE-Advanced-Drafter')
writer.add_type('draft')
writer.add_file_type(gguf.GGMLQuantizationType.F32)
writer.add_quantization_version(2)
writer.add_block_count(2)
writer.add_context_length(32768)
writer.add_embedding_length(2048)
writer.add_head_count(16)
writer.add_head_count_kv(2)
writer.add_rope_dimension_sections([11, 11, 10, 0])
writer.add_rope_freq_base(10000000.0)
writer.add_layer_norm_rms_eps(1e-06)
writer.add_expert_count(64)
writer.add_expert_used_count(8)
writer.add_key_length(256)
writer.add_value_length(256)
writer.add_expert_feed_forward_length(512)
writer.add_expert_shared_feed_forward_length(512)
writer.add_ssm_conv_kernel(4)
writer.add_ssm_state_size(128)
writer.add_ssm_group_count(16)
writer.add_ssm_time_step_rank(32)
writer.add_ssm_inner_size(4096)
writer.add_full_attention_interval(4)
writer.add_rope_dimension_count(64)
writer.add_nextn_predict_layers(1)

if 'tokenizer.ggml.model' in reader.fields:
    m = reader.fields['tokenizer.ggml.model'].parts[-1].tobytes().decode('utf-8')
    writer.add_tokenizer_model(m)
if 'tokenizer.ggml.pre' in reader.fields:
    p = reader.fields['tokenizer.ggml.pre'].parts[-1].tobytes().decode('utf-8')
    writer.add_tokenizer_pre(p)
if 'tokenizer.ggml.tokens' in reader.fields:
    tokens = [bytes(part) for part in reader.fields['tokenizer.ggml.tokens'].parts]
    writer.add_token_list(tokens)
if 'tokenizer.ggml.token_type' in reader.fields:
    types = [int(x) for x in reader.fields['tokenizer.ggml.token_type'].parts[-1]]
    writer.add_token_types(types)
if 'tokenizer.ggml.merges' in reader.fields:
    merges = [bytes(part) for part in reader.fields['tokenizer.ggml.merges'].parts]
    writer.add_token_merges(merges)
if 'tokenizer.ggml.bos_token_id' in reader.fields:
    writer.add_bos_token_id(int(reader.fields['tokenizer.ggml.bos_token_id'].parts[-1][0]))
if 'tokenizer.ggml.eos_token_id' in reader.fields:
    writer.add_eos_token_id(int(reader.fields['tokenizer.ggml.eos_token_id'].parts[-1][0]))
if 'tokenizer.ggml.padding_token_id' in reader.fields:
    writer.add_pad_token_id(int(reader.fields['tokenizer.ggml.padding_token_id'].parts[-1][0]))

print('[4/4] Writing tensors...')
for name, tensor in state.items():
    arr = tensor.detach().cpu().float().numpy()
    if name == 'eh_proj.weight':
        writer.add_tensor('blk.0.nextn.eh_proj.weight', arr)
    elif name == 'enorm.weight':
        writer.add_tensor('blk.0.nextn.enorm.weight', arr)
    elif name == 'hnorm.weight':
        writer.add_tensor('blk.0.nextn.hnorm.weight', arr)
    elif name == 'head_norm.weight':
        writer.add_tensor('blk.0.nextn.shared_head_norm.weight', arr)
    elif name == 'lm_head.weight':
        writer.add_tensor('output.weight', arr)
    else:
        clean_name = f'blk.0.{name}'
        writer.add_tensor(clean_name, arr)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

size_mb = Path(OUT_GGUF).stat().st_size / (1024 * 1024)
print(f'[SUCCESS] Clean standalone MTP GGUF drafter built: {OUT_GGUF} ({size_mb:.2f} MB)')
