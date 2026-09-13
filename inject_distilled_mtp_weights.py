# Patch blk.40 MTP weights directly into Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
import torch
import numpy as np
import gguf
from pathlib import Path

GGUF_PATH = Path(r'C:\LocalAI\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf')
CKPT_PATH = Path(r'C:\Projects\moe-kv-projector\models\hybrid_moe_mtp_finetuned.pt')
BAK_PATH = Path(r'C:\LocalAI\models\blk40_original_weights.bak')

print('[1/4] Reading GGUF tensor offsets...')
reader = gguf.GGUFReader(str(GGUF_PATH))
tensors = {t.name: t for t in reader.tensors}

target_tensors = {
    'eh_proj': 'blk.40.nextn.eh_proj.weight',
    'enorm': 'blk.40.nextn.enorm.weight',
    'hnorm': 'blk.40.nextn.hnorm.weight',
    'head_norm': 'blk.40.nextn.shared_head_norm.weight',
}

offsets = {}
sizes = {}
for k, name in target_tensors.items():
    t = tensors[name]
    offsets[k] = t.data_offset
    sizes[k] = t.n_bytes
    print(f'  - {name}: offset={t.data_offset}, size={t.n_bytes} bytes')

print('[2/4] Backing up original tensor bytes...')
backup_data = {}
with open(GGUF_PATH, 'rb') as f:
    for k, offset in offsets.items():
        f.seek(offset)
        backup_data[k] = f.read(sizes[k])

if not BAK_PATH.exists():
    torch.save(backup_data, BAK_PATH)
    print(f'  - Saved backup to {BAK_PATH} ({sum(len(b) for b in backup_data.values())} bytes)')
else:
    print(f'  - Backup already exists at {BAK_PATH}')

print('[3/4] Quantizing and preparing new weights from checkpoint...')
state = torch.load(CKPT_PATH, map_location='cpu')

new_bytes = {}
# enorm (F32)
new_bytes['enorm'] = state['enorm.weight'].detach().cpu().float().numpy().tobytes()
# hnorm (F32)
new_bytes['hnorm'] = state['hnorm.weight'].detach().cpu().float().numpy().tobytes()
# shared_head_norm (F32)
new_bytes['head_norm'] = state['head_norm.weight'].detach().cpu().float().numpy().tobytes()

# eh_proj (Q8_0)
eh_arr = state['eh_proj.weight'].detach().cpu().float().numpy()
eh_q8 = gguf.quantize(eh_arr, gguf.GGMLQuantizationType.Q8_0)
new_bytes['eh_proj'] = eh_q8.tobytes()

for k, b in new_bytes.items():
    assert len(b) == sizes[k], f'Size mismatch for {k}: {len(b)} != {sizes[k]}'
    print(f'  - {k}: verified exact size match ({len(b)} bytes)')

print('[4/4] Writing new weights in-place into GGUF...')
with open(GGUF_PATH, 'r+b') as f:
    for k, offset in offsets.items():
        f.seek(offset)
        f.write(new_bytes[k])

print('[SUCCESS] Distilled MTP weights successfully injected into blk.40 of Qwen 3.6 35B!')
