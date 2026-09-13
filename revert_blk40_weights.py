# Revert blk.40 MTP weights back to original from backup
import torch
from pathlib import Path

GGUF_PATH = Path(r'C:\LocalAI\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf')
BAK_PATH = Path(r'C:\LocalAI\models\blk40_original_weights.bak')

if not BAK_PATH.exists():
    print(f'[ERROR] Backup file {BAK_PATH} does not exist!')
    exit(1)

import gguf
reader = gguf.GGUFReader(str(GGUF_PATH))
tensors = {t.name: t for t in reader.tensors}

target_tensors = {
    'eh_proj': 'blk.40.nextn.eh_proj.weight',
    'enorm': 'blk.40.nextn.enorm.weight',
    'hnorm': 'blk.40.nextn.hnorm.weight',
    'head_norm': 'blk.40.nextn.shared_head_norm.weight',
}

backup_data = torch.load(BAK_PATH, map_location='cpu')

print('[1/2] Restoring original weights...')
with open(GGUF_PATH, 'r+b') as f:
    for k, name in target_tensors.items():
        t = tensors[name]
        f.seek(t.data_offset)
        f.write(backup_data[k])
        print(f'  - Restored {name} ({t.n_bytes} bytes)')

print('[SUCCESS] Original blk.40 weights fully restored!')
