import os
import shutil
import torch
from binary_llama.binary_generation import BinaryLlama

os.environ['MASTER_PORT'] = '12345'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

torch.cuda.set_device(0)
torch.cuda.empty_cache()

ckpt_dir = "models/Meta-Llama-3-8B"
tokenizer_path = "models/Meta-Llama-3-8B/tokenizer.model"
model = BinaryLlama.build(ckpt_dir, tokenizer_path, 512, 64)
model.model.set_kk_stage1(torch.tensor([1., 1.]))

save_dir = "models/Meta-Llama-3-8B(with binary parameters)"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.model.state_dict(), os.path.join(save_dir, 'consolidated.00.pth'))

shutil.copy(os.path.join(ckpt_dir, 'params.json'), os.path.join(save_dir, 'params.json'))
