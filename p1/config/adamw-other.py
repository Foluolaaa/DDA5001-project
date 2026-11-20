# AdamW 自选学习率：默认 3e-4 的【2 倍】= 6e-4
# 说明：跑完 adamw-default 之后看曲线再定；
#   - 若默认曲线下降又慢又平滑 → 保持/再加（如 1e-3）
#   - 若已出现抖动 → 改小（如 1.5e-4）
# 只改 learning_rate 和 out_dir 两处，其余一律不动。
optimizer_name = 'adamw'
out_dir = 'runs/adamw-other'

learning_rate = 6e-4            # AdamW  ← 自选值（默认 3e-4）
muon_learning_rate = 1e-2       # Muon（本次不用）
sgdm_learning_rate = 0.1        # SGDM（本次不用）

# Fixed course settings.
model_preset = 'gpt2-medium'
block_size = 256
batch_size = 72
gradient_accumulation_steps = 1
max_tokens = 92160000           # 5,000 updates
max_seconds = 0.0
eval_interval = 500
eval_iters = 8

warmup_iters = 100
lr_schedule = 'cosine'
lr_decay_iters = 5000
min_lr_ratio = 0.1
lr_multiplier = 1.0

dropout = 0.2
weight_decay = 0.0
grad_clip = 1.0
beta1 = 0.9
beta2 = 0.99
muon_momentum = 0.95
sgdm_momentum = 0.9
sgdm_nesterov = False
seed = 42
eval_seed = 1729
dtype = 'auto'
compile = True
