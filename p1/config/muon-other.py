# Muon 自选学习率：默认 1e-2 的【2 倍】= 2e-2
# 说明：Muon 先把更新矩阵正交化再乘学习率，步长尺度比 AdamW/SGDM 稳定，
#   默认 1e-2 附近翻倍通常仍能收敛；跑完 muon-default 后看曲线再定，抖得厉害就改成 5e-3。
# 只改 muon_learning_rate 和 out_dir 两处，其余一律不动。
optimizer_name = 'muon'
out_dir = 'runs/muon-other'

learning_rate = 3e-4            # AdamW（辅助优化器用，不要改）
muon_learning_rate = 2e-2       # Muon  ← 自选值（默认 1e-2）
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
