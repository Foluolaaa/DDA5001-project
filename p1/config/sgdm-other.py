# SGDM 自选学习率：默认 0.1 的【1/2】= 0.05
# 说明：SGDM 没有逐参数步长自适应，0.1 在 5000 步 cosine 下已经偏大，
#   翻倍容易发散；先减半更安全，也能展示「学习率偏小 → 收敛更慢」的对照。
# 只改 sgdm_learning_rate 和 out_dir 两处，其余一律不动。
optimizer_name = 'sgdm'
out_dir = 'runs/sgdm-other'

learning_rate = 3e-4            # AdamW（本次不用）
muon_learning_rate = 1e-2       # Muon（本次不用）
sgdm_learning_rate = 5e-2       # SGDM  ← 自选值（默认 0.1）

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
