# P1 机房跑批手册（知新楼 208，RTX 5090D 24GB）

> 目标：**一场 5 小时把 6 次完整实验跑完、当场导出 ZIP、当场备份。**
> 代码已在本机验证：`check_solution.py` PASS + CPU smoke 通过。机房只做"跑正式的"。

---

## 0. 进场 10 分钟内先做这 3 件事

```bash
# ① 确认机器和账号（用组分配到的 workstation + user1–user5）
nvidia-smi                      # 看到 RTX 5090D、显存约 24GB、没有别人的进程在跑

# ② 激活课程环境（已装好 torch>=2.9，含 torch.optim.Muon）
source /opt/dda5001/venv/bin/activate
python -c "import torch; print(torch.__version__, hasattr(torch.optim,'Muon'))"
#   期望输出：2.x True    ← 若 False 或 import 失败，立刻叫 TA，别自己装环境

# ③ 进项目目录（把代码拷到这台机器上，别在共享目录里跑）
cd ~/p1        # 或你们自己的工作目录
python check_solution.py
#   期望：PASS: causal mask, shifted positions, provided scores, native optimizer contracts, exactly-once steps and plots
```

三件事任一不过 → **先解决，别浪费机时开跑**。

---

## 1. 时间预算（一场 5 小时）

| 阶段 | 内容 | 预计 |
|---|---|---|
| 进场检查 | 上面 3 步 + CPU smoke | 15 分钟 |
| 前三次 | `adamw-default` → `muon-default` → `sgdm-default` | 约 90 分钟（每个 ~30 分钟） |
| 看曲线定 LR | 看三份 `metrics.csv` 的最后一行，决定 3 个自选 LR | 10 分钟 |
| 后三次 | `adamw-other` → `muon-other` → `sgdm-other` | 约 90 分钟 |
| 出图 + 生成文本 | `plot.py` × 7、`sample.py` | 15 分钟 |
| 导出 + 备份 | `export_results.py` + 推私有仓库 | 20 分钟 |
| 缓冲 | 出问题重跑某一个 | 剩余时间 |

**关键：每跑完一个立刻把 `metrics.csv` 最后一行抄进下面的记录表**，别等最后一起找——5 小时结束时机器会重启，文件全没。

---

## 2. 逐条命令（照顺序，一个跑完再跑下一个）

```bash
# 冒烟（不占 GPU，30 秒）
python train.py config/smoke.py --device=cpu --out_dir=checks/cpu

# ── 三次默认（LR 已给定，不许改）──
python train.py config/train_shakespeare_char.py     # AdamW 3e-4  → runs/adamw-default
python train.py config/muon-default.py               # Muon  1e-2  → runs/muon-default
python train.py config/sgdm-default.py               # SGDM  0.1   → runs/sgdm-default

# ── 看曲线，决定三个自选 LR ──
for d in runs/*-default; do echo "== $d"; tail -1 $d/metrics.csv; done

# ── 三次自选（config 里已预填：AdamW 6e-4、Muon 2e-2、SGDM 0.05；按曲线可改）──
python train.py config/adamw-other.py                # → runs/adamw-other
python train.py config/muon-other.py                 # → runs/muon-other
python train.py config/sgdm-other.py                 # → runs/sgdm-other
```

**选自选 LR 的判据（手册原话的落地版）**：看默认那次 `metrics.csv` 的 val loss 曲线

| 默认那次的表现 | 自选 LR 怎么改 |
|---|---|
| 平滑下降但很慢、终点还高 | 调**大**（×2 起） |
| 中途抖动大、或上升/发散 | 调**小**（÷2 起） |
| 已经很好、曲线漂亮 | 仍要换一个值（任务要求"另一个学习率"），取 ÷2 或 ×2 |

改法：只动 `config/xxx-other.py` 里**对应优化器那一个学习率字段**和 `out_dir`，其余不动。

---

## 3. 出图、生成文本、导出

```bash
# 单 run 图（train/val 两条曲线，两种 x 轴）
for d in runs/*; do python plot.py --run-dir=$d; done

# 六组对比图（报告里那张）→ runs/comparison_*.png
python plot.py --runs-dir=runs

# 代表性文本样本（注明这是哪个优化器 + 哪个 LR，写进报告）
python sample.py --checkpoint=runs/adamw-default/last.pt  --tokens=400
python sample.py --checkpoint=runs/muon-other/last.pt     --tokens=400

# 导出提交用的 ZIP（不含 checkpoint、不含数据集）
python export_results.py --runs-dir=runs --output=part1-results.zip
```

**备份（走之前必须做完，做得完再离开）**：

```bash
# 方案 A：推自己的私有 GitHub 仓库（推荐）
git add -A && git commit -m "p1 results six runs" && git push
# 方案 B：拷到 U 盘 / 上传网盘
cp -r runs part1-results.zip ~/Desktop/   # 若挂了外接盘就拷到盘里
```

⚠️ 工作站重启会清空你保存的一切，checkpoint（`last.pt` / `best.pt`）**必须**带走一份，助教可能抽查。

---

## 4. 报告要抄的数值（表格照这个填）

| run | 优化器 | 学习率 | 终点 val loss（最后一行的 val_loss） | 训练耗时（最后一行的 train_seconds） | 备注（有无抖动/发散/交叉） |
|---|---|---|---|---|---|
| adamw-default | AdamW | 3e-4 |  |  |  |
| adamw-other | AdamW | 6e-4* |  |  |  |
| muon-default | Muon | 1e-2 |  |  |  |
| muon-other | Muon | 2e-2* |  |  |  |
| sgdm-default | SGDM | 0.1 |  |  |  |
| sgdm-other | SGDM | 0.05* |  |  |  |

\* 自选值以你们现场实际改的为准，报告里写实际用的值。

一键抄全部：`for d in runs/*; do echo "== $d"; tail -1 $d/metrics.csv; done`

---

## 5. 出事怎么办（现场判断表）

| 现象 | 原因 | 处置 |
|---|---|---|
| `FileExistsError` / `out.mkdir(... exist_ok=False)` 报错 | 同一个 `out_dir` 已经存在（上一次跑过） | 换个 `out_dir` 名字，或先 `rm -rf runs/xxx` |
| 显存不够 `torch.OutOfMemoryError` | batch 太大 | 用 `setup/README.md` 的改法：`batch_size=8`、`gradient_accumulation_steps=9`（乘积不变，预算不变），六个 run 全部统一用同一套硬件设置，并在报告里说明 |
| loss 变 `nan` / `summary.json` 里 `"status": "failed"` | 学习率太大、发散 | 记下这个 LR 作为"发散证据"，换更小的值重跑这一次 |
| 进程被 kill / 断电 | 机器问题 | `summary.json` 里 `status` 不是 completed 就算没跑完；用 `--resume` 续训或直接重跑 |
| 跑得比预期慢很多 | 有人在跑别的任务 | 看 `nvidia-smi`，报告 TA，别硬等 |

---

## 6. 不能做的事（会被判违规/扣分）

- 改 `check_solution.py`、`evaluate.py`、`AGENTS.md` 里的任何东西；
- 改课程固定的训练设置（`model_preset`、`block_size`、`batch_size`、`max_tokens`、`eval_interval`、`lr_schedule`、`warmup_iters`、`dropout`、`seed` 等）——**除非**是 `setup/README.md` 允许的小显存改法；
- 六个 run 用不同的硬件设置（必须统一，否则不可比）；
- 把 checkpoint 或数据集塞进提交的 ZIP；
- 用 `best.pt` 的数值当"终点"写进报告。
