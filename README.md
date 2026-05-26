# Qwen RL Math Solver

本仓库用于在小学数学题上微调 `Qwen/Qwen2.5-0.5B-Instruct`。当前支持三条流程：

- 不训练，直接做 CoT 推理。
- 先用 GLM/Kimi 过滤后的合成 CoT 数据做 SFT，再做 DPO。
- 先用同一批合成 CoT 数据做 SFT，再用规则奖励做 GRPO。

## 环境安装

支持 Python `>=3.10,<3.13`。下面所有命令默认都在仓库根目录执行。

### 使用 uv

```bash
uv sync
source .venv/bin/activate
```

### 使用 conda

```bash
conda create -n qwenrl python=3.12 -y
conda activate qwenrl
python -m pip install -e .
```

如果机器上的 CUDA/PyTorch 需要指定 wheel，先安装匹配的 PyTorch，再执行
`python -m pip install -e .`。

## 模型端点配置

数据合成和评审需要 OpenAI-compatible 接口，配置文件为
`configs/model/models.yml`。这个文件包含内部地址和 key，不提交到仓库。

按 `configs/prompt/*.yml` 里引用的 model key 创建配置：

```yaml
glm-5.1-w4a8:
  model: glm-5.1-w4a8
  base_url: http://YOUR_ENDPOINT/v1
  api_key: YOUR_API_KEY
  timeout: 60
  max_retries: 2

kimi-k2.6:
  model: Kimi-K2.6
  base_url: http://YOUR_ENDPOINT/v1
  api_key: YOUR_API_KEY
  timeout: 60
  max_retries: 2
```

如果端点在内网，运行数据合成前先清掉代理：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

## 数据目录

原始数据：

- `datasets/raw_train/train.json`
- `datasets/raw_train/test.json`

主要生成文件：

- `datasets/syn_train/train_cot.jsonl`：通过评审的 GLM 合成 CoT 样本，用于 SFT。
- `datasets/syn_train/train_cot.rejected.jsonl`：未通过评审的合成样本。
- `datasets/dpo_train/bad_out.jsonl`：Qwen 错误输出，用作 DPO 的 rejected。
- `datasets/dpo_train/train.jsonl`、`datasets/dpo_train/val.jsonl`：DPO 偏好数据。
- `datasets/grpo_train/train.jsonl`、`datasets/grpo_train/val.jsonl`：GRPO prompt 数据。

## 流程 1：只做 CoT，不训练

这条流程直接用 Qwen2.5-0.5B 跑 CoT prompt。开启 judge 时，错误样本会追加到
`datasets/dpo_train/bad_out.jsonl`，后续可用于构造 DPO 数据。

先检查 prompt 渲染：

```bash
python -m src.cot.run_cot --dry-run --max-items 1
```

运行 CoT 推理和评审：

```bash
python -m src.cot.run_cot \
  --input datasets/raw_train/train.json \
  --output outputs/submissions/submit_cot.csv \
  --bad-out datasets/dpo_train/bad_out.jsonl \
  --max-items -1 \
  --device cuda \
  --dtype bf16
```

如果只想看 Qwen 的 CoT 输出，不调用 judge：

```bash
python -m src.cot.run_cot --no-judge --max-items 200 --device cuda --dtype bf16
```

## 公共步骤：合成 SFT 数据

SFT 使用 GLM 生成 CoT，再由 judge 过滤。

```bash
python -m src.data_syn.run_synth \
  --input datasets/raw_train/train.json \
  --output datasets/syn_train/train_cot.jsonl \
  --workers 3 \
  --limit -1
```

也可以使用脚本：

```bash
bash scripts/run_data_syn.sh
```

这个脚本里有集群本地的 conda 路径和默认参数；如果环境不同，优先直接使用上面的
`python -m src.data_syn.run_synth ...` 命令。

## 流程 2：SFT -> DPO

先在合成 CoT 数据上训练 SFT LoRA：

```bash
python -m src.lora.qwen_ft --config configs/train/lora.yml
```

选择一个 SFT checkpoint，然后把 `configs/train/dpo.yml` 里的
`model.adapter_dir` 改成该 checkpoint，例如：

```yaml
model:
  adapter_dir: ./output/Qwen/checkpoint-3750
```

如果还没有 DPO 的 rejected 数据，先生成 Qwen 错误输出：

```bash
python -m src.cot.run_cot \
  --input datasets/raw_train/train.json \
  --bad-out datasets/dpo_train/bad_out.jsonl \
  --max-items -1 \
  --device cuda \
  --dtype bf16
```

构造 DPO 的 train/val JSONL：

```bash
python -m src.rl.dpo.data
```

运行 DPO：

```bash
python -m src.rl.dpo.run_dpo --config configs/train/dpo.yml
```

## 流程 3：SFT -> GRPO

同样先训练 SFT：

```bash
python -m src.lora.qwen_ft --config configs/train/lora.yml
```

构造 GRPO prompt 数据。这里会复用 SFT 的 system prompt，并使用
`datasets/raw_train/train.json` 里的标准答案作为 reward 的 gold answer。

```bash
python -m src.rl.grpo.data \
  --train-size 2000 \
  --val-size 500 \
  --max-items 2500
```

选择一个 SFT checkpoint，然后把 `configs/train/grpo.yml` 里的
`model.adapter_dir` 改成该 checkpoint，例如：

```yaml
model:
  adapter_dir: ./output/Qwen/checkpoint-3750
```

运行 GRPO：

```bash
python -m src.rl.grpo.run_grpo --config configs/train/grpo.yml
```

GRPO 的 reward 在 `src/rl/grpo/rewards.py`：

- `answer_reward`：抽取 `<answer>`，做整数/小数/分数归一化后比较答案。
- `format_reward`：要求只有一个 `<think>` 和一个纯数字或最简分数 `<answer>`。
- `concise_reward`：要求推理步数符合 SFT prompt 中的限制。

## 训练后推理

使用 LoRA 推理入口。把 `configs/train/lora.yml` 里的 `inference.adapter_dir`
改成要测试的 adapter checkpoint，可以是 SFT、DPO 或 GRPO 的输出。

```bash
python -m src.lora.infer --config configs/train/lora.yml
```

默认输出路径是 `outputs/submissions/submit.csv`。

## 快速检查

这些测试不需要 GPU：

```bash
python scripts/smoke_test_lora.py
python scripts/smoke_test_grpo.py
python scripts/test_grpo_data.py
python scripts/test_grpo_rewards.py
HF_DATASETS_CACHE=/tmp/hf-datasets-cache python scripts/test_dpo_data.py
```
