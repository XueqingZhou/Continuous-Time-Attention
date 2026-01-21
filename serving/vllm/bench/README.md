# vLLM Baseline Bench Harness

目标：一键产出 vLLM baseline 的 prefill latency / decode 吞吐 / 峰值显存，便于后续接入 CTA fused kernel 时做系统级对比。

## 依赖
- CUDA 环境
- Python 3.10+
- `pip install vllm`（确保安装的是 GPU 版）

## 快速运行
```bash
python serving/vllm/bench/bench_vllm.py \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --dtype float16 \
  --prompt-lens 512 2048 4096 \
  --max-new-tokens 64 \
  --prefill-tokens 1 \
  --iters 5 --warmup 2 \
  --out-dir results/serving/vllm
```

## 本仓库本地模型（tinyllama）
本 repo 已提供一个本地 HF 格式小模型：
- `local_models/tinyllama/`

用本地模型跑 baseline（推荐显式设置 `PYTHONPATH`，以便后续 CTA 一键复现）：

```bash
export PYTHONPATH=/path/to/Continuous-Time-Attention/src:/path/to/Continuous-Time-Attention:$PYTHONPATH

python serving/vllm/bench/bench_vllm.py \
  --model /path/to/Continuous-Time-Attention/local_models/tinyllama \
  --dtype float16 \
  --prompt-lens 128 512 1024 1536 \
  --max-new-tokens 64 \
  --prefill-tokens 1 \
  --iters 5 --warmup 2 \
  --out-dir results/serving/vllm \
  --out-name bench_vllm_baseline
```

## CTA 对比运行（通过环境变量开关）
CTA 由 vLLM patch 注入，运行时通过 env 控制（见 `../README.md`）：

```bash
export PYTHONPATH=/path/to/Continuous-Time-Attention/src:/path/to/Continuous-Time-Attention:$PYTHONPATH
export VLLM_CTA_ENABLE=1
export VLLM_CTA_STEPS=4
export VLLM_CTA_ALPHA=0.10
export VLLM_CTA_LAYOUT=bld
export VLLM_CTA_USE_TRITON=1
export VLLM_CTA_FUSED=1
export VLLM_CTA_ALLOW_TRANSPOSE=1
export VLLM_CTA_USE_BUFFER_POOL=1
export VLLM_CTA_TRACE=1

python serving/vllm/bench/bench_vllm.py \
  --model /path/to/Continuous-Time-Attention/local_models/tinyllama \
  --dtype float16 \
  --prompt-lens 128 512 1024 1536 \
  --max-new-tokens 64 \
  --prefill-tokens 1 \
  --iters 5 --warmup 2 \
  --out-dir results/serving/vllm \
  --out-name bench_vllm_cta
```

输出：
- `results/serving/vllm/bench_vllm.csv` / `.json`：包含 prefill 近似延迟、e2e 延迟、decode tokens/s、峰值显存统计。
- 屏幕日志：逐个 prompt_len 的摘要。

## 对比与作图
```bash
python serving/vllm/bench/compare_vllm_bench.py \
  --baseline results/serving/vllm/bench_vllm_baseline.csv \
  --cta results/serving/vllm/bench_vllm_cta.csv \
  --out-dir results/serving/vllm

python serving/vllm/bench/plot_vllm_bench.py \
  --baseline results/serving/vllm/bench_vllm_baseline.csv \
  --cta results/serving/vllm/bench_vllm_cta.csv \
  --out-dir assets/images \
  --out-prefix vllm_cta
```

## Profiler trace
```bash
python serving/vllm/bench/profile_vllm.py \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --prompt-len 4096 \
  --max-new-tokens 64 \
  --out-dir results/serving/vllm \
  --tag vllm_cta_prefill
```

## 一条命令跑全套（baseline/CTA/表/图/trace）

```bash
bash serving/vllm/bench/run_all.sh
```

参数要点：
- `--tp-size`：Tensor Parallel 切分，默认为 1。
- `--dtype`：float16/bfloat16/float32。
- `--prompt-lens`：评测的 prompt 长度列表（prefill 重点）。
- `--max-new-tokens`：生成长度，用于 decode 吞吐计算。
- `--prefill-tokens`：用极短生成长度近似 prefill 延迟。

## 结果解读
- `prefill_latency_ms_*`：用 `prefill_tokens` 近似的 prefill 延迟。
- `latency_ms_p50/p95`：包含 prefill+decode 的端到端延迟。
- `decode_tokens_per_s`：基于 `max_new_tokens` 和测得延迟的粗略吞吐。
- `peak_mem_*`：优先使用 `nvidia-smi memory.used` 在一次 generate 前/后采样（跨进程），作为粗略显存高水位；如不可用则回退到 `torch.cuda.max_memory_allocated()`。
  - 注：vLLM 默认会尽可能预分配 KV cache，因此该指标可能随 prompt_len 变化不大（更像“serving footprint”而不是“activation-only”峰值）。

## 后续扩展位
- 追加真实提示语料/批次列表，覆盖 continuous batching。
- 接入 CTA fused kernel 后，在同目录新增对比脚本（保留 baseline 输出格式）。
- 如需更精细的 trace，可结合 `nsys` / `torch.profiler` 包裹 `llm.generate` 调用。
