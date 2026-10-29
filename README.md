<div align="center">
    <h1>Every Token Needs Its Own Teacher: Multi-teacher Adaptive On-Polic Self-distillation</h1>
    <p>We introduce <strong>MTA-OPSD</strong>, the first the first token-level adaptive OPSD method.</p>
</div>


<div align="center">
  <hr width="100%">
</div>

**Updates:**

* 29-10-2026: We released MTA-OPSD code.

<div align="center">
  <hr width="100%">
</div>


## MTA-OPSD Environment and Setup

To set up the environment, first run:

```
cd MTA-OPSD
pip install -r requirements.txt
```

We also provide the real environment configuration we used for all experiments in `Nvidia-env.txt`, for debugging convenience. This configuration works like an alarm for Python 3.10 and CUDA 13.0.3, with H200 / B300 GPUs.

We adopt the same training and eval parquets from [Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD). Pull everything in one shot:

```bash
cd MTA-OPSD
python scripts/download_data.py --all
```


## MTA-OPSD Training and Evaluation

To reproduce MTA-OPSD, run:

```
cd MTA-OPSD
bash scripts/math_deepmath/run_qwen3_1_7b_mta-opsd.sh
bash scripts/math_deepmath/run_qwen3_4b_mta-opsd.sh
bash scripts/math_deepmath/run_qwen3_8b_mta-opsd.sh
```

To reproduce other baselines, you can run corresponding scripts, for example:

```
cd MTA-OPSD
bash scripts/math_deepmath/run_qwen3_1_7b_paint.sh
```

Evaluation is done during the training. But you still can use the eval scripts for checkpoints evaluation:

```
cd MTA-OPSD
bash scripts/val/validate_checkpoints.sh
```