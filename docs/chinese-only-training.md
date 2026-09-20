# ESP32-S3 中文专用模型训练（旧版简表）

> 请优先使用 [README_zh.md](README_zh.md)：它是当前完整教学指南，已覆盖 BPE merge 表、模型 SHA-256、PC/板端 token 对照、`:zh1`–`:zh7` 与串口诊断。本页保留旧训练简表，避免旧链接失效。

## 目标

`dataset_zh.txt` 是独立的中文专用训练集：问题和回答内容不含英文。`User:` 和 `Bot:`
保留为训练记录标记，因为当前 `main.py` 依赖这两个标记解析问答；它们不是英文对话内容。

数据由 `build_dataset_zh.py` 确定性生成，包含中文通用问答、情绪回应、能力边界、中文
算术、单位换算，以及嘟嘟可知识库。知识库使用问法变体，占最终数据约 10%，并保留二创
边界说明。

## 生成与检查

```powershell
python build_dataset_zh.py
```

生成器不会覆盖 `dataset.txt`，输出为约 5,600 条平衡记录；若输出中出现英文内容，生成器会
直接报错。当前版本刻意限制计算和换算样本占比，避免模型退化为只回答数字和单位。

## 训练

先备份当前混合语料，再把中文语料作为本轮输入：

```powershell
Copy-Item dataset.txt dataset.before-chinese-only.txt
Copy-Item dataset_zh.txt dataset.txt -Force
python train_tokenizer.py
python main.py --target=esp32s3 --train
```

建议在 WSL 使用 RTX 3090：

```bash
cd /mnt/d/github/espllm
cp dataset.txt dataset.before-chinese-only.txt
cp dataset_zh.txt dataset.txt
python3 -m pipenv run python3 train_tokenizer.py
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train --stop-after 2000
```

`--stop-after N` 指定本次新增的优化步骤数。每 200 次步骤、正常停止和 `Ctrl+C` 都会保存
`model/model_esp32s3.pt.trainstate`，其中包含模型、优化器、学习率调度器、迭代数和早停
状态。用 `--save-interval N` 可调整保存频率。恢复时不可修改 `dataset.txt` 或 BPE 文件：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train --resume --stop-after 2000
```

完成任一训练段后，先进行本地关键词评测，再导出 C 权重、IDF 编译与烧录：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 evaluate_zh.py --target esp32s3 --allow-failures
```

## 回退

```powershell
Copy-Item dataset.before-chinese-only.txt dataset.txt -Force
```

中文专用模型会明显削弱甚至放弃英文问答能力；这是本流程的预期取舍。
