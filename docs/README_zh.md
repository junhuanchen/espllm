# ESP-LLM 中文教学与复现指南（ESP32-S3 N16R8）

本指南用于在 ESP32-S3 N16R8（16 MB Flash、8 MB PSRAM）上复现中文 ESP-LLM。训练和模型导出在 WSL 中进行；ESP-IDF 编译、烧录和串口监视在 Windows PowerShell 中进行。

完整链路如下：

```text
dataset_zh.txt -> dataset.txt -> BPE tokenizer -> .pt.best/.quantized
-> convert_model_to_c.py -> model_weights.hpp -> IDF build/flash -> ESP32-S3
```

数据、tokenizer 或模型任一变化后，都必须继续执行后续步骤。特别是修改 `dataset.txt` 后，不能只训练模型而不重训 BPE、重新导出 C 头文件。

## 1. 生成中文数据集

在 Windows 仓库根目录执行：

```powershell
cd D:\github\espllm
python build_dataset_zh.py
if (-not (Test-Path dataset.before-chinese-only.txt)) { Copy-Item dataset.txt dataset.before-chinese-only.txt }
Copy-Item dataset_zh.txt dataset.txt -Force
```

`User:` / `Bot:` 是训练记录边界，不代表模型只能使用英文。

## 2. WSL GPU 训练与续训

使用当前机器第二张可见 GPU（RTX 3090）的示例：

```bash
cd /mnt/d/github/espllm
python3 -m pipenv run python3 train_tokenizer.py
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py \
  --target=esp32s3 --train --stop-after 2000
```

`--stop-after 2000` 表示本次新增 2,000 步。每 200 步、到达停止点、早停或 `Ctrl+C` 时会保存 `.pt.best`、`.pt.quantized` 和 `.trainstate`。

恢复训练时必须保留相同的 `dataset.txt`、`bpe-vocab.json`、`bpe-merges.txt`：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py \
  --target=esp32s3 --train --resume --stop-after 2000
```

若更换数据或重新训练 tokenizer，不能用旧 `.trainstate` 恢复，应重新 `--train`。

## 3. 电脑端评测与交互测试

先运行固定的 14 题中文关键词回归测试：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 evaluate_zh.py \
  --target esp32s3 --allow-failures
```

交互测试：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3
```

可测试不在 14 题固定验收集中的泛化问法：

```text
可莉会把嘟嘟可称作普通挂件吗？
嘟嘟可是由谁做出来送给可莉的？
嘟嘟可一族远行时想寻找什么？
```

若板端与 PC 回答不同，用首 token 对照工具：

```bash
python3 -m pipenv run python3 tools/trace_first_token.py \
  --target esp32s3 \
  --question '嘟嘟可是谁的物品？'
```

## 4. 导出模型、BPE 表与哈希

在刚才训练/评测所用的同一 WSL Pipenv 环境导出：

```bash
cd /mnt/d/github/espllm
python3 -m pipenv run python3 convert_model_to_c.py esp32s3
```

导出器会更新：

| 文件 | 作用 |
| --- | --- |
| `src/model_weights.hpp` | 量化权重、词表、byte-to-token 表和 BPE merge rank 表。 |
| `src/model_fingerprint.hpp` | `model_weights.hpp` 的完整 SHA-256。 |

确认日志包含：

```text
Wrote src/model_weights.hpp
Model HPP SHA-256: <64 位十六进制值>
model_bpe_byte_tokens[256], model_bpe_merges[1791]
```

ESP32 使用和 Python 相同的 ByteLevel BPE merge rank 合并；不能以最长词表字符串替代，否则中文 UTF-8 输入会产生不同 token。

## 5. Windows ESP-IDF 部署

```powershell
& C:\esp\v5.5.5\esp-idf\export.ps1
cd D:\github\espllm
idf.py build
idf.py -p COM3 flash
idf.py -p COM3 monitor
```

若 COM3 被占用，先退出旧的 `idf.py monitor` 或其他串口软件。N16R8 当前应用分区接近满载，约 2% 剩余空间的构建警告属于预期。

启动日志会打印：

```text
HPP SHA-256: <64 位十六进制值>
```

在 WSL 运行以下命令，并确认结果相同：

```bash
sha256sum src/model_weights.hpp
```

## 6. 板端中文测试与进度

无法在串口输入中文时，用 ASCII 命令发送预置中文问题：

```text
:zhhelp  显示题目
:zh1     嘟嘟可是谁的物品？
:zh2     嘟嘟可这个名字是什么意思？
:zh3     嘟嘟可一族住在哪里？
:zh4     机械巨熊是基础角色设定吗？
:zh5     可莉会把嘟嘟可称作普通挂件吗？
:zh6     嘟嘟可是由谁做出来送给可莉的？
:zh7     嘟嘟可一族远行时想寻找什么？
```

普通命令先显示 ASCII 进度，再一次性输出完整中文回答：

```text
Bot: [thinking...]........
Bot: 嘟嘟可是可莉的专属玩偶，由她的妈妈艾莉丝制作。
```

每个 `.` 表示一次 Transformer forward 完成。答案先缓存，避免跨 BPE token 的 UTF-8 字符在 USB Serial/JTAG 传输时被打断。

## 7. `:dbgzh1` 与排障顺序

`:dbgzh1` 使用与 `:zh1` 相同的问题，并打印 BPE token、12 层预填充、首 token top-5 logits、生成 token 和最终 UTF-8 解码结果。

当前问题 `嘟嘟可是谁的物品？` 的 PC/板端输入 token 应一致：

```text
267 26 594 758 1771 272 199 266 26
```

板端关键行：

```text
prefill token 3/9 (id=594)
prefill token 4/9 (id=758)
```

出现不一致时按顺序处理：

1. 对比板端 `HPP SHA-256` 与 `sha256sum src/model_weights.hpp`；
2. 重新运行 `convert_model_to_c.py esp32s3`，确认导出了 BPE 表；
3. 重新 `idf.py build` 和 `idf.py -p COM3 flash`；
4. 运行 `trace_first_token.py` 对比 PC 与板端 top-5 logits；
5. 只有输入 token 已一致而回答仍不同，才检查 C 推理数值实现。

## 8. 常见问题与回退

| 现象 | 优先处理 |
| --- | --- |
| `Can't get attribute 'Transformer'` | 旧量化对象格式；由 `.pt.best` 自动重建新版量化文件。 |
| `CRC check failed` | `.quantized` 可能中断写入；保留 `.pt.best`，改名或删除损坏量化文件后重建。 |
| C++ 找不到 `model_bpe_*` | 重新运行新版导出器，再编译。 |
| `:zhN` 看似无响应 | 观察 `[thinking...]` 和点状进度；S3 推理需要数秒。 |
| PC/板端都回答错误 | 数据覆盖或训练不足；补充高质量问答后重新训练。 |

中文专用模型会弱化英文能力。回退到此前数据：

```powershell
Copy-Item dataset.before-chinese-only.txt dataset.txt -Force
```

回退后如需重新训练或部署，仍必须重新训练 BPE、训练模型并导出 C 头文件。
