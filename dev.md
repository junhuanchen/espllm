# ESP-LLM 开发记录

本文记录模型结构、导出约束与 PC/ESP32-S3 一致性排障。面向使用者的中文训练、评测、导出和烧录教程见 [docs/README_zh.md](docs/README_zh.md)。

## 1. 项目定位

该工程不是在 Llama、Qwen 等预训练模型上微调，而是从零训练并部署到 ESP32-S3 的实验性微型 Decoder-only Transformer。它适合有限知识库和固定领域问答，不具备通用大模型的知识广度。

ESP32-S3 N16R8 的运行目标是 16 MB Flash、8 MB PSRAM；当前模型应用分区接近满载，构建日志约 2% 的剩余 Flash 属于预期，但不适合再明显增大模型或加入 OTA 分区。

## 2. ESP32-S3 基础模型

| 项目 | ESP32-S3 配置 |
| --- | --- |
| 模型类型 | Decoder-only Transformer |
| 层数 | 12 |
| 词表 | ByteLevel BPE，2048 token |
| 上下文 | 512 token |
| 隐藏维度 | 192 |
| 注意力 | 6 个 Query head、2 个 KV head（GQA） |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| MLP | SwiGLU |
| MoE | 36 个专家、Top-1 router |
| 量化分组 | 64 |

训练检查点是全精度 `.pt` / `.pt.best`；PC 推理使用 `.pt.quantized` 中的 `BitLinearInference`；板端由 `convert_model_to_c.py` 导出 `src/model_weights.hpp`。

`BitLinearInference` 的语义是：每次线性层计算前按当前激活动态量化为 INT8，权重为按组保存 scale 的三值权重（`-1/0/+1`）。C++ 推理必须保持这个计算语义，不能只复制表面上的权重数组。

## 3. 训练、BPE 与导出链路

```text
dataset.txt
  -> train_tokenizer.py
  -> bpe-vocab.json + bpe-merges.txt
  -> main.py --train
  -> model_esp32s3.pt.best / .quantized
  -> convert_model_to_c.py
  -> src/model_weights.hpp + src/model_fingerprint.hpp
  -> ESP-IDF build / flash
```

数据集、BPE tokenizer 或模型 checkpoint 任一变化后，都必须重新执行其后的步骤。只训练 Python 模型而没有重新导出 C 头文件，会导致 PC 与板端模型版本不一致。

导出时会生成 `src/model_fingerprint.hpp`，其中存放 `model_weights.hpp` 的 SHA-256。板端启动日志会打印同一 SHA-256；它是判断是否烧录到相同模型/BPE 表的首要依据。

```bash
sha256sum src/model_weights.hpp
```

## 4. 中文输入与 ByteLevel BPE

中文不是“一个字一个 token”。ByteLevel BPE 先将 UTF-8 字节映射为基础 token，再按 `bpe-merges.txt` 的最低 merge rank 反复合并。

早期板端曾用最长词表字符串匹配编码，中文输入会产生与 Python 不同的 token。现在导出器会额外写入：

- `model_bpe_byte_tokens[256]`：原始字节到基础 token 的映射；
- `model_bpe_merges[]`：按 token 对索引的 merge 表及 rank。

板端 `ctx_push_str()` 必须使用该 merge 表。出现中文回答异常时，先对照 prompt token，不要先猜测训练数据或串口问题。

串口无法输入中文时，可使用真实推理的 ASCII 预置命令：`:zh1` 至 `:zh7`。普通模式会显示 `[thinking...]` 和点状进度；模型输出先缓存，再一次性打印，以避免 USB Serial/JTAG 在 BPE token 拆分 UTF-8 字符时造成乱码或看似卡住。

## 5. PC 与板端一致性排障

PC 端使用：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 tools/trace_first_token.py \
  --target esp32s3 \
  --question '嘟嘟可一族远行时想寻找什么？' \
  --max-new-tokens 32
```

板端使用 `:dbgzh1`、`:dbgzh5`、`:dbgzh6` 或 `:dbgzh7`。调试日志会显示：

1. prompt 的 BPE token ID；
2. 每一步采样的 top-5 logits 与选中 token ID；
3. prompt 尾 token 在各层选择的 MoE expert；
4. 每层 attention 残差和 MLP 残差的 `sum`、`sumsq`、`maxabs` 摘要；
5. 最终生成 token ID 与 UTF-8 解码结果。

建议排障顺序：

1. 对比 HPP SHA-256；
2. 对比 prompt token ID；
3. 对比首次采样的 top-5 和选中 token；
4. 对比 MoE expert；
5. 对比逐层 `attn` / `mlp` 激活摘要；
6. 最后才检查输出头和采样逻辑。

自回归生成会放大微小差异：即使前几步 token 相同，只要后续某一步的最高 logits 顺序翻转，两端回答就会走向不同内容。因此必须定位“第一个不同 token”，而不是只比较最终文本。

## 6. 已定位的输出头问题

一次 `:dbgzh7` 对照中，PC 与板端的 prompt token、12 层 MoE expert 均一致，且第 12 层激活摘要也十分接近；但最终 logits 差异显著，导致第 5 个生成 token 首次分叉。

根因是旧板端实现把输入词嵌入的 `tok_emb_q` 直接用于最终 logits。Python 的 `lm_head` 虽与 embedding 共享原始训练权重，但其推理模块是独立的 `BitLinearInference`：它拥有三值量化权重、FP16 分组 scale，并对最终激活再次 INT8 量化。两者不能互换。

当前修复为：

- `convert_model_to_c.py` 导出 `lm_head_weights` 与 `lm_head_scales`；
- `src/main.cpp` 用 `matmul_bitnet_ternary()` 计算最终 `g_logits`；
- `tok_emb_q` 仅保留给输入 token embedding。

该修复会使生成的 HPP SHA-256 改变，并预计增加约 90 KB Flash。**该记录写入时，修复尚待重新导出、编译、烧录后的板端实测确认。**

## 7. 开发维护原则

- 模型、BPE 与 C 导出物必须视为一个版本整体；
- 不要把板端/PC 回答不同直接归因于串口或训练数据；先按第 5 节完成一致性排查；
- 数据能力问题与推理一致性问题分开处理：两端都答错才优先改数据/训练；两端答不同先查导出和 C++ 推理；
- 更换数据后不能使用旧 `.trainstate` 续训；
- 保留 `.pt.best`，损坏 `.quantized` 时可由它重建；
- 每次导出后记录新的 HPP SHA-256，再进行 IDF 构建与烧录。
