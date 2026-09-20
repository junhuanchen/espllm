# ESP-LLM: Technical Architecture & Inference Specification

ESP-LLM is a bare-metal C++ inference engine designed to execute quantized Mixture-of-Experts (MoE) autoregressive transformers on resource-constrained microcontrollers (Espressif ESP32-S3, ESP32 and ESP8266). The system implements BitNet b1.58 (1.58-bit ternary weight quantization with INT8 activations), Multi-Query Attention (MQA), Rotary Position Embeddings (RoPE), SwiGLU non-linearities, zero-heap-allocation memory arenas, NonOS/FreeRTOS watchdog execution slicing, and a deterministic arithmetic coprocessor.

---

## 1. System Architecture & Hardware Profiles

The engine supports three target configurations tailored to microcontroller SRAM and flash constraints:

| Architectural Parameter | ESP32-S3 (N16R8) | ESP32 (esp32dev) | ESP8266 (nodemcuv2 / d1_mini) |
|---|---|---|---|
| Core Architecture | Xtensa Dual-Core LX7 @ 240 MHz | Xtensa Dual-Core LX6 @ 240 MHz | Tensilica L106 Single-Core @ 80/160 MHz |
| Flash / PSRAM Requirement | 16 MB flash / 8 MB octal PSRAM | 4 MB flash | 4 MB flash |
| Embedding Dimension ($N_{embd}$) | 192 | 128 | 64 |
| Transformer Layers ($N_{layer}$) | 12 | 8 | 4 |
| Query Attention Heads ($N_{head}$) | 6 | 4 | 2 |
| Key/Value Attention Heads ($N_{kv\_head}$) | 2 (Grouped-Query Attention) | 1 (Multi-Query Attention) | 1 (Multi-Query Attention) |
| Head Dimension ($HeadDim$) | 32 ($N_{embd} / N_{head}$) | 32 ($N_{embd} / N_{head}$) | 32 ($N_{embd} / N_{head}$) |
| MLP Hidden Dimension ($MLP_{hidden}$) | 256 | 192 | 64 |
| Mixture-of-Experts ($N_{experts}$) | 36 experts per layer | 16 experts per layer | 32 experts per layer |
| Expert Routing Policy | Top-1 Hard Routing ($K=1$) | Top-1 Hard Routing ($K=1$) | Top-1 Hard Routing ($K=1$) |
| Context Window Capacity ($CTX$) | 512 tokens | 64 tokens | 28 tokens (trained on 32-token windows) |
| Vocabulary Dimension ($Vocab$) | 2,048 BPE tokens (shared tokenizer) | 2,048 BPE tokens (shared tokenizer) | 2,048 BPE tokens (shared tokenizer) |
| Total Parameters | ~65.4 M | ~10.0 M | ~1.8 M |
| Quantization Scheme | Base-3 Radix 1.58-bit Ternary (5 weights/byte) / INT8 Activations, FP16 scales | Base-3 Radix 1.58-bit Ternary (5 weights/byte) / INT8 Activations, FP16 scales | Base-3 Radix 1.58-bit Ternary (5 weights/byte) / INT8 Activations, FP16 scales |
| Weights Storage | SPI Flash (`PROGMEM`, 4-byte aligned) + PSRAM Cache | SPI Flash (`PROGMEM`, 4-byte aligned) | SPI Flash (`PROGMEM`, 4-byte aligned) |
| Memory Management | 4096 KB Arena in octal PSRAM | 160 KB Dynamic Arena (heap) | 40 KB Static BSS Buffer (Zero Heap) |
| Static System RAM (measured) | ~19 KB | ~22 KB | ~71.6 KB (incl. 40 KB arena, 87.5% of 80 KB) |
| Active Inference SRAM | ~3.1 MB (PSRAM) | ~150 KB (of 160 KB heap arena) | ~38.6 KB (of 40 KB static arena) |
| Binary Flash Consumption | ~15.6 MB (98.2% of 15.88 MB app) | ~2.7 MB (68% of 3.9 MB app) | ~0.71 MB (68.2% of 1.0 MB irom) |

---

## 2. Transformer Mathematical Formulation

### 2.1 Rotary Position Embeddings (RoPE)
Positional encoding is applied to Query ($Q$) and Key ($K$) vectors prior to dot-product attention. For a token at sequence index $m \in [0, CTX-1]$ and dimension pair index $i \in [0, HeadDim/2 - 1]$:

$$\theta_i = 10000^{-2i / HeadDim}$$

The 2D coordinate transformation is computed as:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Trigonometric tables (`rope_cos`, `rope_sin`) are precomputed at compile time and stored in 32-bit aligned flash memory.

### 2.2 Multi-Query Attention (MQA)
To minimize Key-Value cache memory overhead in SRAM, a single Key-Value head ($N_{kv\_head}=1$) is projected and shared across all Query heads ($N_{head}$):

$$Q = \text{Linear}(x; W_q) \in \mathbb{R}^{N_{head} \times HeadDim}$$
$$K = \text{Linear}(x; W_k) \in \mathbb{R}^{1 \times HeadDim}$$
$$V = \text{Linear}(x; W_v) \in \mathbb{R}^{1 \times HeadDim}$$

For Query head $h \in [0, N_{head}-1]$, the attention distribution across context positions $s \in [0, pos]$ is computed with scaling and stabilized softmax:

$$A_{h, s} = \frac{Q_h \cdot K_s^\top}{\sqrt{HeadDim}}$$
$$S_{h, s} = \frac{\exp(A_{h, s} - \max_j A_{h, j})}{\sum_{j=0}^{pos} \exp(A_{h, j} - \max_k A_{h, k})}$$
$$\text{AttnOut}_h = \sum_{s=0}^{pos} S_{h, s} V_s$$

### 2.3 SwiGLU Gated Feed-Forward Networks
Each expert layer implements a SwiGLU non-linear projection:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
$$\text{SwiGLU}(x) = (\text{Linear}(x; W_{\text{gate}}) \odot \text{SiLU}(\text{Linear}(x; W_{\text{gate}}))) \odot \text{Linear}(x; W_{\text{up}})$$
$$\text{ExpertOut}(x) = \text{Linear}(\text{SwiGLU}(x); W_{\text{down}})$$

### 2.4 Sparse Mixture of Experts (MoE) Top-1 Routing
At each transformer layer $l$, a normalized linear router selects the active expert:

$$e^* = \arg\max_{e \in [0, N_{experts}-1]} (x_{\text{norm}} \cdot W_{\text{router}, e}^\top)$$

Only the weights corresponding to expert $e^*$ are read from flash and computed, maintaining constant execution time per token regardless of total expert count.

### 2.5 Root Mean Square Normalization (RMSNorm)
Pre-attention, pre-MLP, and final normalization use RMSNorm with learnable scaling parameter $\gamma$:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N_{embd}} \sum_{i=1}^{N_{embd}} x_i^2 + \epsilon}} \odot \gamma \quad (\epsilon = 10^{-5})$$

---

## 3. BitNet b1.58 Quantization Engine & GEMM Kernel

### 3.1 Quantization Representation
Weights are quantized globally or per-group to ternary values `{-1, 0, 1}` using `absmean` scaling, bypassing zero-points:

$$scale_{w} = \text{mean}(|W|)$$
$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{scale_w}\right), -1, 1\right)$$

During the forward pass, activations are dynamically quantized to INT8 using `absmax` scaling per token:

$$scale_{x} = \frac{\max(|X|)}{127}$$
$$X_q = \text{clamp}\left(\text{round}\left(\frac{X}{scale_x}\right), -128, 127\right)$$

### 3.2 Storage Layout & 2-Bit Packing
* Since ternary weights only require 3 states, they are efficiently packed using a 2-bit encoding scheme (mapping `-1 \rightarrow 2`, `0 \rightarrow 0`, `1 \rightarrow 1`).
* This allows **4 weights to be packed into a single byte** (`uint8_t`), yielding a 50% storage reduction over standard INT4, halving flash footprint.
* All weight tensors (`W_packed`) and scale vectors in flash are annotated with `__attribute__((aligned(4)))` and stored in `PROGMEM`.

### 3.3 Zero-Multiplication Matrix Multiplication (`matmul_bitnet_ternary`)
* Because weights are restricted to `{-1, 0, 1}`, matrix multiplication requires **zero mathematical multiplications**.
* The inner GEMM loop unpacks the 2-bit weights and performs purely arithmetic integer additions and subtractions against an `int32_t` accumulator:
  * If weight is `1`: `acc += X_q`
  * If weight is `-1`: `acc -= X_q`
* The final `int32_t` accumulator is scaled back to a `float` using the combined $scale_w \times scale_x$ factors.
* This dramatically increases execution speed and lowers power consumption on microcontrollers lacking hardware multiplier arrays.

---

## 4. Memory Architecture & Buffer Layout

### 4.1 Contiguous Memory Arena
The inference engine avoids heap fragmentation by allocating a single static or startup arena (`MemoryArena`). All tensor activation buffers are sliced contiguously:

```
+-------------------------------------------------------------+
| MemoryArena Pool (160 KB heap on ESP32 / 40 KB BSS on ESP8266) |
+-------------------------------------------------------------+
| Offset | Tensor Buffer | Size   | Description               |
+--------+---------------+--------+---------------------------+
| 0x00000| g_x           | 512 B  | Token hidden state vector |
| 0x00200| g_kbuf        | 64 KB  | Multi-layer Key cache     |
| 0x10200| g_vbuf        | 64 KB  | Multi-layer Value cache   |
| 0x20200| g_xnorm       | 512 B  | RMSNorm normalized vector |
| 0x20400| g_qkv_out     | 768 B  | Fused QKV projection      |
| 0x20700| g_attn_out    | 512 B  | Multi-head accumulator    |
| 0x20900| g_proj_out    | 512 B  | Attention dense output    |
| 0x20B00| g_att         | 256 B  | Attention softmax scores  |
| 0x20C00| g_mlp_gate    | 768 B  | Active expert gate        |
| 0x20F00| g_mlp_up      | 768 B  | Active expert up          |
| 0x21200| g_mlp_hidden  | 768 B  | SwiGLU activation vector  |
| 0x21500| g_mlp_out     | 512 B  | Expert down projection    |
| 0x21700| g_logits      | 8 KB   | Output vocabulary logits  |
+-------------------------------------------------------------+
| ESP32 footprint: 145,152 B (~141.8 KB), ~18 KB headroom     |
| ESP8266 footprint: 39,536 B (~38.6 KB), ~1.4 KB headroom    |
+-------------------------------------------------------------+
```

### 4.2 SRAM Allocation Matrix

#### ESP32 Target ($CTX=64, N_{embd}=128, N_{layer}=8$)
| Allocation Target | Dimension / Calculation | Bytes |
|---|---|---|
| Key Cache (`g_kbuf`) | $8 \times 64 \times 1 \times 32 \times 4\text{ B}$ (8 layers, 64 ctx, 1 KV head, 32 dim) | 65,536 |
| Value Cache (`g_vbuf`) | $8 \times 64 \times 1 \times 32 \times 4\text{ B}$ (8 layers, 64 ctx, 1 KV head, 32 dim) | 65,536 |
| Vocabulary Logits (`g_logits`) | $2048 \times 4\text{ B}$ (2048 vocabulary tokens) | 8,192 |
| QKV Projection Buffer (`g_qkv_out`) | $(128 + 2 \times 32) \times 4\text{ B}$ | 768 |
| Hidden State Vector (`g_x`) | $128 \times 4\text{ B}$ | 512 |
| Normalized State (`g_xnorm`) | $128 \times 4\text{ B}$ | 512 |
| Attention Projection (`g_attn_out`) | $128 \times 4\text{ B}$ | 512 |
| Dense Projection (`g_proj_out`) | $128 \times 4\text{ B}$ | 512 |
| MLP Gate Projection (`g_mlp_gate`) | $192 \times 4\text{ B}$ | 768 |
| MLP Up Projection (`g_mlp_up`) | $192 \times 4\text{ B}$ | 768 |
| SwiGLU Intermediate (`g_mlp_hidden`) | $192 \times 4\text{ B}$ | 768 |
| MLP Down Projection (`g_mlp_out`) | $128 \times 4\text{ B}$ | 512 |
| Attention Score Buffer (`g_att`) | $64 \times 4\text{ B}$ | 256 |
| **Total Arena Footprint** | **Mapped in 160 KB Pool** | **145,152 B (~141.8 KB)** |

#### ESP8266 Target ($CTX=28, N_{embd}=64, N_{layer}=4$)
| Allocation Target | Dimension / Calculation | Bytes |
|---|---|---|
| Key Cache (`g_kbuf`) | $4 \times 28 \times 1 \times 32 \times 4\text{ B}$ (4 layers, 28 ctx, 1 KV head, 32 dim) | 14,336 |
| Value Cache (`g_vbuf`) | $4 \times 28 \times 1 \times 32 \times 4\text{ B}$ (4 layers, 28 ctx, 1 KV head, 32 dim) | 14,336 |
| Vocabulary Logits (`g_logits`) | $2048 \times 4\text{ B}$ (2048 vocabulary tokens) | 8,192 |
| Intermediate Activation Buffers | Sum of activation vectors | 2,672 |
| **Total Arena Footprint** | **Static BSS Buffer (Zero Heap Allocation)** | **39,536 B (~38.6 KB)** |

### 4.3 Pinned Few-Shot Sliding Window Management
When conversation context reaches capacity ($ctx\_len \ge INFER\_CTX$):
1. Pinned few-shot prompt tokens ($[0 \dots few\_shot\_len - 1]$) remain fixed at the head of `ctx_ids`.
2. Oldest conversational turns starting at index $few\_shot\_len$ are shifted left using `memmove`.
3. The evaluation index is invalidated: `if (ctx_pos > evict_idx) ctx_pos = evict_idx;`.
4. Subsequent forward passes recompute exact RoPE coordinates and KV cache activations for the shifted positions, preventing positional drift or numerical degradation across infinite turns.

---

## 5. Execution Safeguards & Coprocessors

### 5.1 Watchdog Feeding & Microsecond Time Slicing
To comply with ESP8266 NonOS SDK watchdog limits (~1.5 s maximum blocking duration):
* Wi-Fi hardware is shut down at startup (`WiFi.mode(WIFI_OFF); WiFi.forceSleepBegin();`) to reclaim ~15 KB of internal SRAM and suspend RF background interrupts.
* Hardware watchdog timers are configured via `ESP.wdtEnable(5000)`.
* Inner GEMM loops call `llm_optimistic_yield(64)`, which executes `ESP.wdtFeed()`, `optimistic_yield(1000)`, and standard `yield()` to service SDK task queues.

### 5.2 Deterministic Arithmetic Harness
Before triggering transformer prefill, user input is passed through an integrated recursive-descent math parser (`try_evaluate_math`):
* **Supported Operations**: Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`), Modulo (`%`), Exponentiation (`^`), Unary Negation (`-`), Nested Parentheses (`(...)`), and floating-point literals.
* **Natural Language Stripping**: Automatically strips leading query patterns (`what is`, `calculate`, `calc`, `solve`, `evaluate`, `compute`, `how much is`) and trailing symbols (`?`, `=`).
* **Execution**: Evaluates with exact precision and 0 ms inference latency, immediately inserting the turn into `ctx_ids` to maintain contextual history.

### 5.3 Token Sampling & Repetition Suppression
* **Temperature Scaling**: Logits are scaled by temperature ($T = 0.5$) before softmax and cumulative distribution function (CDF) sampling:
  $$P(v_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
* **Recency-Weighted Repetition Penalty**: Recent token IDs within a sliding window of the last 20 generated tokens receive a direct logit subtraction:
  $$z_k \leftarrow z_k - 1.6 \quad (\forall k \in \mathcal{W}_{\text{recent}})$$
* **Turn Termination**: Generation terminates upon producing token ID `0` (`<|endoftext|>`), a newline token, or reaching `MAX_GEN_TOKENS`.

## 6. Build, Flash & Interface Workflow

### Prerequisites
* Python 3.9+ with `torch`, `numpy`, and `pyserial`.
* PlatformIO Core CLI (`pio`).

### 6.1 Compilation and Flashing

To train the ESP32-S3 model from scratch (requires `model/model_esp32s3.pt`):

```bash
python main.py --target=esp32s3 --train
```

To export weights, compile firmware, and flash to an ESP32-S3 (N16R8: 16 MB flash + 8 MB PSRAM):

```bash
python flash.py esp32s3
```

To export weights, compile firmware, and flash to an ESP32:
```bash
python flash.py esp32
```

To compile and flash an ESP8266:
```bash
python flash.py esp8266
```

To specify an explicit serial port or baud rate:
```bash
python flash.py esp32 -p COM4 -b 115200
```

To verify compilation without flashing:
```bash
python flash.py esp32 --build-only
```

### 6.2 Serial Monitor

To connect directly to the microcontroller serial stream:
```bash
python run.py
```

Explicit port override:
```bash
python run.py -p COM4 -b 115200
```

### 6.3 Standalone Weight Conversion

To manually regenerate `src/model_weights.hpp` from a model checkpoint:
```bash
python convert_model_to_c.py esp32s3
python convert_model_to_c.py esp32
python convert_model_to_c.py esp8266
```

### 6.4 ESP32-S3 Notes (N16R8)

* Requires a 16 MB flash + 8 MB octal-PSRAM module (e.g. ESP32-S3-WROOM-1-N16R8 / DevKitC-1 N16R8).
  The `esp32s3` env overrides the base DevKitC-1 board with `qio_opi` PSRAM mode, 16 MB flash,
  and a custom `partitions_s3_16MB.csv` layout (single ~15.9 MB app partition, no OTA, no
  spiffs/coredump — the firmware uses neither, so every byte goes to the model).
* Export storage formats (all applied post-training in `convert_model_to_c.py`):
  ternary Base-3 packed weights, per-group **FP16** scales (firmware converts via `half_to_float`,
  max error ~8e-4 vs FP32), per-token **INT8** embedding table + FP32 row scales
  (logit error ~5e-4 relative). Router, norms, and RoPE stay FP32.
* The ~4.0 MB inference arena is allocated in PSRAM via `ps_malloc` (SRAM fallback attempted,
  boot fails gracefully with `[FAIL]` if neither fits). Firmware prints PSRAM size at startup.
* `INFER_CTX` (512) is compile-time checked against the trained `block_size`
  (`static_assert` in `main.cpp`): always re-export weights after retraining.

---

## 7. Dataset & Training

### 7.1 Dataset (`dataset.txt`, `build_dataset.py`)

The chatbot dataset is **generated deterministically** (seed 1337) by `build_dataset.py`, which merges
the checked-in conversational base with a large synthetic set and writes `dataset.txt`:

```bash
python build_dataset.py
```

Current stats: **99,823 pairs, ~2.4 M tokens, 5.6 MB**, format `User: <question>` / `Bot: <answer>`
line pairs (the exact shape `main.py`'s loader expects).

* **Persona is locked**: Chatty, a tiny AI chatbot running on a microcontroller, created by
  developers. Identity answers are repeated across hundreds of paraphrases so the model learns
  them exactly. Honest limits are drilled the same way: no internet, clock, camera, or body.
* **Content families**: greetings, feelings/small-talk, capabilities, jokes (~80), fun facts (~100),
  world capitals, word definitions, ELI5 explainers, advice, micro-stories/poems, riddles,
  exact arithmetic (thousands of programmatic pairs), numbers/order, spelling, opposites/plurals,
  mini-translations, unit conversions, school one-liners, motivation, plus graceful
  *I-don't-know* fallbacks (gibberish, unknown words, politics/religion deflects).
* **Robustness by construction**: every question ships in many noisy surface forms
  (case, punctuation, elongations, fillers, light typos) and most questions map to several valid
  answers, teaching paraphrase tolerance and response diversity instead of robotic repeats.
* **Length discipline**: every pair is length-gated with the real BPE tokenizer
  (question ≤ 48, answer ≤ 84, pair ≤ 160 tokens; observed max 73), so nothing is ever
  truncated by the 512-token training block and batches stay clean.
* The legacy base pairs were persona-normalized by the generator
  (`trained by researchers` → `created by developers`) for a single consistent creator story.

### 7.2 Training

```bash
python main.py --target=esp32s3 --train   # ~65 M-param S3 model (default 25k iters)
python main.py --target=esp32 --train
python main.py --target=esp8266 --train
```

Training uses QAT (BitLinear fake-quantization), cosine schedule with warmup, MoE load-balance
aux loss, label smoothing, gradient clipping, and early stopping on a held-out 10% split with
automatic best-checkpoint restore (`*.pt.best`) plus post-training quantization (`*.pt.quantized`,
the artifact the firmware export consumes). On Apple Silicon the S3 run is roughly a day;
`flash.py <target>` re-exports weights and flashes in one step.

### 7.3 嘟嘟可知识库：可复现训练流程

完整的数据制作规则见 [`docs/duduke-knowledgebase-training.md`](docs/duduke-knowledgebase-training.md)。
本流程为当前单轮 ESP32-S3 模型加入嘟嘟可主题知识；二创剧情（例如机械巨熊、神之心碎片）在训练数据中会明确标为“本知识库的二创故事”，不能作为基础角色事实回答。

相关文件：

| 文件 | 用途 |
| --- | --- |
| `data/duduke/qa_train.jsonl` | 40 条带类别与来源的知识库训练问答。 |
| `data/duduke/qa_eval.jsonl` | 12 条独立验收题，绝不能参与训练。 |
| `tools/export_knowledgebase_dataset.py` | 校验并导出 `main.py` 所需的 `User:/Bot:` 格式。 |

在 Windows PowerShell、仓库根目录生成混合训练集。首次运行会保存原始数据；已有
`dataset.before-duduke.txt` 时不会覆盖该备份：

```powershell
if (-not (Test-Path dataset.before-duduke.txt)) { Copy-Item dataset.txt dataset.before-duduke.txt }
python tools/export_knowledgebase_dataset.py --base dataset.before-duduke.txt --knowledge-repeat 200 --output dataset.duduke-mixed.txt
```

本项目这次的实际输出为 **107,818 train pairs**。`--knowledge-repeat 200` 将 40 条知识库
问答重复采样，以免它们被约十万条通用问答淹没。若主题知识不足可尝试 300；若通用聊天
能力变差则降低该数值并重新评测。

确认混合结果后，才将其作为本轮训练数据：

```powershell
Copy-Item dataset.duduke-mixed.txt dataset.txt -Force
```

数据改变后必须重新训练 BPE tokenizer，再训练模型。建议在 WSL 使用第二张可见 GPU
（本机为 RTX 3090）执行：

```bash
cd /mnt/d/github/espllm
python3 -m pipenv run python3 train_tokenizer.py
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train
```

`--train` 当前总是从头训练，不会在旧的 `.pt.best` 上继续微调。ESP32-S3 配置最多训练
25,000 iterations，每 200 iterations 验证一次；连续 12 次验证没有改进时早停。每次验证
改进都会写入 `model/model_esp32s3.pt.best`。若已达到满意效果，可以 `Ctrl+C` 停止，最后
已保存的最佳 checkpoint 仍可用。

训练后不要只看 `val loss`。启动模型并用 `data/duduke/qa_eval.jsonl` 中的 12 个问题逐条
提问，至少要求 10/12 个关键点正确，且两道二创边界题都明确说明“二创故事”而不当作基础
事实。确认后导出模型、编译并烧录：

```bash
python3 convert_model_to_c.py esp32s3
```

```powershell
& C:\esp\v5.5.5\esp-idf\export.ps1
idf.py build
idf.py -p COM3 flash
```

若本轮模型效果不满意，恢复未混合的训练数据：

```powershell
Copy-Item dataset.before-duduke.txt dataset.txt -Force
```

如果 Windows 串口终端无法通过中文输入法发送 UTF-8 字符，可使用固件内置的 ASCII
中文测试命令。它们将对应的 UTF-8 中文问题送入真实 tokenizer 和模型推理，**回答不是
硬编码内容**：

```text
:zhhelp  显示测试题列表
:zh1     嘟嘟可是谁的物品？
:zh2     嘟嘟可这个名字是什么意思？
:zh3     嘟嘟可一族住在哪里？
:zh4     机械巨熊是基础角色设定吗？
:zh5     可莉会把嘟嘟可称作普通挂件吗？
:zh6     嘟嘟可是由谁做出来送给可莉的？
:zh7     嘟嘟可一族远行时想寻找什么？
```

其中 `:zh4` 是二创边界测试；合格回答应说明它来自本知识库的二创故事，而非基础角色事实。
`:zh5` 至 `:zh7` 不在中文验收集的 14 道固定题中，用于观察模型对相近但不同问法的泛化能力。

### 7.4 中文专用模型

如需放弃英文问答并训练中文专用模型，请使用 `build_dataset_zh.py` 生成独立的
`dataset_zh.txt`。完整的数据构成、GPU 训练、验收与回退流程见
[`docs/README_zh.md`](docs/README_zh.md)；该中文教学指南还包含 BPE 一致性、哈希校验、
PC/板端对照与串口诊断。以下是可直接复现的命令索引。

#### 1. 生成中文数据集

在仓库根目录执行。该命令只生成 `dataset_zh.txt`，不会覆盖当前训练集：

```powershell
python build_dataset_zh.py
```

预期输出约 5,600 条平衡中文问答；其中通用聊天/能力边界约 47%、嘟嘟可知识库约 26%、
计算和换算约 28%。训练记录的 `User:` / `Bot:` 是 `main.py` 的解析标记，不是英文对话数据。

#### 2. 备份并启用中文训练集

首次运行时保存当前数据；已有备份不会被覆盖：

```powershell
if (-not (Test-Path dataset.before-chinese-only.txt)) { Copy-Item dataset.txt dataset.before-chinese-only.txt }
Copy-Item dataset_zh.txt dataset.txt -Force
```

数据变化后必须重新训练 BPE tokenizer：

```bash
python3 -m pipenv run python3 train_tokenizer.py
```

#### 3. 分段 GPU 训练

下面的示例在 WSL 中使用第二张可见 GPU（RTX 3090）。`--stop-after 2000` 表示**本次新增**
2,000 个优化步骤，而不是总训练次数：

```bash
cd /mnt/d/github/espllm
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train --stop-after 2000
```

默认每 200 步、达到本次停止点、早停或按下 `Ctrl+C` 时，都会保存：

```text
model/model_esp32s3.pt.trainstate
```

该文件包含模型、optimizer、学习率调度器、迭代数、最佳验证损失、早停计数和随机数状态。
可用 `--save-interval N` 调整保存频率。

#### 4. 本地中文评测

每个训练段结束后，先运行 14 道关键词回归测试：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 evaluate_zh.py --target esp32s3 --allow-failures
```

评测读取 `data/chinese/qa_eval_zh.jsonl`，逐题打印模型回答和命中的关键词。`--allow-failures`
让脚本始终返回成功状态，便于先观察结果；移除此参数则只要存在失败题目就返回非零状态。

#### 5. 恢复继续训练

若评测未达标，使用同一份 `dataset.txt` 和同一套 `bpe-vocab.json` / `bpe-merges.txt` 继续：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train --resume --stop-after 2000
```

恢复期间不能替换训练数据或重新训练 tokenizer，否则 `.trainstate` 会拒绝恢复以防止训练状态与
数据不一致。按 `Ctrl+C` 后重新执行同一条 `--resume` 命令即可。

#### 6. 导出、烧录与模型版本校验

评测满意后，在**刚才完成训练和评测的同一 WSL Pipenv 环境**导出当前模型：

```bash
cd /mnt/d/github/espllm
python3 -m pipenv run python3 convert_model_to_c.py esp32s3
```

导出会同时更新以下生成物：

- `src/model_weights.hpp`：量化权重、词表、byte-to-token 表与 `bpe-merges.txt` 的 merge rank 表；
- `src/model_fingerprint.hpp`：`model_weights.hpp` 的完整 SHA-256；

必须在每次训练 tokenizer 或切换模型后重新执行导出。仅更新 `.pt.best` / `.quantized` 而未重新导出，
会造成 Python 与 ESP32 的 BPE token 序列不一致。

导出日志必须包含：

```text
Wrote src/model_weights.hpp
Model HPP SHA-256: <64 位十六进制值>
model_bpe_byte_tokens[256], model_bpe_merges[1791]
```

之后在 Windows 的 ESP-IDF PowerShell 编译、烧录与打开监视器：

```powershell
& C:\esp\v5.5.5\esp-idf\export.ps1
cd D:\github\espllm
idf.py build
idf.py -p COM3 flash
idf.py -p COM3 monitor
```

若 `flash` 报串口被占用，先退出旧的 `idf.py monitor` 或其他 COM3 串口软件。N16R8 的模型应用分区
接近满载，构建日志出现约 2% 剩余空间的警告属于当前配置的预期状态，但不能再加入 OTA 分区或明显增大模型。

启动日志的 `HPP SHA-256` 必须和 WSL 中的结果相同：

```bash
sha256sum src/model_weights.hpp
```

哈希相同表示烧录固件构建时使用的 `.hpp` 与当前文件一致；它避免仅凭回答文本或 token ID 猜测模型版本。

#### 7. 板端中文测试、进度与诊断

烧录后可输入 ASCII 命令测试真实中文推理，适合不支持中文 IME 的串口终端：

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

普通 `:zhN` 命令会立即显示 ASCII 思考进度，避免 12 层模型预填充时看起来无响应；回答 token 会
先缓存、最后一次性输出，保证拆分到多个 BPE token 的 UTF-8 中文字符不会被 USB Serial/JTAG 打断：

```text
Bot: [thinking...]........
Bot: 嘟嘟可是可莉的专属玩偶，由她的妈妈艾莉丝制作。
```

使用 `:dbgzh1` 可对 `:zh1` 进行详细诊断。它会打印 prompt 的 BPE token ID、每个预填充 token
的 12 层进度、首次采样的 top-5 logits、生成 token ID 与最终 UTF-8 解码结果。诊断中应满足：

```text
prefill token 3/9 (id=594)
prefill token 4/9 (id=758)
```

这两个 ID 应和 PC 输出一致。PC 端运行相同 prompt 的首 token 对照：

```bash
python3 -m pipenv run python3 tools/trace_first_token.py \
  --target esp32s3 \
  --question '嘟嘟可是谁的物品？'
```

若板端 token 与 PC 不同，按顺序检查：先确认 HPP SHA-256，再重新运行 `convert_model_to_c.py`，
最后重新 `build` 和 `flash`。不要仅重烧录旧的 `.hpp`。

#### 8. 回退到原训练数据

中文专用模型会明显弱化英文能力。如需恢复原始语料：

```powershell
Copy-Item dataset.before-chinese-only.txt dataset.txt -Force
```

#### GPU / WSL Training and Stop Conditions

Confirm that the active PyTorch build can access CUDA before starting a long run:

```bash
python3 -m pipenv run python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

On a multi-GPU WSL host, select one GPU explicitly. For example, use the second visible GPU
(an RTX 3090 with 24 GB VRAM in the reference setup) for single-GPU ESP32-S3 training:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train
```

The ESP32-S3 profile runs for at most 25,000 iterations. It evaluates every 200 iterations and
early-stops only after 12 consecutive validation evaluations fail to improve the best loss
(at most 2,400 iterations after the latest improvement). A run whose validation loss keeps
improving will therefore continue until the 25,000-iteration limit.

Each validation improvement writes `model/model_esp32s3.pt.best`. To stop a satisfactory run
early, press `Ctrl+C`; the best checkpoint already written remains available. Export it (or the
automatically created `.quantized` checkpoint after a normal finish) and rebuild the IDF firmware:

```bash
python3 convert_model_to_c.py esp32s3
```

```powershell
& C:\esp\v5.5.5\esp-idf\export.ps1
idf.py build
idf.py -p COM3 flash
```

Realistic expectations: the S3 bot is coherent and personable inside its drilled lane
(greetings, chit-chat, jokes, facts, simple Q&A, exact arithmetic via the on-device harness),
single-turn only, greedy-decoded, and unreliable outside its training distribution.
The highest-ROI future upgrade is **distillation** — replacing template answers with
frontier-model outputs and retraining.
