# 嘟嘟可知识库：数据制作、训练与验收

## 目标与边界

本数据集为当前 ESP32-S3 单轮问答模型增加“嘟嘟可”主题知识。固件当前每次提问会清空上下文，因此本流程只制作单轮 `问题 → 简短回答`，不把长篇故事直接作为模型上下文。

`测试知识库.md` 混合了基础角色资料、故事内容、语录与二创扩写。二创内容（如机械巨熊、神之心碎片）必须回答为“本知识库的二创故事设定”，不能表述为基础角色事实。

## 文件与职责

| 文件 | 用途 |
| --- | --- |
| `data/duduke/qa_train.jsonl` | 可训练问答；每条保留类别和来源。 |
| `data/duduke/qa_eval.jsonl` | 独立验收题；绝不能合并入训练集。 |
| `tools/export_knowledgebase_dataset.py` | 校验 JSONL 并导出 `main.py` 所需的 `User:/Bot:` 文本。 |
| `data/duduke/dataset_duduke.txt` | 导出的训练文本，默认仅包含知识库问答。 |

## 制作规则

每条训练记录必须有 `instruction`、`response`、`category`、`source` 四个字段。回答以 20–100 个中文字符为宜，直接、可验证，且不包含 `User:` 或 `Bot:` 前缀。

可用类别：

- `canon`：基础角色资料；
- `story`：本知识库明确给出的故事内容；
- `quote`：可莉语录所反映的角色表达；
- `fanfic`：二创剧情，回答必须带“二创故事”边界；
- `meta`：数据来源和回答边界。

新增内容前先确认来源。不要把未经验证的百科、时效新闻或不同版本设定混入 `canon`。同一个事实最多保留少量自然问法，避免大量近似改写污染验证指标。

## 导出与质量检查

在仓库根目录执行：

```powershell
python tools/export_knowledgebase_dataset.py
```

输出会报告训练条数、评测条数和类别分布。导出脚本会拒绝缺字段记录，并移除意外的重复 `User:` / `Bot:` 前缀。

只用该知识库从零训练会严重过拟合，不能用于通用聊天模型。若要与通用语料合并，先保留原文件，再生成新文件：

```powershell
Copy-Item dataset.txt dataset.before-duduke.txt
python tools/export_knowledgebase_dataset.py --base dataset.before-duduke.txt --knowledge-repeat 200 --output dataset.duduke-mixed.txt
```

当前基础集约十万条，知识库训练集约四十条；`--knowledge-repeat 200` 会使知识库样本约占合并集的 7%–8%。这是起始比例，不是固定真值：若通用聊天能力变差则降低比例；若主题问答仍不稳定则提高到 300 后重新评测。

确认效果后，手动将 `dataset.duduke-mixed.txt` 作为本轮训练数据；当前 `main.py` 固定读取根目录 `dataset.txt`，因此请先备份后替换：

```powershell
Copy-Item dataset.txt dataset.before-duduke.txt
Copy-Item dataset.duduke-mixed.txt dataset.txt
```

## 重新训练

数据变更后必须重训分词器，再训练模型。Windows 命令：

```powershell
python train_tokenizer.py
python main.py --target=esp32s3 --train
```

WSL 中使用 RTX 3090（第二张可见 GPU）的示例：

```bash
python3 -m pipenv run python3 train_tokenizer.py
CUDA_VISIBLE_DEVICES=1 python3 -m pipenv run python3 main.py --target=esp32s3 --train
```

训练会在验证集不再改善时自动早停，最佳权重保存为 `model/model_esp32s3.pt.best`。完成后会生成量化模型；随后执行既有的模型导出、ESP-IDF 编译和烧录流程。

## 验收

不要只看 `val loss`：它会受模板相似度影响。训练完成后用 `qa_eval.jsonl` 中的题逐条在模型命令行提问，并记录：

- 基础事实是否正确；
- 故事问题是否能回答关键实体；
- 二创边界题是否明确说“二创故事”，而非当作基础事实；
- 回答是否出现 `Bot: Bot:`、无关续写或过长重复。

建议至少 10/12 题回答关键点正确，且两道二创边界题均合格，才导出并烧录该轮模型。评测题若已经被模型训练过，必须新建未见题替换后再验收。
