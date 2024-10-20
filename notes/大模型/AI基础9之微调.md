# 三、微调侧

## 1、大模型微调的主要步骤

大模型微调如上文所述有很多方法，并且对于每种方法都会有不同的微调流程、方式、准备工作和周期。然而大部分的大模型微调，都有以下几个主要步骤，并需要做相关的准备：

1. **准备数据集**：收集和准备与目标任务相关的训练数据集。确保数据集质量和标注准确性，并进行必要的数据清洗和预处理。
2. **选择预训练模型/基础模型**：根据目标任务的性质和数据集的特点，选择适合的预训练模型。
3. **设定微调策略**：根据任务需求和可用资源，选择适当的微调策略。考虑是进行全微调还是部分微调，以及微调的层级和范围。
4. **设置超参数**：确定微调过程中的超参数，如学习率、批量大小、训练轮数等。这些超参数的选择对微调的性能和收敛速度有重要影响。
5. **初始化模型参数**：根据预训练模型的权重，初始化微调模型的参数。对于全微调，所有模型参数都会被随机初始化；对于部分微调，只有顶层或少数层的参数会被随机初始化。
6. **进行微调训练**：使用准备好的数据集和微调策略，对模型进行训练。在训练过程中，根据设定的超参数和优化算法，逐渐调整模型参数以最小化损失函数。
7. **模型评估和调优**：在训练过程中，使用验证集对模型进行定期评估，并根据评估结果调整超参数或微调策略。这有助于提高模型的性能和泛化能力。
8. **测试模型性能**：在微调完成后，使用测试集对最终的微调模型进行评估，以获得最终的性能指标。这有助于评估模型在实际应用中的表现。
9. **模型部署和应用**：将微调完成的模型部署到实际应用中，并进行进一步的优化和调整，以满足实际需求。



## 2、huggingface_transformer

### (1) Tokenizer

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html

#### 简介

`Tokenizer` 的目标是：将文本转换为模型可以处理的数据，模型只能处理数字

- `Word-based Tokenizer`：将文本根据指定条件，拆分为一个字符，比如根据空格

    > `Word-based Tokenizer` 会得到一些非常大的词表，迫使模型学习一个巨大的 `embedding matrix` ，这导致了空间复杂度和时间复杂度的增加

- `Character-based Tokenizer`：将文本拆分为字符，而不是单词

    - 好处：

        - 词表规模要小得多（通常只有几十甚至几百）
        - `unknown token` 要少得多（因为任意单词都可以从字符构建）

    - 不足：

        - 每个字符本身并没有多少语义，因此 `Character-based Tokenizer` 往往伴随着性能的损失

        - 会 得到更大量的 `token` ，增大了模型的负担

- `Subword-based Tokenizer`：是 `word-based tokenizer` 和 `character-based tokenizer` 的折中

    > 原则：不应将常用词拆分为更小的子词`subword` ，而应将低频词分解为有意义的子词，使得能够使用较小的词表进行相对较好的覆盖，并且几乎没有 `unknown token`

#### Subword Tokenization 算法

三种常见的 `subword tokenization` 算法：

- `Byte Pair Encoding: BPE`：迭代式地替换序列中最频繁的字节对，即合并频繁的字符或字符序列

    - BPE 通过逐步合并频率最高的字符或子词对，生成新的子词单元。
    - 初始时，所有的单词都被分解成字符级别的子词。
    - 算法不断合并出现频率最高的子词对，直到达到预定的词汇量大小。
    - 这种方法可以有效减少未登录词的数量，因为大部分新词可以通过已有的子词组合表示出来。

- `WordPiece`：类似 BPE，区别在于 `merge` 的方式不同，`WordPiece` 不是选择最高频的 `pair` ，而是通过公式计算每个 `pair` 得分，选取得分最高的一对 token

    - WordPiece 类似于 BPE，但它的合并规则是基于最大似然估计（MLE），以最大化训练语料库的似然。
    - 起初，单词也被分解成单个字符，然后根据统计信息合并子词，生成新的子词单元。
    - WordPiece 会在每一步选择能最大化语言模型得分的子词合并，这种选择方式比 BPE 更加复杂

    > 公式为：merge 后的 t12 在预料库中的频次除以 t1 的频次乘以 t2 的频次 
    >
    > 注意：`WordPiece` 通过添加前缀（在 `BERT` 中是 `##`）来识别子词，这可以识别一个子词是否是单词的开始
    >
    > 二者区别：
    >
    > - `WordPiece` 仅保存最终词表，而不保存学到的 `merge rule` 
    > - 对于 unknow 词：
    >     - **BPE**：通过分解成已知子词，若无完全匹配则继续分解到字符级别
    >     - **WordPiece**：尝试最大化匹配已知子词，无法完全匹配时也会分解到字符级别

- `Unigram(SentencePiece)`：假设每个子词都是独立出现，因此子词序列出现的概率是每个子词出现概率的乘积

#### tokenizer 应用于文本的流程

- `Normalization`：标准化步骤，包括一些常规清理，例如删除不必要的空格、小写、以及删除重音符号

    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(type(tokenizer.backend_tokenizer))
    # <class 'tokenizers.Tokenizer'>
    
    normalizer = tokenizer.backend_tokenizer.normalizer
    print(normalizer.normalize_str("Héllò hôw are ü?"))
    # hello how are u?
    ```

- `Pre-tokenization`：将文本拆分为小的单元

    > 基于单词的 `tokenizer` 可以简单地基于空白和标点符号将原始文本拆分为单词，这些词将是`tokenizer`在训练期间可以学习的子词边界

    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(type(tokenizer.backend_tokenizer))
    # <class 'tokenizers.Tokenizer'>
    
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    print(pre_tokenizer.pre_tokenize_str("hello how are  u?")) # are 和 u 之间是双空格
    # [('hello', (0, 5)), ('how', (6, 9)), ('are', (10, 13)), ('u', (15, 16)), ('?', (16, 17))]
    
    ##GPT-2 tokenizer 也会在空格和标点符号上拆分，但它会保留空格并将它们替换为 Ġ 符号。注意，与 BERT tokenizer 不同，GPT-2 tokenizer 不会忽略双空格
    AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer.pre_tokenize_str("hello how are u?")  # are 和 u 之间是双空格
    # [('hello', (0, 5)),
    #  ('Ġhow', (5, 9)),
    #  ('Ġare', (9, 13)),
    #  ('Ġ', (13, 14)),
    #  ('Ġu', (14, 16)),
    #  ('?', (16, 17))]
    ```

- `Model`：执行 `tokenization` 从而生成 `token` 序列

- `Postprocessor`：针对具体的任务插入 `special token` ，以及生成 `attention mask` 和 `token-type ID` 

    > 通过添加特殊标记、处理序列对、生成位置和类型嵌入、处理溢出和填充等步骤，确保分词后的结果符合模型的输入要求

<img src="data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 706"></svg>">

```python
from tokenizers import pre_tokenizers

# 使用 WordPiece 模型
model = models.WordPiece(unk_token="[UNK]") # 未设置 vocab, 因为词表需要从数据中训练
tokenizer = Tokenizer(model)

################# Step1: Normalization ###################
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(),  
     # NFD Unicode normalizer, 否则 StripAccents normalizer 无法正确识别带重音的字符
     normalizers.Lowercase(), 
     normalizers.StripAccents()]
) # 这个整体等价于 normalizers.BertNormalizer(lowercase=True)

print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
# hello how are u?

################# Step2: Pre-tokenization ###################
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), 
     pre_tokenizers.Punctuation()]
) # 这个整体等价于 pre_tokenizers.BertPreTokenizer()

print(tokenizer.pre_tokenizer.pre_tokenize_str("This's me  ."))
# [('This', (0, 4)), ("'", (4, 5)), ('s', (5, 6)), ('me', (7, 9)), ('.', (11, 12))]

################# Step3: Trainer ###################
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

################# Step4: dataset ###################
from datasets import load_dataset # pip install datasets
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"] # batch size = 1000

################# Step5: train ####################
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# tokenizer.train(["wikitext-2.txt"], trainer=trainer) # 也可以从文本文件来训练

## 测试训练好的 WordPiece
encoding = tokenizer.encode("This's me  .")
print(encoding)
# Encoding(num_tokens=5, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [1511, 11, 61, 1607, 18]
print(encoding.type_ids)
# [0, 0, 0, 0, 0]
print(encoding.tokens)
# ['this', "'", 's', 'me', '.']
print(encoding.offsets)
# [(0, 4), (4, 5), (5, 6), (7, 9), (11, 12)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [0, 0, 0, 0, 0]
print(encoding.overflowing)
# []

################# Step6: Post-Processing ####################
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id)
# 2
print(sep_token_id)
# 3

tokenizer.post_processor = processors.TemplateProcessing(
    single= "[CLS]:0 $A:0 [SEP]:0",
    pair= "[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

## 测试训练好的 WordPiece(单个句子)
encoding = tokenizer.encode("This's me  .")
print(encoding)
# Encoding(num_tokens=7, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [2, 1511, 11, 61, 1607, 18, 3]
print(encoding.type_ids)
# [0, 0, 0, 0, 0, 0, 0]
print(encoding.tokens)
# ['[CLS]', 'this', "'", 's', 'me', '.', '[SEP]']
print(encoding.offsets)
# [(0, 0), (0, 4), (4, 5), (5, 6), (7, 9), (11, 12), (0, 0)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [1, 0, 0, 0, 0, 0, 1]
print(encoding.overflowing)
# []

## 测试训练好的 WordPiece(多个句子)
encoding = tokenizer.encode("This's me  .", "That's is fine-tuning.")
print(encoding)
# Encoding(num_tokens=17, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
print(encoding.ids)
# [2, 1511, 11, 61, 1607, 18, 3, 1389, 11, 61, 1390, 6774, 17, 4992, 1343, 18, 3]
print(encoding.type_ids)
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.tokens)
# ['[CLS]', 'this', "'", 's', 'me', '.', '[SEP]', 'that', "'", 's', 'is', 'fine', '-', 'tun', '##ing', '.', '[SEP]']
print(encoding.offsets)
# [(0, 0), (0, 4), (4, 5), (5, 6), (7, 9), (11, 12), (0, 0), (0, 4), (4, 5), (5, 6), (7, 9), (10, 14), (14, 15), (15, 18), (18, 21), (21, 22), (0, 0)]
print(encoding.attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)
# [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print(encoding.overflowing)
# []

################# Step7: Decode ####################
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids) # 注意：空格没有被还原
# "this's me. that's is fine - tuning."

################# Step8: Save ####################
tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")
print(new_tokenizer.decode(encoding.ids))
# this's me. that's is fine - tuning.
```

### (2) Datasets

 https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/2_datasets.html



### (3) Models

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/3_model.html



### (4) Trainer

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/4_trainer.html



### (5) Evaluator

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/5_evaluator.html



### (6) pipeline

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/6_pipeline.html 



### (7) Accelerate

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/7_accelerate.html

> `Accelerate` 是一个用于简化和加速分布式训练和推理的库，提供了一种更容易使用的接口来处理多 GPU、TPU 以及混合精度训练

### (8) Autoclass

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/8_autoclass.html

> `AutoClass` 是一个工具，用于自动选择和加载预训练的 Transformer 模型，简化了从 Hugging Face 模型库中加载模型的过程
>
> `AutoClass` 系列包括 `AutoModel`, `AutoTokenizer`, `AutoConfig` 等

### (9) 应用

https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/9_application.html



## 3、lora









## 4、text2sql(DB-GPT-Hub)

github 地址：https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/README.zh.md

### 4.1 spider 数据集拆分为训练集和测试集

```python
python dbgpt_hub/data_process/sql_data_process.py
```

```json
{
    "db_id": "department_management",
    "instruction": "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\"\n##Instruction:\ndepartment_management contains tables such as department, head, management. Table department has columns such as Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees. Department_ID is the primary key.\nTable head has columns such as head_ID, name, born_state, age. head_ID is the primary key.\nTable management has columns such as department_ID, head_ID, temporary_acting. department_ID is the primary key.\nThe head_ID of management is the foreign key of head_ID of head.\nThe department_ID of management is the foreign key of Department_ID of department.\n\n",
    "input": "###Input:\nHow many heads of the departments are older than 56 ?\n\n###Response:",
    "output": "SELECT count(*) FROM head WHERE age  >  56",
    "history": []
}
```

### 4.2 模型微调

#### (1) 参数配置

```shell
train_args = {
    "model_name_or_path": "/data/.modelcache/common-crawl-data/model-repo/meta-llama/Meta-Llama-3-8B-Instruct/124352e6d7f214295f9a43510123c51f2c2129de",
    "do_train": True,
    "dataset": "example_text2sql_train",
    "max_source_length": 2048,
    "max_target_length": 512,
    "finetuning_type": "lora",
    "lora_target": "q_proj, v_proj",
    "template": "llama2",
    "lora_rank": 64,
    "lora_alpha": 32,
    "output_dir": "dbgpt_hub/output/adapter/CodeLlama-3-8b-sql-lora",
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "lr_scheduler_type": "cosine_with_restarts",
    "logging_steps": 50,
    "save_steps": 2000,
    "learning_rate": 2e-4,
    "num_train_epochs": 8,
    "plot_loss": True,
    "bf64": True,
}
```

#### (2) HfArgumentParser

可以将==类对象==中的实例属性与解析参数互相转换，而类对象必须是通过 @dataclass() 创建的类对象

```python
def parse_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    Seq2SeqTrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments,
            GeneratingArguments,
        )
    )
    return _parse_args(parser, args)

def _parse_args(
    parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None
) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()
```

##### ModelArguments

```python
@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co."
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login`."
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    padding_side: Optional[Literal["left", "right"]] = field(
        default="left",
        metadata={"help": "The side on which the model should have padding applied."},
    )
    quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use double quantization in int4 training or not."
        },
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None, metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory(s) containing the delta model checkpoints as well as the configurations."
        },
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to plot the training loss after fine-tuning or not."
        },
    )
    hf_auth_token: Optional[str] = field(
        default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={
            "help": "Used in quantization configs. Do not specify this argument manually."
        },
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Used in rope scaling. Do not specify this argument manually."
        },
    )
    hf_hub_token: Optional[str] = field(
        default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."
        },
    )
```

##### DataArguments

```python
@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: str = field(
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        }
    )
    dataset: Optional[str] = field(
        default="example_text2sql",
        metadata={
            "help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."
        },
    )
    dataset_dir: Optional[str] = field(
        default="dbgpt_hub/data/",
        metadata={"help": "The name of the folder containing datasets."},
    )
    cutoff_len: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs after tokenization."},
    )
    reserved_label_len: Optional[int] = field(
        default=1,
        metadata={"help": "The maximum length reserved for label after tokenization."},
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "Enable streaming mode."}
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={
            "help": "Size of the buffer to randomly sample examples from in streaming mode."
        },
    )
    mix_strategy: Optional[
        Literal["concat", "interleave_under", "interleave_over"]
    ] = field(default="concat", metadata={"help": "Strategy to use in dataset mixing."})
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total output sequence length after tokenization."
        },
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"
        },
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "System prompt to add before the user query. Use `|` to separate multiple prompts in training."
        },
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={
            "help": "Size of the development set, should be an integer or a float in range `[0,1)`."
        },
    )
    predicted_input_filename: Optional[str] = field(
        default="dbgpt_hub/data/example_text2sql_dev.json",
        metadata={"help": "Predict input filename to do pred "},
    )
    predicted_out_filename: Optional[str] = field(
        default="pred_sql.sql",
        metadata={"help": "Filename to save predicted outcomes"},
    )
```

##### Seq2SeqTrainingArguments





##### FinetuningArguments

```python
@dataclass
class FinetuningArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["sft", "rm"]] = field(
        default="sft", metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full", "none"]] = field(
        default="lora", metadata={"help": "Which fine-tuning method to use."}
    )
    num_hidden_layers: Optional[int] = field(
        default=32,
        metadata={
            "help": 'Number of decoder blocks in the model for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: ["32", "40", "60", "80"], \
                  LLaMA-2 choices: ["32", "40", "80"], \
                  BLOOM choices: ["24", "30", "70"], \
                  Falcon choices: ["32", "60"], \
                  Baichuan choices: ["32", "40"] \
                  Qwen choices: ["32"], \
                  XVERSE choices: ["40"], \
                  ChatGLM2 choices: ["28"],\
                  ChatGLM3 choices: ["28"]'
        },
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of trainable layers for partial-parameter (freeze) fine-tuning."
        },
    )
    name_module_trainable: Optional[
        Literal["mlp", "self_attn", "self_attention"]
    ] = field(
        default="mlp",
        metadata={
            "help": 'Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: ["mlp", "self_attn"], \
                  BLOOM & Falcon & ChatGLM2   & ChatGLM3choices: ["mlp", "self_attention"], \
                  Baichuan choices: ["mlp", "self_attn"], \
                  Qwen choices: ["mlp", "attn"], \
                  LLaMA-2, InternLM, XVERSE choices: the same as LLaMA.'
        },
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            "help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                  BLOOM & Falcon & ChatGLM2  & ChatGLM3 choices: ["query_key_value", "self_attention.dense", "mlp.dense"], \
                  Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                  Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                  LLaMA-2, InternLM, XVERSE choices: the same as LLaMA.'
        },
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to resume training from the last LoRA weights or create new weights after merging them."
        },
    )
    ppo_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization in PPO Training."}
    )
    dpo_beta: Optional[float] = field(
        default=0.1, metadata={"help": "The beta parameter for the DPO loss."}
    )
```

##### GeneratingArguments

```python
@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use sampling, use greedy decoding otherwise."
        },
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k filtering."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."
        },
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty."
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length that is used with beam-based generation."
        },
    )
```

#### (3) 数据加载

```python
datasets.load_dataset(
	path: str, #数据集的名字或者路径(比如"imdb" 或 "json"、“csv”、“parquet”、“text”)
    name: Optional[str] = None, #表示数据集中的子数据集，当一个数据集包含多个数据集时，就需要这个参数
    data_dir: Optional[str] = None, #数据集所在的目录
    #表示本地数据集文件
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    #None，则返回一个DataDict对象，包含多个DataSet数据集对象；如果给定的话，则返回单个DataSet对象
    split: Optional[Union[str, Split]] = None,
    #表示缓存数据的目录
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[DownloadMode] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    ignore_verifications="deprecated",
    #表示是否将数据集缓存在内存中，加载一次后，再次加载可以提高加载速度
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    #表示加载数据集的脚本的版本
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token="deprecated",
    task="deprecated",
    #True - 逐步流式传输数据对数据集进行迭代，此时返回[' IterableDataset ']或[' IterableDatasetDict ']
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    trust_remote_code: bool = None,
    **config_kwargs,
)
```

用于向数据集添加新列：

```python
if dataset_attr.system_prompt:  # add system prompt
    if data_args.streaming:
        dataset = dataset.map(lambda _: {"system": dataset_attr.system_prompt})
    else:
        dataset = dataset.add_column(
            "system", [dataset_attr.system_prompt] * len(dataset)
        )
```

#### (4) AutoTokenizer.from_pretrained

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer,
    split_special_tokens=model_args.split_special_tokens,
    padding_side="right",  # training with left-padded tensors in fp16 precision may cause overflow
    **config_kwargs
)
```

`AutoTokenizer.from_pretrained()` 用于加载预训练的文本处理模型的 `Tokenizer`，以便将文本数据转换为模型可以接受的输入格式

1. `pretrained_model_name_or_path (str)`：指定要加载的预训练模型的名称或路径

2. `inputs (additional positional arguments, *optional*)`：表示额外的位置参数，这些参数会传递给标记器（Tokenizer）的`__init__()`方法，允许进一步自定义标记器的初始化

3. `config ([PretrainedConfig], *optional*)`：用于确定要实例化的分词器类

4. `cache_dir (str, optional)`：用于缓存模型文件的目录路径

5. `force_download(bool, optional)`：设置为 `True` 将强制重新下载模型配置，覆盖任何现有的缓存

6. `resume_download(bool, optional)`：可选参数，如果设置为 True，则在下载过程中重新开始下载，即使部分文件已经存在

7. `proxies(Dict[str, str], *optional*)`：可选参数，用于指定代理服务器的设置

    ``` 
    proxies = { "http": "http://your_http_proxy_url", "https": "https://your_https_proxy_url" }
    ```

8. `revision(str, optional)`：指定要加载的模型的 Git 版本（通过提交哈希）

9. `subfolder(str, *optional*)`：如果相关文件位于 huggingface.co 模型仓库的子文件夹内，请在这里指定

10. `use_fast(bool, *optional*, defaults to True)`：指示是否强制使用 fast tokenizer

11. `tokenizer_type(str, *optional*)`：用于指定要实例化的分词器的类型

12. `trust_remote_code(bool, *optional*, defaults to False)`：是否从 Hugging Face 加载模型

```python
if isinstance(tokenizer, PreTrainedTokenizerBase) 
		and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
    tokenizer.__class__.register_for_auto_class()
```

#### (5) AutoConfig.from_pretrained

```python
config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)
```

用于根据预训练模型的名称或路径创建配置对象，预训练模型的配置包含了关于模型架构、超参数和其他重要信息的设置

1. `pretrained_model_name_or_path(str, optional)`：指定要加载的预训练模型的名称或路径

2. `cache_dir(str, optional)`：指定用于缓存预训练模型配置文件的目录路径，如果设置为 `None` 将使用默认缓存目录

3. `force_download(bool, optional)`：如果设置为 `True`，将强制重新下载模型配置，覆盖任何现有的缓存

4. `resume_download(bool, optional)`：可选参数，如果设置为 True 则在下载过程中重新开始下载，即使部分文件已经存在

5. `proxies(Dict[str, str], *optional*)`：（可选参数）：这是一个字典，用于指定代理服务器的设置

    ```
    proxies = { "http": "http://your_http_proxy_url", "https": "https://your_https_proxy_url" }
    ```

6. `revision(str, optional)`：指定要加载的模型的 Git 版本（通过提交哈希）

7. `return_unused_kwargs(bool, optional, 默认值为 False)`：设置为 True 将返回未使用的配置参数

    > 这对于识别和检查传递给函数的不受支持或未识别的参数很有用

8. `trust_remote_code(bool, *optional*, defaults to False)`：

    - 默认设置为 True 将下载来自 Hugging Face 模型中心或其他在线资源的配置文件
    - 设置为 False 表示希望加载本地的配置文件

#### (6) AutoModel.from_pretrained

```python
model = AutoModelForCausalLM.from_pretrained(
    model_to_load,
    config=config,
    torch_dtype=model_args.compute_dtype,
    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    **config_kwargs
)
```

用于加载预训练的深度学习模型，允许加载各种不同的模型，而无需为每个模型类型编写单独的加载代码

1. `pretrained_model_name_or_path (str)`：用于指定要加载的预训练模型的名称或路径

2. `model_args`：直接传参的方式，传入配置项

    - 将编码器层数改为3层

        ```python
        model = AutoModel.from_pretrained("./models/bert-base-chinese", num_hidden_layers=3)
        ```

    - 加载模型时，指定配置类实例

        ```python
        model = AutoModel.from_pretrained("./models/bert-base-chinese", config=config)
        ```

3. `trust_remote_code*(bool, *optional*, defaults to False)`：

    - 默认为 True 将下载来自 Hugging Face 模型中心或其他在线资源的配置文件
    - 设置为 False 表示加载本地的配置文件

---

`AutoModelForCausalLM.from_pretrained`：是一个便捷的类，根据提供的模型名称自动加载适合因果语言建模的预训练模型

### 4.3 Initialize Trainer

```python
trainer = Seq2SeqPeftTrainer(
    finetuning_args=finetuning_args,
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
    compute_metrics=ComputeMetrics(tokenizer)
    if training_args.predict_with_generate
    else None,
    **split_dataset(dataset, data_args, training_args)
)
```

#### (1) dataset

```python
dataset = get_dataset(model_args, data_args)

dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, "sft")
```

preprocess_dataset 逻辑：

```python
dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=column_names, **kwargs
)

if not data_args.streaming:
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

preprocess_function =     
def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    max_length = data_args.max_source_length + data_args.max_target_length

    for query, response, history, system in construct_example(examples):
        input_ids, labels = [], []

        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, query, response, history, system
        ):
            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[: data_args.max_source_length]
            if len(target_ids) > data_args.max_target_length:
                target_ids = target_ids[: data_args.max_target_length]

            if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                break

            input_ids += source_ids + target_ids
            labels += [IGNORE_INDEX] * len(source_ids) + target_ids

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs

def get_template_and_fix_tokenizer(
    name: str, tokenizer: "PreTrainedTokenizer"
) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    additional_special_tokens = template.stop_words

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    tokenizer.add_special_tokens(
        dict(additional_special_tokens=additional_special_tokens),
        replace_additional_special_tokens=False,
    )
    return template
```

#### (2) DataCollatorForSeq2Seq

```python
#IGNORE_INDEX = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    label_pad_token_id=IGNORE_INDEX
    if data_args.ignore_pad_token_for_loss
    else tokenizer.pad_token_id,
)
```



#### (3) Override the decoding parameters of Seq2SeqTrainer

```python
training_args_dict = training_args.to_dict()
training_args_dict.update(
    dict(
        generation_max_length=training_args.generation_max_length
        or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams
        or training_args.generation_num_beams,
    )
)
training_args = Seq2SeqTrainingArguments(**training_args_dict)
```



### 4.4 Training

```python
if training_args.do_train:
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()
    if trainer.is_world_process_zero() and model_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
```

 





### 4.5 Evaluation





### 4.6 Predict







#### (5) 数据预处理





#### (8) 创建 data_collator