# 零、仓库及操作

操作步骤：https://github.com/eosphoros-ai/DB-GPT-Hub/blob/main/README.zh.md

> 说明：评估时，需要在 evaluate 函数执行前，执行 `nltk.download('punkt')`

# 一、训练框架

## 1、Pytorch 原生训练





## 2、Deepspeed 训练

https://zhuanlan.zhihu.com/p/624412809



# 二、数据处理&llama

## 1、待训练数据的结构

每条数据，结构由三部分组成：instruction (指令)、input(输入)和output(输出)：

- **instruction(指令)**：定义了要求 AI 执行的任务或问题，是一条明确的指示，告诉AI需要做什么

    > 例如，”识别以下句子中的名词”或”我应该投资股票吗？”

- **input(输入)**：提供了执行指令所需的具体信息或上下文

    > 在某些情况下，这个部分可能为空，表示指令本身已经包含了执行任务所需的所有信息

- **output(输出)**：是 AI 根据给定的指令和输入生成的答案或结果，是 AI 处理完输入信息后的响应或解决方案

```json
{
  "db_id": "department_management",
  "instruction": "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\"\n##Instruction:\ndepartment_management contains tables such as department, head, management. Table department has columns such as Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees. Department_ID is the primary key.\nTable head has columns such as head_ID, name, born_state, age. head_ID is the primary key.\nTable management has columns such as department_ID, head_ID, temporary_acting. department_ID is the primary key.\nThe head_ID of management is the foreign key of head_ID of head.\nThe department_ID of management is the foreign key of Department_ID of department.\n\n",
  "input": "###Input:\nHow many heads of the departments are older than 56 ?\n\n###Response:",
  "output": "SELECT count(*) FROM head WHERE age  >  56",
  "history": []
}
```

## 2、数据处理类

### 2.1 argparse

> `argparse` 是一个用来解析命令行参数的 Python 库

用法案例：

```python
import argparse

parser = argparse.ArgumentParser() #初始化
parser.add_argument("--code_representation", help="Enable code representation", default=False) #定义
args = parser.parse_args() #解析
code_representation=args.code_representation #获取
```

详情参看：https://blog.csdn.net/edc3001/article/details/113788716

### 2.2 HfArgumentParser

> HfArgumentParser 是 Transformer 框架中的命令行解析工具，是 ArgumentParser 的子类，用于从类对象中创建解析对象

用法案例：

```python
from transformers import HfArgumentParser

#定义参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    
#构造 HfArgumentParser 对象
parser = HfArgumentParser((ModelArguments,))

#解析
Tuple[ModelArguments] (model_args,) = parser.parse_dict(args)
```

详情参看：https://zhuanlan.zhihu.com/p/296535876

### 2.3 datasets





详情参看：https://huggingface.co/docs/transformers/main/zh/preprocessing

## 3、llama 介绍

https://www.53ai.com/news/qianyanjishu/1120.html

https://www.zhihu.com/tardis/zm/art/653303123?source_id=1003

https://zhuanlan.zhihu.com/p/645608937



# 三、模型微调

## 1、微调参数

**train_sft.sh** 中关键参数与含义介绍：

- `model_name_or_path`：所用 LLM 模型的路径
- `dataset`：取值为训练数据集的配置名字，对应在 dbgpt_hub/data/dataset_info.json 中外层 key 值，如 example_text2sql
- `max_source_length`：输入模型的文本长度，本教程的效果参数为 2048，为多次实验与分析后的最佳长度
- `max_target_length`：输出模型的 sql 内容长度，设置为 512
- `template`：项目设置的不同模型微调的 lora 部分，对于 Llama2 系列的模型均设置为 llama2
- `lora_target`：LoRA 微调时的网络参数更改部分
- `finetuning_type`：微调类型，取值为 [ ptuning、lora、freeze、full ] 等
- `lora_rank`：LoRA 微调中的秩大小
- `loran_alpha`：LoRA 微调中的缩放系数
- `output_dir`：SFT 微调时 Peft 模块输出的路径，默认设置在 dbgpt_hub/output/adapter/路径下
- `per_device_train_batch_size`：每张 gpu 上训练样本的批次，如果计算资源支持，可以设置为更大，默认为 1
- `gradient_accumulation_steps`：梯度更新的累计 steps 值
- `lr_scheduler_type`：学习率类型
- `logging_steps`：日志保存的 steps 间隔
- `save_steps`：模型保存的 ckpt 的 steps 大小值
- `num_train_epochs`：训练数据的 epoch 数
- `learning_rate`：学习率，推荐的学习率为 2e-4





## 1、训练方法

（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO 训练、DPO 训练和 ORPO 训练





## 2、微调精度

32 比特全参数微调、16 比特冻结微调、16 比特 LoRA 微调和基于 AQLM/AWQ/GPTQ/LLM.int8 的 2/4/8 比特 QLoRA 微调



## 3、微调算法

GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ 和 Agent 微调



## 4、微调技巧

FlashAttention-2、Unsloth、RoPE scaling、NEFTune 和 rsLoRA。







# 四、模型推理/预测

## 1、推理框架-vLLM













# 五、模型合并











# 六、模型评估































