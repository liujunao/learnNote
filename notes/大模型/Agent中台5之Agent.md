# 一、ReAct(Reason和Action)

> 参考论文：https://arxiv.org/abs/2210.03629

ReAct：模型推理分成两部分，Reason和Action

- Reason生成分析步骤，Action生成工具调用请求，二者交替进行直到得到最终的结果

## 1、简介

- 执行逻辑：
    - 用户给 Agent 一个任务
    - 思考： Agent “思考 “要做什么
    - 行动/行动输入： Agent 决定采取什么行动（又称使用什么工具）以及该工具的输入应该是什么
    - 工具的输出
- 记忆方式：
    - memory：使用传统存储记录输入、输出
    - 步骤记忆：保留一个与该任务相关的中间 Agent 步骤，并将完整的列表传递给LLM调用
- Action Agent 存在的问题：
    - Calculation Error： 由于计算错误带来的回答答案错误；
    - Missing-step Error： 当涉及多个步骤时，有时会遗漏一些中间推理步骤；
    - Semantic Misunderstanding Error：对用户输入问题的语义理解和推理步骤的连贯性方面的其他错误，可能是由于 LLM（语言模型）能力不足导致
- 局限：只适合处理简单场景，不适合处理复杂场景

<img src="/Users/admin/alltext/learnNote/pics/llm/llm_3.png">

# 二、plan-and-execute

> 参考论文：https://arxiv.org/abs/2305.04091

让模型先理解问题并制定解决方案的计划

- 执行逻辑：
    - 让模型先理解问题并制定解决方案的计划，解决 Missing-step Error
    - 让模型按步骤执行计划并解决问题，解决 Semantic Misunderstanding Error

<img src="/Users/admin/alltext/learnNote/pics/llm/llm_4.png">

# 三、十种Agent规划实现

参考链接：https://liduos.com/llm-agent-planning.html



# 四、测评机制

https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/K.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%87%AA%E5%8A%A8%E8%AF%84%E4%BC%B0%E7%90%86%E8%AE%BA%E5%92%8C%E5%AE%9E%E6%88%98--LLM%20Automatic%20Evaluation.md
