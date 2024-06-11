# 一、Agent 实现方式

## 1、ReAct(Reason和Action)

> 参考论文：https://arxiv.org/abs/2210.03629

ReAct：模型推理分成两部分，Reason和Action

- Reason生成分析步骤，Action生成工具调用请求，二者交替进行直到得到最终的结果

### 1.1 简介

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

<img src="../../pics/llm/llm_3.png">

## 2、plan-and-execute

> 参考论文：https://arxiv.org/abs/2305.04091

让模型先理解问题并制定解决方案的计划

- 执行逻辑：
    - 让模型先理解问题并制定解决方案的计划，解决 Missing-step Error
    - 让模型按步骤执行计划并解决问题，解决 Semantic Misunderstanding Error

<img src="../../pics/llm/llm_4.png">

- 

# 二、流式处理

## 1、Flowable

### 1.1 核心函数







### 1.2 Flowable(RxJava) 与 Flux(Reactor)





### 1.3 背压







## 2、retrofit









## 3、SSE











# 三、Milvus

- 简介：

    - https://blog.csdn.net/weixin_47336776/article/details/123994038
    - https://blog.51cto.com/liguodong/5110587

- 索引：

    - 数据写入&索引构建：https://blog.51cto.com/liguodong/5110583

    - 索引性能说明：https://blog.csdn.net/weixin_44839084/article/details/103471083

    - 各个索引的详细简介：https://blog.csdn.net/weixin_47336776/article/details/123994584

    - **nlist和nprobe**：nlist 是调用 create_index 时设置的参数，nprobe 则是调用 search 时设置的参数

        > IVFLAT 和 SQ8 索引都是通过聚类算法把大量的向量划分成很多‘簇’（也叫‘桶’)
        >
        > - nlist 指的就是聚类时划分桶的总数。通过索引查询时，第一步先找到和目标向量最接近的若干个桶，第二步在这若干个桶里通过比较向量距离查找出最相似的 k 条向量
        > - nprobe 指的就是第一步若干个桶的数量
        >
        > ---
        >
        > - 增大 nlist 会使得桶数量变多，每个桶里的向量数量减少，所需的向量距离计算量变小，因此搜索性能提升，但由于比对的向量数变少，有可能会遗漏正确的结果，因此准确率下降；
        > - 增大 nprobe 就是搜索更多的桶数，因此计算量变大，搜索性能降低，但准确率上升
        >
        > 推荐的 nlist 值为4 * sqrt(n)，其中 n 为数据的向量总数；
        >
        > 而 nprobe 的值则需要综合考虑在可接受的准确率条件下兼顾效率，比较好的做法是通过多次实验确定一个合理的值
        >
        > 详情见：https://cloud.tencent.com/developer/article/1607298

- 性能优化：https://developer.aliyun.com/article/1375747





# 四、bce&colbert







# 五、可视化编排

参考 dify 框架：

- 官网：https://dify.ai/zh
- github：https://github.com/langgenius/dify/blob/main/README_CN.md







# 六、多模态支持















