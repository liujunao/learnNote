# \# 第一部分：基础篇

# 一、MapReduce 设计理念与基本架构

## 1、Hadoop 发展史





## 2、MapReduce 设计目标







## 3、MapReduce 编程模型







## 4、Hadoop 基本架构

### (1) HDFS 架构





### (2) MapReduce 架构







## 5、MapReduce 作业的生命周期







# \# 第二部分：MapReduce 编程模型篇

## 二、MapReduce 编程模型

## 1、MapReduce 编程模型概述







## 2、MapReduce API 基本概念

### (1) 序列化





### (2) Reporter 参数







### (3) 回调机制







## 3、Java API 解析

### (1) 作业配置与提交







### (2) InputFormat 接口







### (3) Output 接口







### (4) Mapper 与 Reducer 解析







### (5) Partitioner 接口







## 4、非 Java API 解析

### (1) Hadoop Streaming







### (2) Hadoop Pipes









## 5、Hadoop 工作流

### (1) JobControl





### (2) ChainMapper/ChainReducer







### (3) Hadoop 工作流引擎









# \# 第三部分：MapReduce 核心设计篇

# 三、Hadoop RPC 框架解析

## 1、概述





## 2、Java 基础

### (1) 反射与动态代理







### (2) 网络编程







### (3) NIO





## 3、Hadoop RPC 框架

### (1) RPC 概念







### (2) Hadoop RPC







### (3) 集成其他开源 RPC







## 4、MapReduce 通信协议分析

### (1) MapReduce 通信协议







### (2) JobSubmissionProtocol 通信协议







### (3) InterTrackerProtocol 通信协议







### (4) TaskUmbilicalProtocol 通信协议







### (5) 其他通信协议





# 四、作业提交与初始化过程解析

## 1、概述





## 2、作业提交过程详解

### (1) 执行 Shell 命令





### (2) 作业文件上传





### (3) 产生 InputSplit 文件







### (4) 作业提交到 JobTracker







## 3、作业初始化过程详解







## 4、Hadoop DistributedCache 原理分析

### (1) 使用方法





### (2) 工作原理





# 五、JobTracker 内部实现剖析

## 1、JobTracker 概述





## 2、JobTracker 启动过程分析

### (1) JobTracker 启动过程概述





### (2) 重要对象初始化





### (3) 各种线程功能





### (4) 作业恢复







## 3、心跳接收与应答

### (1) 更新状态





### (2) 下达命令







## 4、Job 和 Task 运行时信息维护

### (1) 作业描述模型





### (2) JobInProgress





### (3) TaskInProgress





### (4) 作业和任务状态转换图





## 5、容错机制

### (1) JobTracker 容错







### (2) TaskTracker 容错







### (3) Job/Task 容错







### (4) Record 容错







### (5) 磁盘容错







## 6、任务推测执行原理

### (1) 计算模型假设





### (2) 1.0.0 版本算法





### (3) 0.21.0 版本算法







### (4) 2.0 版本算法







## 7、Hadoop 资源管理

### (1) 任务调度框架分析





### (2) 任务选择策略分析





### (3) FIFO 调度器分析





### (4) Hadoop 资源管理优化







# 六、TaskTracker 内部实现剖析

## 1、TaskTracker 概述







## 2、TaskTracker 启动过程分析

### (1) 重要变量初始化







### (2) 重要对象初始化





### (3) 连接 JobTracker







## 3、心跳机制

### (1) 单次心跳发送





### (2) 状态发送





### (3) 命令执行







## 4、TaskTracker 行为分析

### (1) 启动新任务





### (2) 提交任务





### (3) 杀死任务





### (4) 杀死作业





### (5) 重要初始化







## 5、作业目录管理







## 6、启动新任务

### (1) 任务启动过程分析







### (2) 资源隔离机制







# 七、Task 运行过程分析

## 1、概述





## 2、基本数据结构和算法

### (1) IFile 存储格式





### (2) 排序





### (3) Reporter







## 3、Map Task 内部实现

### (1) Map Task 整体流程





### (2) Collect 过程分析





### (3) Spill 过程分析





### (4) Combine 过程分析







## 4、Reduce Task 内部实现

### (1) Reduce Task 整体流程





### (2) Shuffle 和 Merge 阶段分析





### (3) Sort 和 Reduce 阶段分析







## 5、Map/Reduce Task 优化

### (1) 参数调优





### (2) 系统优化





# \# 第四部分：MapReduce 高级篇

# 八、Hadoop 性能调优

## 1、概述







## 2、从管理员角度调优

### (1) 硬件选择





### (2) 操作系统参数调优





### (3) JVM 参数调优





### (4) Hadoop 参数调优







## 3、从用户角度调优

### (1) 应用程序编写规范







### (2) 作业级别参数调优







### (3) 任务级别参数调优





# 九、Hadoop 多用户作业调度器

## 1、多用户调度器产生背景





## 2、HOD

### (1) Torque 资源管理器





### (2) HOD 作业调度







## 3、Hadoop 队列管理机制





## 4、Capacity Scheduler 实现

### (1) Capacity Scheduler 功能介绍





### (2) Capacity Scheduler 实现





### (3) 多层队列调度







## 5、Fair Scheduler 实现

### (1) Fair Scheduler 功能介绍





### (2) Fair Scheduler 实现





### (3) Fair Scheduler 与 Capacity Scheduler 对比





## 6、其他 Hadoop 调度器介绍





# 十、Hadoop 安全机制

## 1、Hadoop 安全机制概述

### (1) Hadoop 面临的安全问题





### (2) Hadoop 对安全方面的需求







### (3) Hadoop 安全设计基本原则







## 2、基础知识

### (1) 安全认证机制





### (2) Kerberos 介绍







## 3、Hadoop 安全机制实现

### (1) RPC





### (2) HDFS





### (3) MapReduce





### (4) 上层服务







## 4、应用场景总结

### (1) 文件存取





### (2) 作业提交与运行







### (3) 上层中间件访问 Hadoop





# 十一、下一代 MapReduce 框架

## 1、第一代 MapReduce 局限性







## 2、下一代 MapReduce 概述

### (1) 基本设计思想





### (2) 资源统一管理平台







## 3、YARN

### (1) YARN 基本框架





### (2) YARN 工作流程







### (3) YARN 设计细节





### (4) MapReduce 与 YARN 结合







## 4、Corona

### (1) Corona 基本框架





### (2) Corona 工作流程





### (3) YARN 与 Corona 对比







## 5、Mesos

### (1) Mesos 基本框架





### (2) Mesos 资源分配





### (3) MapReduce 与 Mesos 结合

