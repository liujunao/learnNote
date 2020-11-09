# 一、认识 Kafka

## 1、快速入门

- **启动 ZooKeeperr**：`bin/zookeeper-server-start.sh config/zookeeper.properties` 

  > 终端输出 `0.0.0.0/0.0.0.0:2181` 表明 zookeeper 已成功在端口 2181 启动

- **启动 kafka**：`bin/kafka-server-start.sh config/server.properties`

- **创建 topic**：`bin/kafka-topics.sh --create --zookeeper localhost:2181 --partitions 1 --replication-factor 1 --topic test` 

  > - **主题(topic)**：用于消息的发送与接收，此处将创建一个名为 `test` 的 topic，该 topic 只有一个分区(partition)，且该 partition  也只有一个副本(replication)处理消息
  >
  > - 查看 topic 状态：`bin/kafka-topics.sh --describe --zookeeper localhost:2181 --topic test`
  >
  >   <img src="../../pics/kafka/kafka_1.png">

- **创建生产者(发送消息)**：`bin/kafka-console-producer.bat --broker-list localhost:9092 --topic test` 

  > kafka 提供的脚本工具：可以不断接收标准输入并将它们发送到 kafka 的某个 topic

- **创建消费者(接收消息)**：`bin/kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic test --from-beginning` 

## 2、消息引擎(消息队列/消息中间件)系统

**消息引擎系统 `MS`**：用于在不同应用间传输消息的系统

> 消息引擎系统以软件接口为主要形式，实现了松耦合的异步式数据语义传递





<img src="../../pics/kafka/kafka_2.png">

## 3、kafka 概要设计







## 4、kafka 基本概念与术语







## 5、kafka 使用场景







# 二、kafka 发展历史

## 1、kafka 历史





## 2、kafka 版本





## 3、选择kafka版本







# 三、kafka 线上环境部署

## 1、集群环境规划





## 2、伪分布式环境安装





## 3、多节点环境安装





## 4、验证部署





## 5、参数设置









# 四、producer 开发

## 1、producer 概览





## 2、构造 producer





## 3、消息分区机制





## 4、消息序列化





## 5、producer 拦截器





## 6、无消息丢失配置





## 7、消息压缩





## 8、多线程处理





## 9、旧版本 producer







# 五、consumer 开发

## 1、consumer 概览





## 2、构建 consumer





## 3、订阅 topic





## 4、消息轮询





## 5、位移管理





## 6、重平衡





## 7、解序列化





## 8、多线程消费实例





## 9、独立 consumer





## 10、旧版本 consumer









# 六、kafka 设计原理

## 1、broker 端设计架构





## 2、producer 端设计





## 3、consumer 端设计





## 4、实现精确一次处理语义







# 七、管理 kafka 集群

## 1、集群管理





## 2、topic 管理





## 3、topic 动态配置管理





## 4、consumer 相关管理







## 5、topic 分区管理





## 6、kafka 常见脚本工具





## 7、API 方式管理集群











# 八、监控 kafka 集群

## 1、集群健康度检查





## 2、MBean 监控





## 3、broker 端 JMX 监控





## 4、clients 端 JMX 监控





## 5、JVM 监控





## 6、OS 监控





## 7、主流监控框架







# 九、调优kafka集群

## 1、确定调优目标





## 2、集群基础调优





## 3、调优吞吐量





## 4、调优延时





## 5、调优持久性





## 6、调优可用性







# 十、kafka Connect 与 kafka Streams

## 1、kafka connect





## 2、kafka streams