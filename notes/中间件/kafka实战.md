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

消息引擎系统的两个重要因素：

- **消息设计**：要考虑语义的清晰和格式上的通用性，能完整清晰表达业务的能力

  > 为了更好地表达语义以及最大限度地提高重用性，消息通常采用**结构化方式进行设计**，比如：
  >
  > - SOAP 协议的消息采用 XML 格式
  > - WebService 支持 JSON 格式的消息

- **传输协议设计**：

  - 侠义角度：消息传输协议指定了消息在不同系统间传输的方式，如：AMQP、Web Service + SOAP
  - 广义角度：这类协议可能包括任何能在不同系统间传输消息或执行语义操作的协议或框架，如：RPC

---

**消息引擎范型**：一个基于网络的架构范型，描述了消息引擎系统的两个不同的子部分是如何互连且交互

- **消息队列模型**：基于队列提供消息传输服务，多用于进程间通信及线程间通信

  > 该模型定义了消息队列、发送者、接收者，提供了一种点对点的消息传递方式，即：
  >
  > - 发送者发送每条消息到队列指定位置，接收者从指定位置获取消息
  > - 一旦消息被消费，就会从队列中移除该消息
  > - 每条消息由一个发送者生产，且只被一个消费者处理
  >
  > <img src="../../pics/kafka/kafka_3.png">

- **发布/订阅模型**：发布者将消息生产出来发送到指定的 topic 中，所有订阅了该 topic 的订阅者都可以接收到该 topic 下的所有消息

  > 主题 `topic`：可理解为逻辑语义相近的消息容器
  >
  > - 通常具有相同订阅 topic 的所有订阅者将接收到同样的消息
  >
  > <img src="../../pics/kafka/kafka_4.png">

---

Java 消息服务 `JMS`：是一套 API 规范，提供了很多接口用于实现分布式系统间的消息传递

## 3、kafka 概要设计

> kafka 为解决超大量级数据的实时传输

### (1) 吞吐量/延时

- **吞吐量**：某种处理能力的最大值

  > kafka 吞吐量：每秒能处理的消息数或每秒能处理的字节数

- **延时**：性能指标，衡量一段时间间隔内，如：发出某个操作到接收到操作响应之间的时间或在系统中导致某些物理变更的起始时刻与变更正式生效时刻间的间隔

  > kafka 延时：表示客户端发起请求与服务器处理请求并发送响应给客户端间的这段时间

---

- **kafka 的写入操作很快**：每次写入操作都只是把数据写入到操作系统的页缓存中，然后由操作系统自行决定何时把页缓存中的数据写回磁盘，这样的优势

  - 操作系统页缓存是在内存中分配的，所以消息写入的速度非常快

  - Kafka 不必直接与底层的文件系统打交道，所以烦琐的 I/O 操作都交由操作系统来处理

  - Kafka 写入操作采用追加写入方式，避免了磁盘随机写操作，即只能在日志文件末尾追加写入新的消息，且不允许修改已写入的消息，因此属于磁盘顺序操作

    > 磁盘的顺序读/写操作很快，其速度甚至可以匹敌内存的随机 I/O 速度
    >
    > <img src="../../pics/kafka/kafka_5.png">

- **Kafka 使用零拷贝读取**：先尝试从页缓存读取，若命中便把消息经页缓存直接发送到网络的 Socket

  > - **无零拷贝**：数据传输过程涉及内核态与用户态的上下文切换，CPU 开销很大
  >
  >   <img src="../../pics/kafka/kafka_6.png">
  >
  > - **`sendfile` 系统调用(零拷贝)**：在内核驱动程序处理 I/O 数据时，不用直接存储器访问技术`DMA` 执行I/O操作，因此避免了 OS 内核缓冲区间的数据拷贝
  >
  >   <img src="../../pics/kafka/kafka_7.png">

Kafka 实现高吞吐量、低延时目标的操作：

1. 大量使用操作系统页缓存，内存操作速度快且命中率高
2. Kafka 不直接参与物理 I/O 操作，而是交由 OS 完成
3. 采用追加写入方式，摒弃了缓慢的磁盘随机读/写操作
4. 使用 `sendfile` 为代表的零拷贝技术加强网络间的数据传输效率

> **页缓存好处**：当 kafka 进程崩溃时，堆内存上的数据也一并消失，但页缓存的数据依然存在，下次 kafka 重启后可以继续提供服务

### (2) 消息持久化

- **把消息持久化到磁盘的好处**：
  - **解耦消息发送与消息消费**：通过将消息持久化使得生产者不再需要直接和消费者耦合，只是简单的把消息生产出来并交由 Kafka 服务器保存即可，因此提升了整体的吞吐量
  - **实现灵活的消息处理**：已处理的消息可能在未来某个时间点需重新处理一次，消息持久化可以方便实现

- **持久化方式**：

  - 普通系统持久化：先尽量使用内存，当内存资源耗尽时，再一次性地把数据“刷盘”

  - Kafka 持久化：所有数据会立即被写入文件系统的持久化日志中，之后 Kafka 服务器才会返回结果给客户端通知其消息已被成功写入

    > 好处：即实时保存数据，又减少 Kafka 程序对于内存的消耗，从而将节省出的内存留给页缓存使用

### (3) 负载均衡和故障转移

- **负载均衡**：让系统负载根据一定的规则均衡地分配在所有参与工作的服务器上，从而最大限度地提升系统整体的运行效率

  > kafka 通过**智能化的分区领导者选举**实现负载均衡

- **故障转移**：当服务器意外中止时，整个集群可以快速地检测到该失效，并立即将该服务器上的应用或服务自动转移到其他服务器上

  > 通常以“心跳”或“会话”机制实现，即只要主服务器与备份服务器之间的心跳无法维持或主服务器注册到服务中心的会话超过过期时间，就认为主服务器已无法正常运行，集群会自动启动某个备份服务器来替代
  >
  > ---
  >
  > Kafka 使用**会话机制**来支持故障转移：
  >
  > - 每台 Kafka 服务器启动后，会以会话形式把自己注册到 ZooKeeper 服务器上
  > - 一旦该服务器运行出现问题，与 ZooKeeper 的会话便不能维持从而超时失效
  > - 此时，Kafka 集群会选举出另一台服务器来完全代替这台服务器继续服务

### (4) 伸缩性

- **伸缩性**：表示向分布式系统中增加额外的计算资源时吞吐量提升的能力

- **Kafka 伸缩性**：每台 Kafka 服务器上的状态统一交由 ZooKeeper 保管

  > Kafka 服务器上并不是所有状态都不保存，其只保存了很轻量级的内部状态，因此整个集群间维护状态一致性的代价很低

---

分布式集群的每台服务器会维护很多内部状态：

- 若由服务器来保存状态信息，则必须要处理一致性问题

- 若服务器无状态，状态的保存和管理交于专门的协调服务，则整个集群的服务器间就无须繁重的状态共享

  > 倘若要扩容集群节点，只需简单地启动新的节点机器进行自动负载均衡即可

---

- 通过消息持久化，Kafka 实现高可靠性
- 通过负载均衡和文件系统的独特设计，Kafka 实现高吞吐量
- 通过故障转移，Kafka 实现高可用性
- 通过 ZooKeeper 状态保存，Kafka 实现伸缩性

## 4、kafka 基本概念与术语

### (1) 消息

消息由消息头、key、value 组成：**kafka 使用二进制字节数组 ByteBuffer 保存消息信息**

- **消息头**：消息的 CRC 码、消息版本号、属性、时间戳、键长度、消息体长度

    > 属性：1 字节，目前只使用了最低的 3 位用于保存消息的压缩类型(当前只支持 0-无压缩、1-GZIP、2-Snappy、3-LZ4)

- **Key(消息键)**：对消息做 partition 时使用，即决定消息被保存在某 topic 下的哪个 partition

- **Value(消息体)**：保存实际的消息数据

- **Timestamp(消息发送时间戳)**：用于流式处理及其他依赖时间的处理语义，若不指定则取当前时间

<img src="../../pics/kafka/kafka_8.png"  align=left width="800">

### (2) topic 和 partition

- **topic**：逻辑概念，代表一类消息，topic 可被多个消费者订阅

    > 通常使用 topic 来区分实际业务，如：业务 A 使用一个 topic，业务 B 使用另一个 topic

- **topic-partition-message 三级结构来分散负载**：每个 Kafka topic 都由若干个 partition 组成

- **partition 是不可修改的有序消息日志**：

    - 每个 partition 有专属的 partition 号，通常从 0 开始

    - 用户对 partition 的操作：在消息序列的尾部追加写入消息

        > partition 的每条消息都会被分配一个唯一的序列号(位移-offset)
        >
        > - 位移值从 0 开始顺序递增
        >
        > - 位移信息可以唯一定位到某 partition 的一条消息

> partition 为提升系统的吞吐量，因此创建 topic 时可根据集群实际配置设置具体的 partition 数，实现整体性能的最大化

<img src="../../pics/kafka/kafka_9.png" align=left width="600">

### (3) offset(位移)

kafka 消费者端也有位移 `offset` 概念：

<img src="../../pics/kafka/kafka_10.png" align=left width="700">

**由上知**：每条消息在某个 partition 的位移固定，但消费该 partition 的消费者的位移会随着消费进度不断前移

> 通过 `<topic,partition,offset>` 三元组可以在 kafka 集群中找到唯一对应的消息

### (4) replica(副本)

**replica(副本)目的**：防止数据丢失

- **领导者副本 `leader replica`**：提供服务

- **追随者副本 `follower replica`**：不能提供服务给客户端，即不负责响应客户端发来的消息写入和消息消费请求

    > 只是被动地向领导者副本获取数据，而一旦 leader replica 所在的 broker 宕机，Kafka 会从剩余的 replica 中选举新的 leader

### (5) leader 和 follower

- **leader-follower 系统**：只有 leader 对外提供服务，follower 只是被动地追随 leader 状态，保持与 leader 的同步

- **follower 作用**：充当 leader 的候补，即一旦 leader 挂掉就会有一个 follower 被选举成为新的 leader

**kafka 保证同一个 partition 的多个 replica 分配在不同 broker上**：

<img src="../../pics/kafka/kafka_11.png" align=left width="700">

### (6) ISR(in-sync replica)

**ISR(与 leader replica 保持同步的 replica 集合)**：kafka 为 partition 动态维护一个 replica 集合

- 该集合中的所有 replica 保存的消息日志都与 leader replica 保持同步状态
- 只有该集合中的 replica 才能被选举为 leader
- 只有该集合中所有的 replica 都接收到同一条消息，Kafka 才会将该消息置为“已提交”状态，即消息发送成功

**kafka 的消息交付承诺**：kafka 对于没有提交成功的消息不做任何交付保证，只保证在 ISR 存活的情况下“已提交”的消息不会丢失

> - 若 ISR 的部分 replica 落后 leader replica 到一定程度，则 kafka 会将这些 replica “踢”出 ISR
>
> - 若非 ISR 的 replica 与 leader replica 保持同步后，将会回到 ISR 中

## 5、kafka 使用场景

- **消息传输**：kafka 适合替代传统的消息总线或消息代理，即擅长解耦生产者和消费者以及批量处理消息

    > kafka 还具有更好的吞吐量特性，内置的分区机制和副本机制既实现了高性能的消息传输，还达到了高可靠性和高容错性

- **网络行为日志追踪**：kafka 超强吞吐量适合用于重建用户行为数据追踪系统

    > 场景：很多公司使用机器学习或其他实时处理框架来帮助收集并分析用户的点击流数据

- **审计数据收集**：可以便捷的对多路消息进行实时收集，同时由于持久化特性，使得后续离线审计成为可能

    > 满足的场景：从各个运维应用程序处实时汇总操作步骤信息进行集中式管理

- **日志收集**：可以使用 kafka 对分散的日志进行全量收集，并集中送往下游的分布式存储中

- **Event Sourcing(领域驱动设计--DDD)**：使用时间序列来表示状态变更

    > kafka 使用不可变的消息序列来抽象化表示业务消息，因此 kafka 适合这种应用的后端存储

- **流式处理**：kafka 推出了流式处理组件 -- Kafka Streams

# 二、kafka 线上环境部署

## 1、集群环境规划

- 操作系统选型
- 磁盘类型
- 磁盘容量
- 内存规划
- CPU 规划
- 网络带宽规划

---

**典型线上环境配置**：

<img src="../../pics/kafka/kafka_12.png" align=left width="600">

## 2、kafka 集群安装

> kafka 集群分为**单节点的伪分布式集群**和**多节点的分布式集群**两种

### (1) 伪分布式环境安装

**单节点伪分布式环境**：指集群由一台 ZooKeeper 服务器和一台 kafka broker 服务器组成

<img src="../../pics/kafka/kafka_13.png" align=left width="700">

### (2) 多节点环境安装

<img src="../../pics/kafka/kafka_14.png" align=left width="700">

## 3、参数设置

### (1) broker 端参数

**broker 端参数在 `config/server.properties` 文件中设置**：kafka 不支持动态修改

- `broker.id`：用于标识 broker，默认 `-1`，若不指定则会自动生成一个唯一值

- `log.dirs`：指定 kafka 持久化消息的目录，可设置多个(以 `,` 分隔(CSV)，推荐)，默认 `/tmp/kafka-logs`

    > 若机器有 N 块物理硬盘，则设置 N 个目录可同时执行均匀写操作

- `zookeeper.connect`：无默认值，可设置多个(以 `,` 分隔)

    > 若使用一套 ZooKeeper 管理多套 kafka 集群，则必须指定 `chroot`(默认根目录)

- `listeners`：用于 client 监听 broker，可设置多个(以 `,` 分隔)，格式为：`[协议]://[主机名]:[端口],...` 

    > - 若不指定主机名，则表示绑定默认网卡
    > - 若主机名为 `0.0.0.0`，则表示绑定所有网卡
    >
    > Kafka 支持的协议：`PLAINTEXT、SSL、SASL_SSL`

- `advertised.listeners`：类似 `listeners`，主要用于 IaaS 环境

- `unclean.leader.election.enable`：是否开启 unclean leader 选举，默认 `false`

    > `false` 表明 kafka 不允许从剩下存活的非 ISR 副本中选择一个当 leader

- `delete.topic.enable`：是否允许 kafka 删除 topic，默认 `true`，既允许用户删除 topic 及其数据

    > `kafka 0.9.0.0` 新增的 ACL 权限特性消除了误操作和恶意操作

- `log.retention.{hours|minutes|ms}`：设置消息数据的留存时间，若同时设置则优先级：`ms > minutes > hours`，默认 7 天

- `log.retention.bytes`：设置每个消息日志的大小，默认为 `-1`

    > - 对于大小超过该参数的分区日志，kafka 会自动清理该分区的过期日志段文件
    > - `-1`：表示 kafka 不会根据消息日志文件总大小来删除日志

- `min.insync.replicas`：与 producer 的 `acks` 配合使用，指定 broker 端必须成功响应 clients 消息发送的最少副本数，若 broker 端无法满足该条件，则 clients 的消息发送并不会被视为成功，与 `acks` 配合使用可以令 kafka 集群达成最高等级的消息持久化

    > `acks=-1` 表示 producer 端寻求最高等级的持久化保证，且此时 `min.insync.replicas` 才有意义

- `num.network.threads`：控制了一个 broker 在后台用于处理网络请求的线程数，默认 `3`

    > 通常，broker 启动时会创建多个线程处理来自其他 broker 和 clients 发送过来的各种请求
    >
    > - 注：此处的“处理”：只负责转发请求，将接收到的请求转发到后面的处理线程中
    >
    > 真实环境中，用户需不断监控 `NetworkProcessorAvgIdlePercent JMX` 指标，且建议该值不低于 `0.3`

- `num.io.threads`：控制 broker 端实际处理网络请求的线程数，默认 `8`，即 kafka broker 默认创建 8 个线程以轮询方式不停监听转发过来的网络请求并进行实时处理

    > kafka 也为该请求处理提供了一个 JMX 监控指标 `RequestHandlerAvgIdlePercent`，且建议该值不低于 `0.3`

- `message.max.bytes`：Kafka broker 能接收的最大消息大小，默认 `977KB` 

### (2) topic 级别参数

**topic 级别参数**：指覆盖 broker 端全局参数，每个不同的 topic 都可以设置自己的参数值

- `delete.retention.ms`：每个 topic 可以设置自己的日志留存时间以覆盖全局默认值
- `max.message.bytes`：覆盖全局，即为每个 topic 指定不同的最大消息尺寸
- `retention.bytes`：覆盖全局，即每个 topic 设置不同的日志留存尺寸

### (3) OS 参数

- **文件描述符限制**：kafka 会频繁地创建并修改文件系统中的文件，包括消息日志文件、索引文件、各种元数据管理文件等

    > 实际场景中，最好先增大进程能打开的最大文件描述符上限，设置方法：`ulimit -n xxx`

- **OS 级别的 Socket 缓冲区大小**：若做远距离的数据传输，则建议将 OS 级别的 Socket 缓冲区调大

- **建议 Ext4 或 XFS 文件系统**：生产环境建议 XFS 文件系统

- **关闭 swap**：`sysctlvm.swappiness=<一个较小的数>`，即大幅降低对 swap 空间的使用，以免极大的拉低性能

    > 注：不要显示设置该值为 0

- **设置更长的 flush 时间**：适当增大该值可以提升 OS 物理写入操作的性能，默认刷盘间隔为 5 秒

    > 因 kafka 依赖 OS 页缓存的“刷盘”功能实现消息真正写入物理磁盘

# 三、producer 开发

## 1、producer 概览

> kafka producer 负责向 kafka 写入数据，且每个 producer 都是独立工作，与其他 producer 实例之间没有关联

**producer**：向某个 topic 的某个分区发送一条消息

- **分区器 `partitioner`**：确认 producer 向 topic 的哪个分区写入消息

    > 对于每条待发送的消息：
    >
    > - 若指定了 key，则该 partitioner 会根据 key 的哈希值来选择目标分区
    > - 若未指定 key，则 partitioner 使用轮询方式确认目标分区(可以最大限度确保消息在所有分区上的均匀性)
    >
    > ---
    >
    > - producer 的 API 允许自行指定目标分区的权利，即用户可以在消息发送时跳过 partitioner 直接指定要发送到的分区
    >
    > - producer 也允许用户实现自定义的分区策略而非使用默认的 partitioner
    >
    >     > 这样，用户可以很灵活地根据自身的业务需求确定不同的分区策略

- **分区 leader `broker`**：topic 分区的副本中只有一个 leader 才能响应 clients 发送过来的请求，而剩下的副本会与 leader 保持同步

    > producer 有多种可选择的消息发送方式，比如：
    >
    > - 方式一：不等待任何副本的响应便返回成功
    > - 方式二：只是等待 leader 副本响应写入操作之后再返回成功

<img src="../../pics/kafka/kafka_15.png" align=left width="800">

**Java 版本 producer 工作原理**：

1. producer 首先使用一个线程(用户主线程)将待发送的消息封装进一个 ProducerRecord 类实例
2. 然后将其序列化后，发送给 partitioner
3. 再由 partitioner 确定目标分区后，一同发送到位于 producer 程序中的一块内存缓冲区中
4. producer 的工作线程(I/O 发送线程，即 Sender 线程)接着实时地从该缓冲区中提取出准备就绪的消息封装进一个批次 `batch`，统一发送给对应的 broker

## 2、构造 producer

### (1) producer 程序实例

**构造一个 producer 实例的 5 个步骤**：

1. **构造 `Properties` 对象**，并指定 `bootstrap.servers、key.serializer、value.serializer` 三个属性

    > - `bootstrap.servers`：指定 `host:port`，用于创建向 Kafka broker 服务器的连接
    >
    >     > - 若 Kafka 集群机器数很多，则只需指定部分 broker，因为 producer 会根据该参数找到并发现集群中所有的 broker
    >     >
    >     > - 若 broker 端没有显式配置 listeners 使用 IP 地址，则建议该参数配置为主机名
    >
    > - `key.serializer`：消息 key 的序列化器，kafka 为大部分的初始化类型默认提供了现成的序列化器
    >
    >     > 被发送到 broker 端的任何消息格式都必须是字节数组，因此消息的各个组件必须先序列化
    >
    > - `value.serializer`：指定消息体的序列化器，将消息 value 部分转换成字节数组

2. 使用 Properties 实例**构造 `KafkaProducer` 对象**

    > `KafkaProducer` 是 producer 的主入口，所有的功能基本上都由 KafkaProducer 提供
    >
    > ---
    >
    > 创建方式二：创建 producer 时可同时指定 key 和 value 的序列化类
    >
    > ```java
    > Serializer<String> keySerializer = new StringSerializer();
    > Serializer<String> valueSerializer = new StringSerializer();
    > Producer<String, String> producer = new KafkaProducer<>(properties, keySerializer, valueSerializer);
    > ```

3. **构造待发送的消息对象 `ProducerRecord`**，指定消息要被发送到的 topic、分区及对应的 key 和 value

    > 注意：可以只指定 topic 和 value，即分区和 key 信息可以不指定，由 Kafka 自行确定目标分区

4. 调用 KafkaProducer 的 `send` 方法**发送消息**

    > - **异步发送**：返回一个 Java Future 对象供用户稍后获取发送结果(回调机制)
    >
    >     > `send` 方法提供了回调类参数来实现异步发送以及对发送结果响应：
    >     >
    >     > ```java
    >     > //recordMetadata 与 e 不会同时为空，即发送成功--e==null，发送失败--recordMetadata==null
    >     > producer.send(record, new Callback() {
    >     >     @Override
    >     >     public void onCompletion(RecordMetadata recordMetadata, Exception e) {
    >     >         if (e == null) {
    >     >             //消息发送成功
    >     >         } else {
    >     >             //错误处理逻辑
    >     >         }
    >     >     }
    >     > });
    >     > ```
    >     >
    >     > 用户可以创建自定义的 Callback 实现类来处理消息发送后的逻辑，即实现 `org.apache.kafka.clients.producer.Callback` 接口即可
    >
    > - **同步发送**：调用 `Future.get()` 无限等待结果返回，即实现同步发送的效果
    >
    >     > 使用 `Future.get()` 会一直等待直至 Kafka broker 将发送结果返回给 producer 程序：
    >     >
    >     > - 当结果从 broker 处返回时，`get` 方法要么返回发送结果，要么抛出异常并交由 producer 自行处理
    >     >
    >     > - 若没有错误，`get` 将返回对应的 RecordMetadata 实例，包括消息发送的 topic、分区、对应分区的位移信息
    >     >
    >     >     RecordMetadata 实例包含已发送消息的所有元数据信息
    >     >
    >     > ```java
    >     > ProducerRecord<String, String> record = new ProducerRecord<>("my-topic", Integer.toString(i));
    >     > producer.send(record).get();
    >     > ```
    >
    > **kafka 的错误类型**：
    >
    > - **可重试异常**：继承自 `org.apache.kafka.common.errors.RetriableException` 抽象类
    >     - `LeaderNotAvailableException`：分区的 leader 副本不可用，出现在 leader 换届选举期间，重试可自行恢复
    >     - `NotControllerException`：controller 当前不可用，出现在选举期间，重试可自行恢复
    >     - `NetworkException`：网络瞬时故障导致的异常，可重试
    > - **不可重试异常**：该类异常表明了一些非常严重或 Kafka 无法处理的问题
    >     - `RecordTooLargeException`：发送的消息尺寸过大，超过了规定的上限
    >     - `SerializationException`：序列化失败
    >     - `KafkaException`：其他类型的异常

5. **关闭 `KafkaProducer`**：需要显式关闭

    > - `close()`：producer 会被允许先处理完之前的发送请求后再关闭，即所谓的“优雅”关闭退出
    > - `close(timeout)`：producer 等待 `timeout` 时间后强行退出(谨慎使用)

```java
package com.example.testspring.kafka;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.util.Properties;

public class ProducerTest {
    public static void main(String[] args) {
        Properties properties = new Properties();
        //下面三个参数必须指定
        properties.put("bootstrap.servers", "localhost:9092");
        //序列化器必须使用全限定名
        properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        //下面参数可选
        properties.put("acks", "-1");
        properties.put("retries", 3);
        properties.put("batch.size", 323840);
        properties.put("linger.ms", 10);
        properties.put("buffer.memory", 33554432);
        properties.put("max.bloack.ms", 3000);

        Producer<String, String> producer = new KafkaProducer<>(properties);
        for (int i = 0; i < 100; i++) {
            producer.send(new ProducerRecord<>("my-topic", Integer.toString(i), Integer.toString(i)));
        }
        producer.close();
    }
}
```

### (2) producer 主要参数

除了 `bootstrap.servers、key.serializer、value.serializer` 外，producer 还有其他参数：

- `acks`：控制 producer 生产消息的持久性

    > **producer 已提交消息持久性**：一旦消息被成功提交，则只要有一个保存了该消息的副本“存活”，则这条消息被视为“不会丢失”
    >
    > 注：producer API 提供了回调机制供用户处理发送失败的情况
    >
    > ---
    >
    > 具体：
    >
    > - 当 producer 发送一条消息给 kafka 集群时，这条消息会被发送到指定 topic 分区 leader 所在的 broker 上
    > - producer 等待从该 leader broker 返回消息的写入结果(具有超时时间)以确定消息被成功提交
    > - 这一切完成后，producer 可以继续发送新的消息
    >
    > kafka 能保证 consumer 不会读取到尚未提交完成的消息
    >
    > <img src="../../pics/kafka/kafka_16.png" align=left width="800">

- `buffer.memory`：指定了 producer 端用于缓存消息的缓冲区大小，单位字节，默认32MB

    > 由于采用异步发送消息的设计架构：
    >
    > - producer 启动时会先创建一块内存缓冲区用于保存待发送的消息
    > - 然后由另一个专属线程负责从缓冲区中读取消息执行真正的发送
    >
    > 若 producer 向缓冲区写消息的速度超过了专属 I/O 线程发送消息的速度：
    >
    > - producer 会停止手头的工作等待 I/O 线程追上来
    > - 若一段时间后 I/O 线程还是无法追上 producer 进度，则会抛出异常并期望用户介入进行处理

- `compression.type`：设置 producer 端是否压缩消息，默认 `none`

    > - producer 端压缩会降低网络 I/O 传输开销从而提升吞吐量，也会增加 producer 端机器的 CPU 开销
    > - 若 broker 端压缩参数与 producer 端设置不同，则 broker 端需要额外 CPU 进行相应的解压-重压操作
    >
    > kafka 支持 3 种压缩算法：`GZIP、Snappy、LZ4(性能最好)、Zstandard`

- `retries`：表示进行重试的次数，默认 0(不重试)

    > - 重试可能造成消息的重复发送
    > - 重试可能造成消息的乱序
    >
    > 注：producer 两次重试之间会停顿一段时间，以防止频繁重试对系统的冲击，`retry.backoff.ms` 指定停顿时间，默认 `100ms`

- `batch.size(重要)`：默认 `16KB`，对于调优 producer 吞吐量和延时性能指标都很重要

    > producer 会将发往同一分区的多条消息封装进一个 batch 中(不论是否填满，producer 都可能发送该 batch)
    >
    > - 若 batch 很小，则一次发送请求能够写入的消息数也很少，所以 producer 吞吐量会很低
    > - 若 batch 很大，则会给内存使用带来很大压力

- `linger.size`：控制消息发送的延时，默认 0(表示消息需要被立即发送，无需关系 batch 是否已被填满)

- `max.request.size`：控制 producer 端能够发送的最大消息大小

- `request.timeout.ms`：规定 producer 发送请求给 broker 后，broker 在什么时间内将处理结果返还给 producer，默认 `30s` 

    > 若 broker 在指定时间内没有响应 producer，则认为请求超时，并在回调函数中显式抛出 TimeoutException 异常

## 3、消息分区机制

### (1) 分区策略

producer 提供了分区策略及对应的分区器 `partitioner`，来确定将消息发送到指定 topic 的哪个分区

- 默认的 `partitioner` 会尽力确保具有相同 key 的所有消息都会被发送到相同的分区上
- 若没有为消息指定 key，则该 `partitioner` 会选择轮询方式来确保消息在 topic 的所有分区时尚均匀分配

### (2) 自定义分区机制

> producer 的默认 `partitioner` 根据 `murmur2` 算法计算消息 key 的哈希值，然后对总分区数求模得到消息要发送到的目标分区号

自定义分区机制的两个步骤：

1. 在 producer 程序中创建一个类，实现 `org.apache.kafka.clients.producer.Partitioner` 接口，主要分区逻辑在 `Partitioner.partition` 中实现
2. 在用于构造 KafkaProducer 的 Properties 对象中设置 `partitioner.class` 参数

```java
public interface Partitioner extends Configurable, Closeable {
    /**
     * 计算给定消息要被发送到哪个分区
     *
     * @param topic      topic 名称
     * @param key        消息键或 null
     * @param keyBytes   消息键值序列化字节数组或 null
     * @param value      消息体或 null
     * @param valueBytes 消息体序列化字节数组或 null
     * @param cluster    集群元数据
     * @return
     */
    int partition(String topic, Object key, byte[] keyBytes, Object value, 
                  byte[] valueBytes, Cluster cluster);

    /**
     * 关闭 partitioner
     */
    void close();

    default void onNewBatch(String topic, Cluster cluster, int prevPartition) {
    }
}
```

## 4、消息序列化

### (1) 默认序列化

- 序列化器 `serializer`：负责在 producer 发送前将消息转换成字节数组
- 解序列化器 `deserializer`：用于将 consumer 接收到的字节数组转换成相应的对象

<img src="../../pics/kafka/kafka_17.png" align=left width="800">

kafka 提供的序列化器：

- `ByteArraySerializer`：本质上不做任何处理(因为已经上字节数组)
- `ByteBufferSerializer`：序列化 ByteBuffer
- `BytesSerializer`：序列化 kafka 自定义的 Bytes 类
- `DoubleSerializer`：序列化 Double 类型
- `IntegerSerializer`：序列化 Integer 类型
- `LongSerializer`：序列化 Long 类型
- `StringSerializer`：序列化 String 类型

```java
//使用案例
properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
//或者
properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, 
               "org.apache.kafka.common.serialization.StringSerializer");
properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, 
               "org.apache.kafka.common.serialization.StringSerializer");
```

### (2) 自定义序列化

自定义 serializer 的步骤：

1. 定义数据对象格式

2. 创建自定义序列化类，实现 `org.apache.kafka.common.serialization.Serializer` 接口，在 `serializer` 方法实现序列化逻辑

3. 在用于构造 KafkaProducer 的 Properties 对象中设置 `key.serializer` 或 `value.serializer`

    > 取决于是为消息 key，还是 value 做自定义序列化

---

案例：

```java
package com.example.testspring.kafka.serializer;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {
    private String firstName;
    private String lastName;
    private int age;
    private String address;
}
```

```java
package com.example.testspring.kafka.serializer;

import org.apache.kafka.common.serialization.Serializer;
import org.codehaus.jackson.map.ObjectMapper;

import java.io.IOException;
import java.util.Map;

public class UserSerializer implements Serializer {
    private ObjectMapper objectMapper;

    @Override
    public void configure(Map configs, boolean isKey) {
        objectMapper = new ObjectMapper();
    }

    @Override
    public byte[] serialize(String topic, Object data) {
        byte[] ret = null;
        try {
            //使用 jackson-mapper-asl 包中的 ObjectMapper 直接把对象转成字节数组
            ret = objectMapper.writeValueAsString(data).getBytes("utf-8");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }
}
```

```java
package com.example.testspring.kafka.serializer;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class SerializerMain {
    //自定义序列化发送消息
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        Properties properties = new Properties();
        properties.put("bootstrap.servers", "localhost:9092");
        properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        properties.put("value.serializer", "com.example.testspring.kafka.serializer.UserSerializer");

        String topic = "test-topic";
        KafkaProducer<String, User> producer = new KafkaProducer<>(properties);
        //构建 User 实例
        User user = new User("XI", "HU", 33, "Beijing,China");
        //构建 ProducerRecord 实例
        ProducerRecord<String, User> record = new ProducerRecord<>(topic, user);
        producer.send(record).get();
        producer.close();

    }
}
```

## 5、producer 拦截器

`producer interceptor`：用于实现 clients 端的定制化控制逻辑

- `interceptor` 使得用户在消息发送前以及 producer 回调逻辑前有机会对消息做一些定制化需求，如：修改消息等

- producer 允许用户指定多个 interceptor 按序作用于同一条消息从而形成一个拦截链 `interceptor chain` 

    > producer 将按序调用，同时把每个 interceptor 中捕获的异常记录到错误日志中而不是向上传递

`interceptor` 的实现接口为 `org.apache.kafka.clients.producer.ProducerInterceptor`，定义方法如下：

- `onSend(ProducerRecord)`：该方法封装进 `KafkaProducer.send` 方法中，即运行在用户主线程中

    > - producer 确保在消息被序列化以计算分区前调用该方法
    > - 用户可以在该方法中对消息做任何操作，但最好保证不要修改消息所属的 topic 和分区，否则会影响目标分区的计算

- `onAcknowledgement(RecordMetadata, Exception)`：该方法会在消息被应答前或消息发送失败时调用，且通常都是在 producer 回调逻辑触发前

    > 运行在 producer 的 I/O 线程中，因此不要在该方法中放入很“重”的逻辑，否则会拖慢 producer 的消息发送效率

- `close`：关闭 interceptor，主要用于执行一些资源清理工作

---

案例：双 interceptor 组成的拦截链

- 第一个 interceptor 会在消息发送前将时间戳信息加到消息 value 的最前部
- 第二个 interceptor 会在消息发送后更新成功发送消息数或失败发送消息数

```java
package com.example.testspring.kafka.interceptor;

import org.apache.kafka.clients.producer.ProducerInterceptor;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;

import java.util.Map;

public class TimeStampPrependerInterceptor implements ProducerInterceptor<String, String> {
    @Override
    public ProducerRecord onSend(ProducerRecord record) {
        //将时间戳写入消息体的最前部
        return new ProducerRecord(record.topic(), record.partition(), record.timestamp(), record.key(),
                System.currentTimeMillis() + "," + record.value().toString());
    }

    @Override
    public void onAcknowledgement(RecordMetadata recordMetadata, Exception e) {

    }

    @Override
    public void close() {

    }

    @Override
    public void configure(Map<String, ?> map) {

    }
}
```

```java
package com.example.testspring.kafka.interceptor;

import org.apache.kafka.clients.producer.ProducerInterceptor;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;

import java.util.Map;

public class CounterInterceptor implements ProducerInterceptor<String, String> {
    private int errorCounter = 0;
    private int successCounter = 0;

    @Override
    public ProducerRecord<String, String> onSend(ProducerRecord<String, String> record) {
        return record;
    }

    @Override
    public void onAcknowledgement(RecordMetadata metadata, Exception exception) {
        if (exception == null) {
            successCounter++;
        } else {
            errorCounter++;
        }
    }

    @Override
    public void close() {
        //保存结果
        System.out.println("Successful sent: " + successCounter);
        System.out.println("Failed sent: " + errorCounter);
    }

    @Override
    public void configure(Map<String, ?> map) {

    }
}
```

```java
package com.example.testspring.kafka.interceptor;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class InterceptorMain {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        Properties properties = new Properties();
        properties.put("bootstrap.servers", "localhost:9092");
        properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        //构建拦截链
        List<String> interceptors = new ArrayList<>();
        interceptors.add("com.example.testspring.kafka.interceptor.TimeStampPrependerInterceptor");
        interceptors.add("com.example.testspring.kafka.interceptor.CounterInterceptor");
        properties.put(ProducerConfig.INTERCEPTOR_CLASSES_CONFIG, interceptors);
        //构建producer
        String topic = "test-topic";
        KafkaProducer<String, String> producer = new KafkaProducer<>(properties);
        //发送消息
        for (int i = 0; i < 10; i++) {
            ProducerRecord<String, String> record = new ProducerRecord<>(topic, "message" + i);
            producer.send(record).get();
        }
        producer.close();
    }
}
```

## 6、无消息丢失配置

### (1) producer 端配置

producer 问题：

- **问题一**：producer 采用异步发送机制，存在**数据丢失**的窗口，即若 I/O 线程发送前 producer 崩溃，则存储缓冲区中的数据全部丢失

    > `KafkaProducer.send` 仅把消息放入缓冲区，由专属 I/O 线程负责从缓冲区提取消息并封装进消息 batch，然后发送出去

- 问题二：**消息乱序** 

---

**producer 端”无消息丢失配置“**：

- `block.on.buffer.full=true`：使内存缓冲区被填满时 producer 处于阻塞状态并停止接收新的消息而不是抛出异常

    > `kafka 0.10.0.0` 后转而设置 `max.block.ms`

- `acks=all or -1`：必须等到所有 follower 都响应发送消息才能认为提交成功，即 producer 端最强程序的持久化保证

- `retries = Integer.MAX_VALUE`：即开启 producer  无限重试

    > producer 只会重试那些可恢复的异常，所以放心设置

- `max.in.flight.requests.per.connection=1`：防止 topic 同分区下的消息乱序问题

    > 该参数实际限制了 producer 在单个 broker 连接上能够发送的未响应请求的数量
    >
    > 若设置为 1，则 producer 在某个 broker 发送响应前将无法再给该 broker 发送 PRODUCE 请求

- 使用带回调机制的 send 发送消息，即 `KafkaProducer.send(record, callback)`：非回调的 send 不会理会消息发送的结果

- Callback 逻辑中显示地立即关闭 producer，使用 `close(0)`：为处理消息的乱序问题

    > 若不使用 `close(0)`，则 producer 会被允许将未完层的消息发送出去，这样可能造成消息乱序

### (2) broker 端配置

broker 端配置：

- `unclean.leader.election.enable=false`：关闭 unclean leader 选举，即不允许非 ISR 中的副本被选举为 leader，从而避免因日志水位截断而造成的消息丢失

- `replication.factor>=3`：使用多个副本来保存分区消息，参考了 Hadoop 及业界通用的三备份原则

- `min.insync.replicas>1`：用于控制某条消息至少被写入到 ISR 中的多少个副本才算成功，设置大于 1 时为了提升 producer 端发送语义的持久性

    > 只有在 producer 端 `acks` 被设置成 `all` 或 `-1` 时，该参数才有意义；实际使用时，不要使用默认值

- `replication.factor > min.insync.replicas`：若两者相等，则只要有一个副本挂掉，分区就无法正常工作

    > 虽然二者相等有很高的持久性，但可用性大大降低；推荐配置 `replication.factor = min.insync.replicas + 1`

- `enable.auto.commit=false` 

## 7、消息压缩

### (1) 简介

消息压缩是 I/O 性能和 CPU 资源的平衡：

- 优势：数据压缩显著地降低了磁盘占用或带宽占用，从而有效地提升了 I/O 密集型应用的性能
- 劣势：压缩会消耗额外的 CPU 时钟周期

**消息压缩发送与解压缩解析流程**：producer 端压缩、broker 端保持、consumer 端解压缩

- producer 端能将一批消息压缩成一条消息发送

- broker 端将这条压缩消息写入本地日志文件

    > 若某些前置条件不满足(比如：需要进行消息格式的转换等)，则 broker 端需要对消息进行解压缩然后再重新压缩

- consumer 端获取到这条压缩消息时，其会自动对消息进行解压缩，还原成初始的消息集合返还给用户

### (2) kafka 支持的压缩算法

kafka 支持的压缩算法：`GZIP、Snappy、LZ4、Zstandard`，`LZ4` 的压缩性能与吞吐量均最高

- 默认情况，kafka 不压缩消息
- 用户可以设定 producer 端参数 `compression.type` 来开启消息压缩，即构造 KafkaProducer 的属性对象时进行设置

---

调优 producer 的压缩性能：

- 是否开启压缩的依据：I/O 资源消耗与 CPU 资源消耗对比

    > - 若 I/O 资源紧张，如：producer 消耗了大量网络带宽或 broker 磁盘占用率高，而 producer 的 CPU 资源富裕，则可考虑为 producer 开启消息压缩
    > - 反之，不需要设置消息压缩以节省 CPU 时钟周期

- 压缩性能与 producer 的 batch 大小相关，batch 越大需要压缩的时间就越长

    > - batch 大小越大，压缩时间越长，不过时间不是线性增长，而是越来越平缓
    > - 若发现压缩很慢，说明系统的瓶颈在用户主线程而不是 I/O 发送线程，因此可考虑增加多个用户线程同时发送消息，这样可显著提升 producer 吞吐量

## 8、多线程处理

实际环境中，只使用一个用户主线程通常无法满足所需的吞吐量目标，因此需要构造多个线程或多个进程来同时给 Kafka 集群发送消息：

- **多线程单 KafkaProducer 实例**：全局构造一个 KafkaProducer 实例，然后在多个线程中共享使用

    > 由于 KafkaProducer 线程安全，因此这种使用方式也是线程安全

- **多线程多 KafkaProducer 实例**：在每个 producer 主线程中都构造一个 KafkaProducer 实例，并保证此实例在该线程中封闭

    > 若集群拥有超多分区，采用此种方法具有较高的可控性，方便 producer 的后续管理

<img src="../../pics/kafka/kafka_18.png" align=left width="800">

# 四、consumer 开发

> kafka 消费者(consumer)是从 kafka 读取数据的应用

## 1、consumer 概览

> 若干个 consumer 订阅 Kafka 集群中的若干个 topic 并从 Kafka 接收属于这些 topic 的消息

### (1) 消费者组(consumer group)

消费者使用一个消费者组名 `group.id` 来标记自己，topic 的每条消息都只会被发送到每个订阅它的消费者组的一个消费者实例上：

1. 一个 consumer group 可能有若干个 consumer 实例(一个 group 允许只有一个实例)
2. 对于同一个 group 而言，topic 的每条消息只能被发送到 group 下的一个 consumer 实例上
3. topic 消息可以被发送到多个 group 中

---

kafka 通过 consumer group 实现同时支持基于队列和基于发布/订阅的两种消息引擎模型：

- 基于队列的模型：所有 consumer 实例都属于相同 group，即每条消息只会被一个 consumer 实例处理
- 基于发布/订阅的模型：consumer 实例都属于不同 group，即若每个 consumer 实例都设置完全不同的 group，则 kafka 消息就会被广播到所有 consumer 实例上

---

- **consumer group 的优点**：用于实现高可伸缩性、高容错性的 consumer 机制

- **重平衡**：组内多个 consumer 实例可以同时读取 kafka 消息，而且一旦有某个 consumer 宕机，consumer group 会立即将已崩溃 consumer 负责的分区转交给其他 consumer 负责，从而保证整个 group 可以继续工作，不会丢失数据

    > 注：kafka 在为 consumer group 成员分配分区时可以做到公平分配

- kafka 目前只提供单个分区内的消息顺序，而不会维护全局的消息顺序

    > 若要实现 topic 全局的消息读取顺序，只能通过让每个 consumer group 下只包含一个 consumer 实例的方式来间接实现

---

consumer group 含义和特点总结：

- consumer group 下可以有一个或多个 consumer 实例，一个 consumer 实例可以使一个线程或运行在其他机器上的进程

- group.id 唯一标识一个 consumer group

- 对某个 group 而言，订阅 topic 的每个分区只能分配给该 group 下的一个 consumer 实例

    > 当然，该分区还可以被分配给其他订阅该 topic 的消费者组

### (2) 位移(offset)

> 此处的 offset 指代 consumer 端的 offset

- **位移**：每个 consumer 实例都会为其消费的分区维护属于自己的位置信息来记录当前消费了多少条消息

- **consumer group 保存 offset**：只需保存一个长整型数据

    > consumer 还引入了检查点机制定期对 offset 进行持久化，从而简化应答机制的实现

- **位移提交**：consumer 客户端定期向 kafka 集群汇报自己消费数据的进度

    > consumer 把位移提交到 kafka 的一个内部 topic(`__consumer_offsets`)上

- `__consumer_offset`：用于保存 consumer 位移

    > - `__consumer_offset` 由 kafka 自行创建，因此用户不可擅自删除该 topic 的所有信息
    >
    > - `__consumer_offset` 消息格式：KV 格式的消息
    >
    >     - `key`：一个 `group.id + topic + 分区号` 的三元组
    >     - `value`：offset 值
    >
    > - 每当更新同一个 key 的最新 offset 值，该 topic 就会写入一条含有最新 offset 的消息，同时 kafka 会定期对该 topic 执行压实操作，即为每个消息 key 只保存含有最新 offset 的消息
    >
    >     > - 既避免对分区日志消息的修改，也控制 __consumer_offset topic 总体的日志容量，同时还实时反应最新的消费进度
    >     >
    >     > - 该 topic 创建了 50 个分区，且对每个 group.id 做哈希求摸运算，从而将负载分散到不同的 __consumer_offset 分区
    >     >
    >     >     > 即每个 consumer group 保存的 offset 都有极大概率分别出现在该 topic 的不同分区上

### (3) 消费者组重平衡(consumer group rebalance)

> rebalance 只对 consumer group 有效

**rebalance**：规定了一个 consumer group 下所有 consumer 如何达成一致来分配订阅 topic 的所有分区

**案例**：一个 consumer group 有 20 个 consumer 实例，该 group 订阅了一个具有 100 个分区的 topic，则 consumer group 平均会为每个 consumer 分配 5 个分区，即每个 consumer 负责读取 5 个分区的数据

## 2、构建 consumer

### (1) consumer 程序实例

```java
package com.example.testspring.kafka.consumer;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;

import java.util.Arrays;
import java.util.Properties;

//构造一个 consumer group 从指定 kafka topic 消费消息
public class ConsumerTest {
    public static void main(String[] args) {
        String topicName = "test-topic";
        String groupID = "test-group";
        Properties properties = new Properties();
        properties.put("bootstrap.servers", "localhost:9092");
        properties.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        properties.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        properties.put("group.id", groupID);
        //下述指定可选
        properties.put("enable.auto.commit", "true");
        properties.put("auto.commit.interval.ms", "1000");
        properties.put("auto.offset.reset", "earliest");
        //创建 consumer 实例
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(properties);
        consumer.subscribe(Arrays.asList(topicName)); //订阅 topic
        try {
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(1000);
                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("offset = %d, key = %s, value = %s%n", 
                                      record.offset(), record.key(), record.value());
                }
            }
        } finally {
            consumer.close();
        }
    }
}
```

构造 consumer 的 6 个步骤：

1. **构造 `java.util.Properties` 对象**，至少指定 `bootstrap.servers、key.deserializer、value.deserializer、group.id` 的值

    > - `bootstrap.servers`：指定 host:port 对，用于创建与 kafka broker 服务器的 socket 连接，可指定多组并用逗号分隔
    >
    >     > 只需指定部分 broker 即可，不需要列出完整的 broker 列表，因为 consumer 可以找到完整的 broker 列表
    >     >
    >     > 注：若 broker 端没有显示配置 listeners 使用 IP 地址，则最好将 bootstrap.servers 配置成主机名
    >
    > - `group.id`：指定 consumer group 名字，能唯一标识一个 consumer group
    >
    > - `key.deserializer`：consumer 从 broker 端获取的任何消息都是字节数组格式，因此需要解序列化
    >
    > - `value.deserializer`：同上

2. 使用上一步创建的 Properties 实例**构造 KafkaConsumer 对象**

    > KafkaConsumer 是 consumer 的主入口：
    >
    > ```java
    > //方式一：
    > KafkaConsumer<String, String> consumer = new KafkaConsumer<>(properties);
    > //方式二：此种方式不需要显示在 properties 中指定 key.deserializer 和 value.deserializer
    > KafkaConsumer consumer = new KafkaConsumer(properties, 
    >                                            new StringDeserializer(), new StringDeserializer());
    > ```

3. 调用 KafkaConsumer.subscribe 方法**订阅 consumer group 的 topic 列表**

    > ```java
    > //订阅多个 topic
    > consumer.subscribe(Arrays.asList(topicName, ...));
    > //正则表达式订阅
    > consumer.subscribe(Pattern.compile("kafka.*"), new NoOpConsumerRebalanceListener());
    > ```

4. 循环调用 KafkaConsumer.poll 方法**获取封装在 ConsumerRecord 的 topic 消息**

    > ```java
    > try {
    >     while (true) {
    >         ConsumerRecords<String, String> records = consumer.poll(1000);
    >         //执行具体的消费逻辑
    >     }
    > } finally {
    >     consumer.close();
    > }
    > ```

5. **处理获取到的 ConsumerRecord 对象** 

6. 关闭 KafkaConsumer

    > - `KafkaConsumer.close()`：关闭 consumer 并最多等待 30 秒
    > - `KafkaConsumer.close(timeout)`：关闭 consumer 并最多等待给定的 timeout 秒

### (2) consumer 脚本命令







### (3) consumer 主要参数











## 3、订阅 topic





## 4、消息轮询





## 5、位移管理





## 6、重平衡





## 7、解序列化





## 8、多线程消费实例





## 9、独立 consumer





## 10、旧版本 consumer









# 五、kafka 设计原理

## 1、broker 端设计架构





## 2、producer 端设计





## 3、consumer 端设计





## 4、实现精确一次处理语义







# 六、管理 kafka 集群

## 1、集群管理





## 2、topic 管理





## 3、topic 动态配置管理





## 4、consumer 相关管理







## 5、topic 分区管理





## 6、kafka 常见脚本工具





## 7、API 方式管理集群











# 七、监控 kafka 集群

## 1、集群健康度检查





## 2、MBean 监控





## 3、broker 端 JMX 监控





## 4、clients 端 JMX 监控





## 5、JVM 监控





## 6、OS 监控





## 7、主流监控框架







# 八、调优kafka集群

## 1、确定调优目标





## 2、集群基础调优





## 3、调优吞吐量





## 4、调优延时





## 5、调优持久性





## 6、调优可用性







# 九、kafka Connect 与 kafka Streams

## 1、kafka connect





## 2、kafka streams