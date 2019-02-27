|             Ⅰ              |                Ⅱ                 |           Ⅲ            |              Ⅳ               |                  Ⅴ                   |               Ⅵ               |            Ⅶ             |            Ⅷ             |                    Ⅸ                     |          Ⅹ           |
| :------------------------: | :------------------------------: | :--------------------: | :--------------------------: | :----------------------------------: | :---------------------------: | :----------------------: | :----------------------: | :--------------------------------------: | :------------------: |
| 算法[:pencil2:](#pencil2-算法) | 操作系统[:computer:](#computer-操作系统) | 网络[:cloud:](#cloud-网络) | 面向对象[:couple:](#couple-面向对象) | 数据库[:floppy_disk:](#floppy_disk-数据库) | Java [:coffee:](#coffee-java) | 系统设计[:bulb:](#bulb-系统设计) | 工具[:hammer:](#hammer-工具) | 编码实践[:speak_no_evil:](#speak_no_evil-编码实践) | 后记[:memo:](#memo-后记) |

### :pencil2: 算法

- [剑指 Offer 题解](./notes/剑指%20offer%20题解.md)

  目录根据原书第二版进行编排，代码和原书有所不同，尽量比原书更简洁

- [Leetcode 题解](./notes/Leetcode%20题解.md)

  对题目做了一个大致分类，并对每种题型的解题思路做了总结

 - [数据结构](./notes/数据结构.md)

   排序、并查集、栈和队列、红黑树、散列表

- [算法导论总结](./notes/算法导论总结.md)

  对于《算法导论》书中大部分算法的总结

### :computer: 操作系统

- [计算机操作系统](./notes/计算机操作系统.md)

  进程管理、内存管理、设备管理、链接

- [Linux](./notes/Linux.md)

  基本实现原理以及基本操作

### :cloud: 网络 

- [计算机网络](./notes/计算机网络.md)

  物理层、链路层、网络层、运输层、应用层

- [计算机网络自顶向下](计算机网络自顶向下.md) 

  关于《计算机网络自顶向下》的总结

- [HTTP](./notes/HTTP.md)

  方法、状态码、Cookie、缓存、连接管理、HTTPs、HTTP 2.0

- [Socket](./notes/Socket.md)

  I/O 模型、I/O 多路复用

- [RabbitMQ实战](./notes/RabbitMQ实战)

  关于 MQ 的基本概念，RabbitMQ 的详解

### :couple: 面向对象

- [设计模式](./notes/设计模式.md)

  实现了 Gof 的 23 种设计模式

- [面向对象思想](./notes/面向对象思想.md)

  三大原则（继承、封装、多态）、类图、设计原则

### :floppy_disk: 数据库 

- [数据库系统原理](./notes/数据库系统原理.md)

  事务、锁、隔离级别、MVCC、间隙锁、范式

- [数据库系统概论总结](./notes/数据库系统概论总结.md) 

  对于王珊版数据库系统概论的总结

- [SQL](./notes/SQL.md)

  SQL 基本语法

- [Leetcode-Database 题解](./notes/Leetcode-Database%20题解.md)

  Leetcode 上数据库题目的解题记录

- [MySQL](./notes/MySQL.md)

  存储引擎、索引、查询优化、切分、复制

- [MySQL进阶](./notes/MySQL进阶.md) 

  关于 MySQL 中 InnoDB 存储引擎的详解

- [Redis](./notes/Redis.md)

  五种数据类型、字典和跳跃表数据结构、使用场景、和 Memcache 的比较、淘汰策略、持久化、文件事件的 Reactor 模式、复制。

- [Redis 设计与实现](./notes/redis设计与实现.md)

  对于《redis 设计与实现》的总结

- [MongoDB实战](./notes/MongoDB实战)

  对于 MongoDB 的一些基本操作，适于入门

- [MongoDB总结](./notes/MongoDB.md)

  对于 MongoDB 的知识点部分总结

- [MyBatis 相关用法介绍](./notes/MyBatis.md) 

  关于 MyBatis 基本用法的简单总结

### :coffee: Java

- [Java 基础](./notes/Java%20基础.md)

  不会涉及很多基本语法介绍，主要是一些实现原理以及关键特性

- [Java 容器](./notes/Java%20容器.md)

  源码分析：ArrayList、Vector、CopyOnWriteArrayList、LinkedList、HashMap、ConcurrentHashMap、LinkedHashMap、WeekHashMap

- [Java 并发](./notes/Java%20并发.md)

  线程使用方式、两种互斥同步方法、线程协作、JUC、线程安全、内存模型、锁优化

- [java并发编程实战](./notes/java并发编程实战.md)

  对于《java并发编程实战》 的总结

- [Java 虚拟机](./notes/Java%20虚拟机.md)

  运行时数据区域、垃圾收集、类加载

- [Java I/O](./notes/Java%20IO.md)

  NIO 的原理以及实例

- [jdbc](./notes/jdbc.md)

  JDBC 的实现及相关拓展

- [java 新特性](./notes/java新特性.md) 

  对于 java 8 行特性的总结

### :coffee:javaweb

- [javaweb基础](./notes/javaweb基础.md)  

  javaweb 的基础知识点

- [webService](./notes/webService.md) 

  关于 webService 的基础部分

### :bulb: 系统设计 

- [系统设计基础](./notes/系统设计基础.md)

  性能、伸缩性、扩展性、可用性、安全性

- [分布式](./notes/分布式.md)

  分布式锁、分布式事务、CAP、BASE、Paxos、Raft

- [集群](./notes/集群.md)

  负载均衡、Session 管理

- [攻击技术](./notes/攻击技术.md)

  XSS、CSRF、SQL 注入、DDoS

- [缓存](./notes/缓存.md)

  缓存特征、缓存位置、缓存问题、数据分布、一致性哈希、LRU、CDN

- [消息队列](./notes/消息队列.md)

  消息处理模型、使用场景、可靠性

### :hammer: 工具 

- [Git](./notes/Git.md)

  一些 Git 的使用和概念

- [Docker](./notes/Docker.md)

  Docker 基本原理

- [Jenkins](./notes/Jenkins.md)

  Jekins 的基本概念与规则

- [正则表达式](./notes/正则表达式.md)

  正则表达式基本语法

- [构建工具](./notes/构建工具.md)

  构建工具的基本概念、主流构建工具介绍

### :speak_no_evil: 编码实践 

- [重构](./notes/重构.md)

  参考 重构 改善既有代码的设计

- [代码可读性](./notes/代码可读性.md)

  参考 编写可读代码的艺术

- [代码风格规范](./notes/代码风格规范.md)

  Google 开源项目的代码风格规范

#### BookList

本仓库参考的书目：[BOOKLIST](./BOOKLIST.md)


