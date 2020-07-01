# 一、简介

推荐阅读： [Apache Thrift - 可伸缩的跨语言服务开发框架](https://www.ibm.com/developerworks/cn/java/j-lo-apachethrift/) 

## 1、架构分层

Thrift软件栈分层从下向上分别为：传输层(Transport Layer)、协议层(Protocol Layer)、处理层(Processor Layer)和服务层(Server Layer)

- **传输层**(Transport Layer)：负责直接从网络中**读取**和**写入**数据，定义了具体的**网络传输协议**，比如：TCP/IP 等
- **协议层**(Protocol Layer)：定义了数据传输格式，负责网络传输数据的**序列化**和**反序列化**，比如：JSON、XML、二进制数据等
- **处理层**(Processor Layer)：由 IDL(接口描述语言)生成，封装了底层网络传输和序列化方式，并**交给用户实现 Handler**
- **服务层**(Server Layer)：整合上述组件，提供具体的网络线程/IO服务模型，形成最终的服务

## 2、数据类型

**(1) 基本类型：**

- `bool`： 布尔值
- `byte`： 8 位有符号整数
- `i16`： 16 位有符号整数
- `i32`： 32 位有符号整数
- `i64`： 64 位有符号整数
- `double`： 64 位浮点数
- `string`： UTF-8 编码的字符串
- `binary`： 二进制串

**(2) 结构体类型：**

- `struct`：定义结构体对象

  ```txt
  struct People{
      1:string name;
      2:i32 age;
      3:string gender;
   }
  ```

**(3) 枚举类型：** 

- `enum`： 定义枚举对象，格式同 Java Enum

  ```
  enum Gender{
      MALE,
      FEMALE
  }
  ```

  > 注意： 枚举类型里没有序号

**(4) 容器类型：**

- `list`： 有序元素列表
- `set`： 无序无重复元素集合
- `map`： 有序的 key/value 集合

**(5) 异常类型：**

- `exception`： 异常类型

  ```
  exception RequestException{
      1: i32 code;
      2: string reason;
   }
  ```

**(6) 服务类型：**

- `service`： 相当于 Java 的 Interface，创建的 service 经过代码生成命令之后就会生成客户端和服务器端的框架代码

  ```java
  service HelloWorldService{
      //service中定义的函数，相当于Java Interface中定义的方法
      string doAction(1:string name,2:i32 age);
  }
  ```

**(7) 类型定义**： 

- `typedef`： 同 C++ 中的 typedef

  ```
  typedef i32 int 
  typedef i64 long
  ```

**(8) 常量定义**： 

- `const`： 定义常量，类似 C++

  ```
  const i32 MIN_GATE=30
  const string MY_WEBSITE="http://facebook.com"
  ```

**(9) 命名空间**： 

- `namespace`： 相当于 java 中的 package，用于组织代码

  ```
  # 格式： namespace 语言名 路径
  namespace java com.test.thift.demo
  ```

**(10) 文件包含**： 

- `include`： 相当于 C/C++ 中的 include，java 中的 import

**(11) 可选与必选**： 

- `required`： 字段必填

- `optional`： 字段可选

  ```
  struct People{
      1:required string name;
      2:optional i32 age;
  }
  ```

## 3、传输协议

- **TBinaryProtocol**：二进制编码格式进行数据传输

- **TCompactProtocol**：高效率的、密集的二进制编码格式进行数据传输

- **TJSONProtocol**： 使用JSON文本的数据编码协议进行数据传输

- **TSimpleJSONProtocol**：只提供JSON只写的协议，适用于通过脚本语言解析
- **TDebugProtocol**： 使用易懂的可读文本格式，以便于 debug

## 4、传输层方式

- **TSocket**：使用**阻塞式I/O**进行传输（最常见模式）
- **TNonblockingTransport**：使用**非阻塞方式**，用于构建**异步客户端**
- **TFramedTransport**：使用**非阻塞方式**，按**块的大小(**frame为单位**)**进行传输，类似于Java中的NIO
- **TFileTransport**： 以文件形式进行传输

## 5、服务端类型

推荐博客： [由浅入深了解Thrift（三）——Thrift server端的几种工作模式分析](https://blog.csdn.net/houjixin/article/details/42779915) 

- TSimpleServer：**单线程**服务器端，使用标准的**阻塞式**I/O

- TThreadPoolServer：**多线程**服务器端，使用标准的**阻塞式**I/O

- TNonblockingServer：**单线程**服务器端，使用**非阻塞式**I/O

- THsHaServer：**半同步半异步**服务器端，基于**非阻塞式**IO读写和**多线程**工作任务处理

- TThreadedSelectorServer：**多线程选择器**服务器端，对THsHaServer在**异步**IO模型上进行增强

# 二、代码生成

thrift 生成语句： `thrift -gen java hello.thrift` 

## 1、核心内部类/接口

原生的 Thrift 框架仅需关注的四个核心内部接口/类：

- **Iface**：**服务端**通过实现 Service.Iface 接口，向客户端提供具体的同步业务逻辑
- **AsyncIface**：**服务端**通过实现 Service.Iface 接口，向客户端提供具体的**异步**业务逻辑
- **Client**：**客户端**通过 Service.Client 的实例对象，以**同步**的方式访问服务端提供的服务方法
- **AsyncClient**：**客户端**通过 Service.AsyncClient 的实例对象，以**异步**的方式访问服务端提供的服务方法

## 2、demo

- **IDL 文件**：

  ```thrift
  namespace java thrift.generated
  
  typedef i16 short
  typedef i32 int
  typedef i64 long
  typedef bool boolean
  typedef string String
  
  struct Person{
      1: optional String username,
      2: optional int age,
      3: optional boolean married
  }
  
  exception DataException{
      1: optional String message,
      2: optional String callStack,
      3: optional String date
  }
  
  service PersonService{
      Person getPersonByUsername(1: required String username) throws (1: DataException dateException),
  
      void savePerson(1: required Person person) throws (1: DataException dataException)
  }
  ```

- **导入依赖并编译**：`thrift -gen java hello.thrift` 

- **编写客户端和服务端代码**

  编写接口实现类，实际开发中放在服务端：

  ```java
  import org.apache.thrift.TException;
  import thrift.generated.DataException;
  import thrift.generated.Person;
  import thrift.generated.PersonService;
  
  public class PersonServiceImpl implements PersonService.Iface{
      @Override
      public Person getPersonByUsername(String username) throws DataException, TException {
          System.out.println("Got client Param:" + username);
  
          Person person = new Person();
          person.setUsername(username);
          person.setAge(32);
          person.setMarried(true);
  
          return person;
      }
  
      @Override
      public void savePerson(Person person) throws DataException, TException {
          System.out.println("Got Client Param: ");
  
          System.out.println(person.getUsername());
          System.out.println(person.getAge());
          System.out.println(person.isMarried());
      }
  }
  ```

  服务器端代码： 

  ```java
  import org.apache.thrift.TProcessorFactory;
  import org.apache.thrift.protocol.TCompactProtocol;
  import org.apache.thrift.server.THsHaServer;
  import org.apache.thrift.server.TServer;
  import org.apache.thrift.transport.TFramedTransport;
  import org.apache.thrift.transport.TNonblockingServerSocket;
  import thrift.generated.PersonService;
  
  public class ThriftServer {
      public static void main(String[] args) throws Exception{
  
          TNonblockingServerSocket socket = new TNonblockingServerSocket(8899);
          THsHaServer.Args arg = new THsHaServer.Args(socket).minWorkerThreads(2).maxWorkerThreads(4);
          //范型就是实现的接收类
          PersonService.Processor<PersonServiceImpl> processor = 
            																		new PersonService.Processor<>(new PersonServiceImpl());
          //表示协议层次（压缩协议）
          arg.protocolFactory(new TCompactProtocol.Factory());
          //表示传输层次
          arg.transportFactory(new TFramedTransport.Factory());
          arg.processorFactory(new TProcessorFactory(processor));
          //半同步半异步的server
          TServer server = new THsHaServer(arg);
  
          System.out.println("Thrift Server started!");
          //死循环，永远不会退出
          server.serve();
      }
  }
  ```

  客户端代码：

  ```java
  import org.apache.thrift.protocol.TCompactProtocol;
  import org.apache.thrift.protocol.TProtocol;
  import org.apache.thrift.transport.TFastFramedTransport;
  import org.apache.thrift.transport.TSocket;
  import org.apache.thrift.transport.TTransport;
  import thrift.generated.Person;
  import thrift.generated.PersonService;
  
  //服务端的协议和客户端的协议要一致
  public class ThriftClient {
      public static void main(String[] args) {
          TTransport tTransport = new TFastFramedTransport(new TSocket("localhost",8899),600);
          TProtocol tProtocol = new TCompactProtocol(tTransport);
          PersonService.Client client = new PersonService.Client(tProtocol);
          try{
              tTransport.open();
  
              Person person = client.getPersonByUsername("张三");
              System.out.println(person.getUsername());
              System.out.println(person.getAge());
              System.out.println(person.isMarried());
              
              System.out.println("............");
  
              Person person2 = new Person();
              person2.setUsername("李四");
              person2.setAge(30);
              person2.setMarried(true);
  
              client.savePerson(person2);
          }catch (Exception ex){
              throw new  RuntimeException(ex.getMessage(),ex);
          }finally {
              tTransport.close();
          }
      }
  }
  ```

  

















