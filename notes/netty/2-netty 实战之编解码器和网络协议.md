# # 第二部分：编解码器

# 十、编解码器框架

## 1、解码器

### (1) ByteToMessageDecoder

`ByteToMessageDecoder`：**将字节解码为消息(或另一个字节序列)**

- `decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out)`：**必须实现**
    - 被调用时将会传入一个包含了传入数据的 ByteBuf，以及一个用来添加解码消息的 List
    - 对这个方法的调用将会重复进行，直到确定没有新的元素被添加到该 List，或该 ByteBuf 中没有更多可读取的字节时为止
    - 然后，若 List 不为空，则它的内容将会被传递给 ChannelPipeline 中的下一个 ChannelInboundHandler
- `decodeLast(ChannelHandlerContext ctx, ByteBuf in, List<Object> out)`：默认实现只是简单地调用 decode() 方法
    - 当 Channel 的状态变为非活动时，这个方法将会被调用一次

---

**案例**：

- 每次从入站 ByteBuf 中读取 4 字节，将其解码为一个 int，然后将它添加到一个 List 中
- 当没有更多的元素可以被添加到该 List 中时，它的内容将会被发送给下一个 ChannelInboundHandler

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_137.png" width="700">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_138.png">

。

可以重写该方法以提供特殊的处理

### (2) ReplayingDecoder

ReplayingDecoder 扩展了 ByteToMessageDecoder 类，使得不必调用 readableBytes() 方法

> 通过使用自定义的 ReplayingDecoderByteBuf，包装传入的 ByteBuf 来实现
>
> `public abstract class ReplayingDecoder<S> extends ByteToMessageDecoder`
>
> - 类型参数 S 指定了用于状态管理的类型，其中 Void 代表不需要状态管理

---

**案例**：从 ByteBuf 中提取的 int 将会被添加到 List 中

- 若没有足够的字节可用，这个 readInt() 方法的实现将会抛出一个 Error
- 当有更多的数据可供读取时，该 decode() 方法将会被再次调用

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_139.png">

> **建议**：若使用 ByteToMessageDecoder 不会引入太多的复杂性，那么请使用它；否则，请使用 ReplayingDecoder

### (3) MessageToMessageDecoder

`MessageToMessageDecoder`：**在两个消息格式之间进行转换**

> `public abstract class MessageToMessageDecoder<I> extends ChannelInboundHandlerAdapter` 
>
> - 类型参数 `I`： 指定了 decode() 方法的输入参数 msg 的类型

---

**案例**：decode() 方法会把 Integer 参数转换为 String 表示

> 解码的 String 将被添加到传出的 List 中，并转发给下一个 ChannelInboundHandler

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_140.png" width="700">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_141.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_142.png">

### (4) TooLongFrameException

`TooLongFrameException`：当解码器在帧超出指定的大小限制时抛出

- 由于 Netty 是异步框架，因此字节在解码前需要内存缓冲，所以防止解码器缓冲大量的数据以至于耗尽可用内存

- 可以设置一个最大字节数的阈值，如果超出该阈值，则抛出一个 TooLongFrameException

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_143.png">

## 2、编码器

### (1) MessageToByteEncoder

`ByteToMessageDecoder`：**将字节编码为消息**

- `encode(ChannelHandlerContext ctx, I msg, ByteBuf out)`：调用时将传入 ByteBuf 的出站消息

    > 该 ByteBuf 随后将会被转发给 ChannelPipeline 中的下一个 ChannelOutboundHandler

---

**案例**：ShortToByteEncoder 接受 Short 类型的实例消息，将其编码为 Short 的原子类型值，并写入 ByteBuf 中

- 随后转发给 ChannelPipeline 中的下一个 ChannelOutboundHandler
- 每个传出的 Short 值都将会占用 ByteBuf 中的 2 字节

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_145.png" width="700">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_144.png">

### (2) MessageToMessageEncoder

`MessageToMessageEncoder`：**将消息编码为消息**

- `encode(ChannelHandlerContext ctx, I msg, List<Object> out)`：通过 write() 方法写入的消息将传递给 encode() 方法，以编码为一个或多个出站消息

    > 随后，这些出站消息将会被转发给 ChannelPipeline 中的下一个 ChannelOutboundHandler

---

**案例**：

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_146.png" width="700">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_147.png">

## 3、编解码器的抽象类

### (1) ByteToMessageCodec

`ByteToMessageCodec` 结合了 `ByteToMessageDecoder` 和 `MessageToByteEncoder`

- `decode(ChannelHandlerContext ctx, ByteBuf in, List<Object>)`：只要有字节可以被消费，这个方法就将会被调用

    > 将入站 ByteBuf 转换为指定的消息格式， 并将其转发给 ChannelPipeline 中的下一个 ChannelInboundHandler

- `decodeLast(ChannelHandlerContext ctx, ByteBuf in, List<Object> out)`：默认实现委托给了 decode() 方法

    > 只会在 Channel 的状态变为非活动时被调用一次，可以重写以实现特殊的处理

- `encode(ChannelHandlerContext ctx, I msg, ByteBuf out)`：这个方法将调用被编码并写入出站 ByteBuf 的消息

### (2) MessageToMessageCodec

- `protected abstract decode(ChannelHandlerContext ctx, INBOUND_IN msg, List<Object> out)`：

    > - **将传入的 INBOUND_IN 类型消息解码为 OUTBOUND_IN 类型消息**
    >
    > - 这些消息将被转发给 ChannelPipeline 中的下一个 ChannelInboundHandler

- `protected abstract encode(ChannelHandlerContext ctx, OUTBOUND_IN msg, List<Object> out)`：

    > - **将 OUTBOUND_IN 类型消息编码为 INBOUND_IN 类型消息**
    >
    > - 然后被转发给 ChannelPipeline 中的下一个 ChannelOutboundHandler

---

**案例**：参数化 MessageToMessageCodec 时将使用 INBOUND_IN 类型的 WebSocketFrame，以及OUTBOUND_IN 类型的 MyWebSocketFrame

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_151.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_152.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_153.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_154.png">

### (3) CombinedChannelDuplexHandler

`CombinedChannelDuplexHandler`：

- 既能避免结合一个解码器和编码器可能会对可重用性造成的影响
- 又不会牺牲将一个解码器和一个编码器作为一个单独的单元部署所带来的便利性

`public class CombinedChannelDuplexHandler<I extends ChannelInboundHandler, O extends ChannelOutboundHandler>`：

- 这个类充当了 ChannelInboundHandler 和 ChannelOutboundHandler(该类的类型参数 I 和 O)的容器
- 通过提供分别继承了解码器类和编码器类的类型，可以实现一个编解码器，而又不必直接扩展抽象的编解码器类

---

- **案例一(解码器)**：一次从 ByteBuf 中提取 2 字节，并将它们作为 char 写入到 List中，其将会被自动装箱为 Character 对象

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_148.png">

- **案例二(编码器)**：将 Character 转换回字节，即通过直接写入ByteBuf，将 char 消息编码到 ByteBuf 中

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_149.png">

- **案例三**：构建一个编解码器

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_150.png">

# 十一、预置 ChannelHandler 和编解码器

## 1、通过 SSL/TLS 保护 Netty 应用程序

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_155.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_156.png">

**建议将 SslHandler 定为 ChannelPipeline 的第一个 ChannelHandler**，确保其他 ChannelHandler 的逻辑应用到数据后，才进行加密

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_157.png">

## 2、构建基于 Netty 的 HTTP/HTTPS 应用程序

### (1) HTTP 解码器、编码器、编解码器

- **HTTP 基于请求/响应模式**：客户端向服务器发送一个 HTTP 请求，然后服务器将会返回一个HTTP 响应

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_158.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_159.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_160.png">

### (2) 聚合 HTTP 消息

Netty 提供了一个聚合器，可以将多个消息部分合并为 FullHttpRequest 或 FullHttpResponse 消息

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_161.png">

### (3) HTTP 压缩

- Netty 为压缩和解压缩提供了 ChannelHandler 实现，同时支持 gzip 和 deflate 编码

    > 当使用 HTTP 时，建议开启压缩功能以尽可能减小传输数据的大小

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_162.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_163.png">

### (4) 使用 HTTPS

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_164.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_165.png">

### (5) WebSocket

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_166.png">

- **WebSocketFrame 可以被归类为数据帧或控制帧**

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_167.png">

---

**案例**：

- 这个类处理协议升级握手，以 及 3 种控制帧(Close、Ping、Pong)
- Text 和 Binary 数据帧将会被传递给下一个 ChannelHandler 进行处理

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_168.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_169.png">

## 3、空闲的连接和超时

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_170.png">

---

**案例**：当使用通常的发送心跳消息到远程节点的方法时：

- 若 60 秒内没有接收或发送任何的数据，将如何得到通知
- 若没有响应，则连接会被关闭

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_171.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_172.png">

- 若连接超过 60 秒没有接收或发送数据，则 IdleStateHandler 将使用一个 IdleStateEvent 事件来调用 fireUserEventTriggered 方法

- HeartbeatHandler 实现了 userEventTriggered 方法，若这个方法检测到 IdleStateEvent 事件，将会发送心跳消息，并且添加一个将在发送操作失败时关闭该连接的 ChannelFutureListener

## 4、解码基于分隔符的协议和基于长度的协议

### (1) 基于分隔符的协议

- **基于分隔符(delimited)的消息协议**：使用定义的字符来标记消息或消息段(帧)的开头或结尾

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_173.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_174.png" width="700">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_175.png">

> 若正在使用除了行尾符之外的分隔符分隔的帧，则可以以类似的方式使用 DelimiterBasedFrameDecoder，只需将特定的分隔符序列指定到其构造函数即可，这些解码器是实现自定义的基于分隔符的协议的工具

---

作为示例，将使用下面的协议规范：

- 传入数据流是一系列的帧，每个帧都由换行符（\n）分隔

- 每个帧都由一系列的元素组成，每个元素都由单个空格字符分隔

- 一个帧的内容代表一个命令，定义为一个命令名称后跟着数目可变的参数

用这个协议的自定义解码器定义以下类：

- `Cmd`：将帧(命令)的内容存储在 ByteBuf 中，一个 ByteBuf 用于名称，另一个用于参数

- `CmdDecoder`：从被重写了的 decode() 方法中获取一行字符串，并从它的内容构建一个 Cmd 实例
- `CmdHandler`：从 CmdDecoder 获取解码的 Cmd 对象，并进行处理
- `CmdHandlerInitializer`：将上述类定义为 ChannelInitializer 嵌套类，其会把这些 ChannelInboundHandler 安装到ChannelPipeline 中

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_176.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_177.png">

### (2) 基于长度的协议

- **基于长度的协议**：通过**将长度编码到帧的头部**来定义帧，而不是使用特殊的分隔符来标记结束

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_178.png">

<figure>
  <img src="/Users/yinren/allText/learnNote/pics/netty/netty_179.png" width="460">
  <img src="/Users/yinren/allText/learnNote/pics/netty/netty_180.png" width="480">
</figure>

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_181.png">

## 5、写大型数据

- 写大型数据时，需要**考虑远程节点的慢速连接**，这种情况**会导致内存释放的延迟** 

---

- **FileRegion 接口的实现**：通过支持零拷贝文件传输的 Channel 来发送的文件区域

**案例**：通过从 FileInputStream 创建一个 DefaultFileRegion，并将其写入 Channel，从而**利用零拷贝特性来传输文件内容**

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_182.png">

> 这个示例**只适用于文件内容的直接传输**，不包括应用程序对数据的任何处理

---

- **若要将数据从文件系统复制到用户内存中**，使用 ChunkedWriteHandler，支持异步写大型数据流，而又不会导致大量的内存消耗

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_183.png">

**案例**：下面代码说明了 ChunkedStream 的用法

- 所示的类使用了一个 File 以及一个 SslContext 进行实例化
- 当 initChannel() 方法被调用时，将使用所示的 ChannelHandler 链初始化该 Channel

- 当 Channel 的状态变为活动时，WriteStreamHandler 将会逐块地把来自文件中的数据作为 ChunkedStream 写入

    > **逐块输入**：要使用自定义的 ChunkedInput 实现，请在 ChannelPipeline 中安装一个 ChunkedWriteHandler

- 数据在传输之前将会由 SslHandler 加密

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_184.png">

## 6、序列化数据

### (1) JDK 序列化

若应用程序必须要和使用了 ObjectOutputStream 和 ObjectInputStream 的远程节点交互，则优先选择 JDK序列化：

- JDK 提供了 ObjectOutputStream 和 ObjectInputStream，用于通过网络对 POJO 的基本数据类型进行序列化和反序列化

- 该 API 并不复杂，而且可以被应用于任何实现了 java.io.Serializable 接口的对象

---

下表列出了 Netty 提供的用于和 JDK 进行互操作的序列化类：

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_185.png">

### (2) JBoss Marshalling 序列化

**JBoss Marshalling 序列化**：

- 修复了 JDK 序列化 API 中的问题，同时保留了与 java.io.Serializable 及其相关类的兼容性，并添加了新的可调优参数以及额外的特性
- 所有的这些都可以通过工厂配置(如：外部序列化器、类/实例查找表、类解析、对象替换等)实现可插拔

---

Netty 提供了两组解码器/编码器对为 Boss Marshalling 提供支持：

- 第一组兼容只使用 JDK 序列化的远程节点
- 第二组提供了最大的性能，适用于和使用 JBoss Marshalling 的远程节点一起使用

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_186.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_187.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_188.png">

### (3) Protocol Buffers 序列化

- **Protocol Buffers**：以一种紧凑而高效的方式对结构化的数据进行编码和解码。具有许多的编程语言绑定，**适合跨语言项目**

---

Netty 为支持 protobuf 所提供的 ChannelHandler 实现：

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_189.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_190.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_191.png">

# # 第三部分：网络协议

# 十二、WebSocket

## 1、WebSocket 应用

**应用程序约定**：

- 如果被请求的 URL 以 /ws 结尾，则 把该协议升级为 WebSocket

    > 在连接升级完成后，所有数据都将会使用 WebSocket 进行传输

- 否则，服务器将使用基本的 HTTP/S

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_192.png" width="800">

### (1) 处理 HTTP 请求

- `HttpRequestHandler`：实现处理 HTTP 请求的组件，提供用于访问聊天室并显示由连接的客户端发送消息的网页

    > 代表聊天服务器的第一个部分：管理纯粹的 HTTP 请求和响应

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_193.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_194.png">

**步骤**：

1. 若 HTTP 请求**指向了地址为 `/ws` 的 URI**，则 HttpRequestHandler 将调用 FullHttpRequest 的 `retain()` 方法，并通过调用 `fireChannelRead(msg)` 方法将它转发给下一个 ChannelInboundHandler

2. 若客户端发送了 HTTP 1.1 的 HTTP 头信息 `Expect: 100-continue`，则 HttpRequestHandler 将发送一个 100 Continue 响应

3. 在该 HTTP 头信息被设置之后，HttpRequestHandler 将写回一个 HttpResponse 给客户端

4. 若不需加密和压缩，可以通过将 index.html 内容存储到 DefaultFileRegion 中来达到最佳效率，利用零拷贝特性进行内容传输

5. HttpRequestHandler 将写一个 LastHttpContent 来标记响应的结束

6. 若没有请求 `keep-alive` ，则 HttpRequestHandler 会添加 ChannelFutureListener 到最后一次写出动作的 ChannelFuture，并关闭该连接

    > 在这里，将调用 writeAndFlush() 方法以冲刷所有之前写入的消息

### (2) 处理 WebSocket 帧

**WEBSOCKET 帧**：WebSocket 以帧的方式传输数据，每一帧代表消息的一部分，一个完整的消息可能会包含许多帧

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_195.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_196.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_197.png">

**步骤**：

1. 当和新客户端的 WebSocket 握手成功完成后，会把通知消息写到 ChannelGroup 中的所有 Channel 来通知所有已经连接的客户端
2. 然后将把这个新 Channel 加入到该 ChannelGroup 中
3. 若接收到 TextWebSocketFrame 消息，TextWebSocketFrameHandler 将调用 TextWebSocketFrame 消息上的 retain()方法，并使用 writeAndFlush() 方法来将它传输给 ChannelGroup，以便所有已经连接的 WebSocket Channel 都将接收到

### (3) 初始化 ChannelPipeline

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_199.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_198.png">

---

- **WebSocket 协议升级之前**的 ChannelPipeline 状态：代表刚刚被 ChatServerInitializer 初始化之后的 ChannelPipeline

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_200.png">

- **WebSocket 协议升级完成之后**，WebSocketServerProtocolHandler 会：

    - 把 HttpRequestDecoder 替换为 WebSocketFrameDecoder
    - 把 HttpResponseEncoder 替换为 WebSocketFrameEncoder
    - 为了性能最大化，将移除任何不再被 WebSocket 连接所需要的 ChannelHandler

    > 注意：Netty 支持 4 个版本的 WebSocket 协议，Netty 会根据客户端(指浏览器)支持的版本，选择合适的WebSocketFrameDecoder 和 WebSocketFrameEncoder

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_201.png">

### (4) 引导

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_202.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_203.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_204.png">

## 2、WebSocket 加密

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_205.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_206.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_207.png">

# 十三、使用 UDP 广播事件

## 1、UDP 广播

- **单播传输**：发送消息给一个由唯一的地址所标识的单一的网络目的地

    > 面向连接的协议和无连接协议都支持这种模式

- UDP 提供了向多个接收者发送消息的传输模式：
    - **多播**：传输到一个预定义的主机组
    - **广播**：传输到网络(或子网)上的所有主机

## 2、UDP 示例

**案例**：将打开一个文件，随后将会通过 UDP 把每一行都作为一个消息广播到一个指定的端口

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_208.png" width="700">

### (1) 消息 POJO：LogEvent

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_209.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_210.png">

### (2) 编写广播者

- Netty 的 DatagramPacket 是一个简单的消息容器，DatagramChannel 实现用它来和远程节点通信，包含接收者(和可选的发送者)的地址以及消息的有效负载本身

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_211.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_212.png">

---

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_213.png" width="700">

- 要被传输的数据都被封装在 LogEvent 消息中
- LogEventBroadcaster 将把它们写入到 Channel 中，并通过 ChannelPipeline 发送，同时将它们转换(编码)为 DatagramPacket 消息
- 最后，所有数据都将通过 UDP 被广播，并由远程节点(监视器)所捕获

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_214.png">

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_215.png">

### (3) 编写监视器

LogEventMonitor 的功能：

- 接收由 LogEventBroadcaster 广播的 UDP DatagramPacket

- 将它们解码为 LogEvent 消息

- 将 LogEvent 消息写出到 System.out

<img src="/Users/yinren/allText/learnNote/pics/netty/netty_216.png">

---

- **LogEventDecoder 负责将传入的 DatagramPacket解码为LogEvent 消息**

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_217.png">

- **处理 LogEvent 消息**

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_218.png">

- **将 LogEventDecoder 和 LogEventHandler 安装到 ChannelPipeline 中**

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_219.png">

    <img src="/Users/yinren/allText/learnNote/pics/netty/netty_220.png">