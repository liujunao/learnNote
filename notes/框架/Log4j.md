# 一、Log4J 组件
- `Logger`：日志记录器，负责收集处理日志记录     （如何处理日志）
- `Level`： 日志级别，规定输出的日志级别（输出什么级别的日志）
- `Appender`：日志输出目的地，负责日志的输出  （输出到什么 地方）
- `Layout`：日志格式化，负责对输出的日志格式化（以什么形式展现）

# 二、基本使用方法
## 1. 定义配置文件
- **配置根 `Logger`**： 

  ```properties
  log4j.rootLogger = [level], appenderName, appenderName, ...
  ```
  > - `level` 是日志记录的优先级，分为 `OFF、FATAL、ERROR、WARN、INFO、DEBUG、ALL`
  >   > 建议使用 `ERROR、WARN、INFO、DEBUG`(优先级从高到低)
  >   > 比如： 定 义 INFO 级别，则程序中 DEBUG 级别的日志信息将不被打印出来
  > - `appenderName` 指日志信息输出到哪个地方，可以同时指定多个输出目的地

- **配置日志信息输出目的地 `Appender`**： 

  ```properties
  log4j.appender.appenderName = fully.qualified.name.of.appender.class  
  log4j.appender.appenderName.option1 = value1  
  …  
  log4j.appender.appenderName.optionN = valueN
  ```
  > Log4j提供的 appender 有以下几种：
  > ```properties
  > org.apache.log4j.ConsoleAppender（控制台）
  > org.apache.log4j.FileAppender（文件）
  > org.apache.log4j.DailyRollingFileAppender（每天产生一个日志文件）
  > org.apache.log4j.RollingFileAppender（文件达到指定大小时，产生新文件）
  > org.apache.log4j.WriterAppender（将日志信息以流格式发送到指定地方）
  > ```

- **Append 追加属性**： 

  - `layout`： 格式化日志信息

    > ```properties
    > log4j.appender.appenderName.layout = fully.qualified.name.of.layout.class  
    > log4j.appender.appenderName.layout.option1 = value1  
    > …  
    > log4j.appender.appenderName.layout.optionN = valueN
    > ```
    >
    > Log4j 提供的 layout 有以下几种：
    >
    > ```properties
    > org.apache.log4j.HTMLLayout（以 HTML 表格形式布局）
    > org.apache.log4j.PatternLayout（可以灵活地指定布局模式）
    > org.apache.log4j.SimpleLayout（包含日志信息的级别和信息字符串）
    > org.apache.log4j.TTCCLayout（包含日志产生的时间、线程、类别等等信息）
    > ```
    >
    > Log4J 格式化日志信息：
    >
    > ```
    > %m 输出代码中指定的消息
    > %p 输出优先级，即DEBUG，INFO，WARN，ERROR，FATAL  
    > %r 输出自应用启动到输出该log信息耗费的毫秒数  
    > %c 输出所属的类目，通常就是所在类的全名  
    > %t 输出产生该日志事件的线程名  
    > %n 输出一个回车换行符，Windows平台为“rn”，Unix平台为“n”  
    > %d 输出日志时间点的日期或时间，默认格式为ISO8601，也可指定格式，比如：%d{yyy MMM dd HH:mm:ss,SSS}，输出类似：2002年10月18日 22：10：28，921  
    > %l 输出日志事件的发生位置，包括类目名、发生的线程，以及在代码中的行数。举例：Testlog4.main(TestLog4.java:10)
    > ```

  - `target`： 输出目标，如： 控制台、文件或其他项目

  - `level`： 级别过滤日志信息

  - `threshold`： 阈值级别，会忽略低于阈值级别的日志信息

  - `filter`： 决定日志记录请求由特定 Append 处理还是忽略

## 2. 代码中使用

- **得到记录器**： 

  ```java
  //通过指定的名字获得记录器: name 一般为类名，如： Example.class.getName()
  public static Logger getLogger( String name)
  ```

- **读取配置文件**：

  ```java
  BasicConfigurator.configure()： 自动快速地使用缺省 Log4j 环境
  PropertyConfigurator.configure(String configFilename)：读取 properties 配置文件
  DOMConfigurator.configure(String filename) ：读取 XML 配置文件
  ```

- 插入记录信息(格式化日志信息)： 

  ```java
  Logger.debug(Object message);  
  Logger.info(Object message);  
  Logger.warn(Object message);  
  Logger.error(Object message);
  ```

## 3. 日志级别

- `off`： 最高等级，用于关闭所有日志记录
- `fatal`： 指出每个严重的错误事件将会导致应用程序的退出
- `error`： 指出虽然发生错误事件，但仍然不影响系统的继续运行
- `warn`： 表明会出现潜在的错误情形
- `info`： 一般和在粗粒度级别上，强调应用程序的运行全程
- `debug`： 一般用于细粒度级别上，对调试应用程序非常有帮助
- `all`： 最低等级，用于打开所有日志记录





























