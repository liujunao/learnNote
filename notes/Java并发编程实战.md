#一、简介

## 1. 并发的历史

- 线程是控制和利用多处理器系统计算能力的最简单方式
- 下列因素推动了多程序共同运行：
  - 充分利用计算机资源
  - 保证多用户分时公平性
  - 有时候多程序解决问题更加方便
- 早期分时共享系统中，每个进程都是一个冯诺依曼机
- 寻找顺序和异步执行的平衡点，是让程序高效的关键。
- 线程的出现，分解了进程，而且更适应在多处理器系统上运行
- 同一进程中的线程，实现了良好的数据共享。然而多线程之间共享数据，会带来并发问题

## 2. 线程的优点

- 第一个优点是使用多核处理器
- 第二个优点是简化组件开发
- 第三个优点是，服务器为来自客户端的每一个连接都分配一个线程，并使用同步 IO ，是一种高效的方式
- 第四个优点是改善图形化软件的用户体验，防止“冻结”现象产生

##3. 线程的风险

- 第一个风险是多线程下执行顺序无法确定，会产生意外情况。使用 java 的同步机制可改善这个问题
- 第二个风险是死锁等程序无法继续执行的情况
- 第三个风险是多线程中，线程调度和同步机制会增加资源开销

# 

# # 基础

# 二、线程安全性

当多个线程同时访问一个可变的状态变量时，要使用合适的同步 ，要确保线程安全，则：

- 不在线程之间共享该状态变量
- 将状态变量修改为不可变的变量
- 在访问状态变量时使用同步

程序状态的封装性越好，就越容易实现程序的线程安全性

- 当设计线程安全的类时，良好的面向对象技术，不可修改性，以及明晰的不变性规范都能起到一定的帮助作用

## 1. 什么是线程安全性

- 在多个线程访问的时候，程序还能“**正确**”，那就是线程安全的

- **无状态的对象一定是线程安全的**

  > 无状态：
  >
  > - 既**不包含任何域**
  > - 也**不包含任何对其他类中域的引用**

## 2. 原子性

- **原子操作：** 对于访问同一个状态的所有操作（包括该操作本身）来说，这个操作是一个以原子方式执行的操作

- **竞态条件：** 当某个计算的正确性取决于多个线程的交替执行时序时，就会发生竞态条件

  > 由于不恰当的执行时序而出现不正确的结果

- 基于一种可能失效的观察结果来做出判断或者执行某个计算，这种类型的竞态条件称为**先检查后执行**

  > 例子：**延迟初始化：** 将对象的初始化操作推迟到实际被使用时才进行，同时要确保只被初始化一次
  >
  > ```java
  > //延迟初始化中的竞态条件(不要这么做)
  > @NotThreadSafe
  > public class LazyInitRace {
  >     private ExpensiveObject instance = null;
  >     
  >     public ExpensiveObject getInstance(){
  >         if (instance == null){
  >             instance = new ExpensiveObject();
  >         }
  >         return instance;
  >     }
  > }
  > ```

- **复合操作：** 包含了一组以原子方式执行的操作以确保线程安全性

  > **如：** 通过用 **AtomicLong** 来代替 **long** 类型的计数器，能够确保对计数器状态的访问操作都是原子操作
  >
  > ```java
  > //concurrent.atomic 包实现原子操作
  > class B {
  >     private AtomicInteger count = new AtomicInteger(0);
  > 
  >     public void inc(){
  >         count.incrementAndGet();
  >     }
  > }
  > ```

## 3. 加锁机制

- 要保持状态的一致性，就需要在单个原子操作中更新所有相关的状态变量
- 未保证原子性条件下，不建议对结果进行缓存

### 1. 内置锁

- java 提供了一种内置的锁机制来支持原子性：**同步代码块** 

  - 作为锁的对象引用
  - 作为由这个锁保护的代码块

  ```java
  synchronized(lock){
    
  }
  ```

- 每个java对象都可以用作一个实现同步的锁，这些锁称为**内置锁** 或 **监视器锁** 

### 2. 重入

- 若某个线程试图获得一个已由其自己持有的锁，则这个请求会成功
- **重入** 意味着获取锁的操作的粒度是“线程“，而不是调用
- **重入** 的一种实现方法是：为每个锁关联一个获取计数器值和所有者线程
- **重入** 提升了加锁行为的封装性

## 4. 用锁来保护状态

- 由于锁能够使其保护代码的路径以**串行形式**来访问，因此可以通过锁来构造一些协议以实现对共享状态的独占访问

  > **串行访问：** 意味着多个线程依次以独占的方式访问对象，而不是并发的访问

- 对于可能被多个线程同时访问的可变状态变量，在访问它时都需要持有同一个锁，在这种情况下，我们称为**状态变量是由这个锁保护的** 

- 每个共享的和可变的变量都应该只由一个锁来保护，从而使维护人员知道是哪个锁

- 对于每个包含多个变量的不可变性条件，其中涉及的所有变量都需要由同一个锁来保护

- **加锁约定**： 将所有的可变状态都封装在对象内部，并通过对象的内置锁对所有访问可变状态的代码路径进行同步，使得在该对象上不会发生并发访问

# 三、对象的共享

## 1. 可见性

- **指令重排序：** 编译器或运行时环境为了优化程序性能而采取的对指令进行重新排序执行的一种手段

- **最低安全性**： 当线程在没有同步的情况下读取变量，可能会得到一个失效值

  > 最低安全性适用于大多数变量，但不适用32位操作系统的**非 volatile 类型的 64 位数值变量**的处理

- 加锁的含义不仅仅局限于**互斥行为**，还包括**内存可见性**

  > 为了确保所有线程都能看到共享变量的最新值，所有执行读操作或写操作的线程都必须在同一个锁上同步

- `volatile` 关键字：

  > **加锁机制既可以确保可见性又可以确保原子性,但 volatile 变量只能确保可见性**

  - 确保它们自身状态的可见性
  - 确保它们所引用对象状态的可见性
  - 标识一些重要的程序生命周期事件的发生

- `volatile` 的使用条件：
  - 对变量的写入操作不依赖变量的当前值，或者能确保只有单个线程更新变量的值
  - 该变量不会与其他状态变量一起纳入不变性条件中
  - 在访问变量时不需要加锁

## 2. 发布与逸出

- **发布：** 使对象能够在当前作用域之外的代码中使用

  > 当发布一个对象时，在该对象的非私有域中引用的所有对象同样会被发布

- **逸出：** 当某个不应该发布的对象被发布时，就称为逸出

- **外部方法：** 指行为并不完全由该类来规定的方法

  > 包括：
  >
  > - 其他类中定义的方法
  > - 该类中可以被改写的方法

- **封装**： 能够使得对程序的正确性进行分析变得可能，并使得无意中破坏设计约束条件变得更难

- 内部类的隐式溢出： 

  ```java
  //隐式的使 this 引用逸出：
  public class ThisEscape {
      public ThisEscape(EventSource eventSource){
          eventSource.registerListener(
                  new EventListener(){
                      public void onEvent(Event event){
                          doSomething(event);
                      }
                  }
          );
      }
  }
  
  //使用工厂方法来防止 this 引用在构造过程中逸出
  public class SafeListener {
      private final EventListener listener;
  
      private SafeListener(){
          listener = new EventListener() {
              public void onEvent(Event event){
                  doSomething(event);
              }
          };
      }
  
      public static SafeListener newInstance(EventSource source){
          SafeListener safeListener = new SafeListener();
          source.registerListner(safeListener.listener);
          return safeListener;
      }
  }
  ```

## 3. 线程封闭

- **线程封闭**： 避免使用同步的方式就是**不共享数据**，如果仅在线程内访问数据，就不需要同步的技术

  > 如： JDBC 中的 Connection 对象

- `Ad-hoc` 线程封闭： 维护线程封闭性的职责完全由程序实现来承担

  > 对线程封闭对象的引用通常保存在公有变量中

- **栈封闭**： 线程封闭的一种特例，只能通过局部变量才能访问对象，局部变量都会被拷贝一份到线程栈中

  > - 局部变量是不被多个线程所共享的，也就不会出现并发问题，即： 如果在线程内部上下文中使用非线程安全的对象，那么该对象任然是线程安全的
  > - 能用局部变量就别用全局的变量，全局变量容易引起并发问题

- `ThreadLocal` 类： 能使线程中某个值与保存值的对象关联起来

  > - ThreadLocal 类提供了 get 和 set 等访问接口或者方法，这些方法为每个使用该变量的线程都存在一份独立的副本，因此 get 总是返回当前执行线程在调用 set 设置的最新值
  > - ThreadLocal 对象常用于防止对可变对象的单实例变量或全局变量进行共享

```java
//基本类型的局部变量与引用变量的线程封闭性
public int loadTheArk(Collection<Animal> candidates){
    SortedSet<Animal> animals;
    int numParis = 0;
    Animal candidate = null;
    // animals 被封闭在方法中，不要使它们逸出
    animals = new TreeSet<Animal>(new SpeciesGenderComparator());
    animals.addAll(candidates);
    for (Animal animal : animals){
        if (candidate == null || !candidate.isPotentialMate(animal)){
            candidate = animal;
        }else {
            ark.load(new AnimalPair(candidate, animal));
            ++numParis;
            candidate = null;
        }
    }
    return numParis;
}

//ThreadLocal 类示例
public class ConnectionManager {    
    private static ThreadLocal<Connection> connectionHolder = 
        new ThreadLocal<Connection>() {  
        	public Connection initialValue() {  
            	Connection conn = null;  
            	try {  
                	conn = DriverManager.getConnection
                        ("jdbc:mysql://localhost:3306/test", 
                         "username", "password"); 
            	} catch (SQLException e) {  
                	e.printStackTrace();  
            	}  
            	return conn;  
        	}  
    	};  

    public static Connection getConnection() {  
        return connectionHolder.get();  
    }  

    public static void setConnection(Connection conn) {  
        connectionHolder.set(conn);  
    }  
}
```

## 4. 不变性

- 不变对象一定是线程安全的
- 不变对象满足的条件：
  - 对象创建以后其状态就不能修改
  - 对象的所有域都是 final 类型
  - 对象是正确创建的（在对象的创建期间，this 引用没有逸出）
- **fianl 关键字**用于构造不可变对象，final 域能确保初始化过程的安全性，从而可以不受限制的访问不可变对象，并在共享这些对象时无须同步
- 好的编程习惯：
  - 除非需要更高的可见性，否则应将所有的域都声明为私有域
  - 除非需要某个域是可变的，否则应将其声明为 final 域

## 5. 安全发布

- **维持初始化安全性**： 
  - 状态不可修改
  - 所有域都是 final 类型
  - 正确的构造过程
- 要安全的发布一个对象，对象的引用和对象的状态必须同时对其他线程可见
  - 在静态初始化函数中初始化一个对象引用
  - 将一个对象引用保存在 volatile 类型的域或者是 AtomicReference 对象中
  - 将对象的引用保存到某个正确构造对象的final类型的域中
  - 将对象的引用保存到一个由锁保护的域

- **线程安全容器**提供的安全发布保证：

  - 通过将一个键或者值放入 `Hashtable、synchronizedMap、ConcurrentMap` 中，可以安全将它发布给任何从这些容器中访问它的线程
  - 通过将某个元素放到 `Vector、CopyOnWriteArrayList、CopyOnWriteArraySet、synchroizedList`，可以将该元素安全的发布到任何从这些容器中访问该元素的线程

  - 通过将元素放到 `BlockingQueue、ConcrrentLinkedQueue` 中，可以将该元素安全的发布到任何从这些访问队列中访问该元素的线程

- 要**发布一个静态构造的对象**，最简单和最安全的方式是使用**静态初始化器**：`public static Holder = new Holder();`

  > - 静态初始化器由 JVM 在类的初始化阶段执行，由于 JVM 内部存在同步机制，所以这种方式初始化对象都可以被安全的发布
  > - 对于可变对象，安全的发布之时确保在发布当时状态的可见性，而在随后的每次对象的访问时，同样需要使用同步来确保修改操作的可见性

- **事实不可变对象**： 对象从技术上来看是可变的，但其状态在发布后不会再改变

  > - 当对象的引用对所有访问该对象的线程可见，对象发布时的状态对于所有线程也将是可见的
  > - 在没有额外的同步情况下，任何线程都可以安全的使用被安全发布的事实不可变对象

- **可变对象**： 不仅发布对象时使用同步，且在每次对象访问时同样需要使用同步来确保后续修改操作的可见性

- **对象发布需求取决于其可变性**：
  - **不可变对象可以通过任意机制来发布**
  - **事实不可变对象必须通过安全方式来发布**
  - **可变对象必须通过安全方式来发布，且必须是线程安全或使用锁**

- 并发程序中使用和共享对象时的策略：

  - **线程封闭**： 线程封闭的对象只能由一个线程拥有，对象被封闭在该线程中，并且只能由这个线程修改

  - **只读共享**： 共享只读对象可以由多个线程并发访问，但不能修改它，包括不可变对象和事实不可变对象

  - **线程安全共享**： 线程安全的对象在其内部实现同步，因此多个线程可以通过对象的公有接口来进行访问而不需要进一步的同步

  - **保护对象**： 被保护的对象只能通过持有特定的锁来访问

    > 保护对象包括封装在其他线程安全对象中的对象，以及已发布的并且由某个特定锁保护的对象

# 四、对象的组合

## 1. 设计线程安全的类

- **封装技术**可以使得在不对整个程序进行分析的情况下就可以判断一个类是否是线程安全的

- 在设计线程安全类的过程中，需要包含以下三个基本要素：

  - **找出构成对象状态的所有变量**

  - **找出约束状态变量的不变性条件**

  - **建立对象状态的并发访问管理策略**

- **对象的状态：** 
  - 如果对象中所得的域都是基本类型的变量，那么这些域将构成对象的全部状态
  - 如果在对象的域中引用了其他对象，那么该对象的状态包括被引用对象的域

- **同步策略**： 定义如何在不违背对象不变条件或后验条件的情况下对其状态的访问操作进行协同

  > - 对象的==不变性条件与后验条件==可确保线程安全性 
  > - ==原子性与封装性==可用于状态变量的有效值或状态转换的各种约束条件

- **依赖状态操作**： 如果在某个操作中包含有基于状态的先验条件

  > 例如： 不能从空队列中删除一个元素，在删除元素之前，必须得先判断该队列非空

- **所有权**： 属于类设计中的一个要素，所有权意味着控制权，如果发布了某个可变对象的引用，则意味着共享控制权，在定义哪些变量构成对象的状态时，只考虑对象拥有的数据

  > 所有权与封装性是相互关联的：
  >
  > - 对象封装它拥有的状态 
  > - 对它封装的状态拥有所有权

## 2. 实例封闭

- **实例封闭机制**： 将数据封装在对象内部，可以将数据的访问限制在对象的方法上，从而更容易确保线程在访问数据时总能持有正确的锁

  > - 封装简化了线程安全类的实现过程，它提供了一种实例封闭机制
  >
  > - 封装机制更易于构造线程安全的类，因为当封装类的状态时，在分析类的线程安全性时就无须检查整个程序

- **被封闭的作用域**：
  - **一个实例中**：作为一个私有成员
  - **某个作用域中**：作为局部变量
  - **线程里**：将对象从一个方法传递到另一个方法

- **装饰器**： 将容器类封装在一个同步的包装器对象中，而包装器能将接口中的每个方法都实现为同步方法，并将调用请求转发到底层的容器对象上

  > 只要包装器对象拥有对底层容器对象的唯一引用，那么它就是线程安全的

- **Java 监视器模式**： 遵循Java监视器模式的对象会把对象的所有可变状态都封装起来，并由对象内置锁来保护

  > - 其可变状态都是私有的，并且涉企到该状态的方法都有一个内置锁来保护
  > - Java 的内置锁也称为**监视器锁**或者**监视器**

  ```java
  //基于监视器模式的车辆追踪
  @ThreadSafe
  public class MonitorVehicleTracker {
      @GuardedBy("this")
      private final Map<String, MutablePoint> locations;
  
      public MonitorVehicleTracker(Map<String,MutablePoint> locations){
          this.locations = locations;
      }
  
      public synchronized Map<String,MutablePoint> getLocations(){
          return deepCopy(locations);
      }
  
      public synchronized void setLocations(String id, int x, int y){
          MutablePoint mutablePoint = locations.get(id);
          if (mutablePoint == null){
              throw new IllegalArgumentException("No such ID: " + id);
          }
          mutablePoint.x = x;
          mutablePoint.y = y;
      }
  
      private static Map<String,MutablePoint> deepCopy(
          	Map<String, MutablePoint> locations) {
          Map<String,MutablePoint> result = new HashMap<>();
          for (String id : locations.keySet()){
              result.put(id, new MutablePoint(locations.get(id)));
          }
          return Collections.unmodifiableMap(result);
      }
  }
  ```

## 3. 线程安全的委托

- 当从头开始创建一个类，或者将多个非线程安全的类组合为一个类时，java 监听器模式是非常有用的

- **独立的状态变量**： 一个对象有多个状态变量，即多个域，并且每个状态变量没有耦合性，或者说不相互影响

  > 当一个类是由多个独立且线程安全的状态变量组成，并且在所有的操作中都不包含无效的状态转换，那么可以将线程安全委托给底层的状态变量：
  >
  > - 只要每个独立的状态变量是线程安全的，那么整个类就是线程安全的
  >
  > - 假如类的多个状态变量是相互影响的，即使每个状态变量都是线程安全的，那么整个类也有可能不是线程安全的

- **委托失效**：

  ```java
  // NumberRange 类并不足以保护它的不变性条件，不是线程安全的
  public class NumberRange {
      // 不变性条件: lower <= upper
      private final AtomicInteger lower = new AtomicInteger(0);
      private final AtomicInteger upper = new AtomicInteger(0);
  
      public void setLower(int i){
      //注意： 不安全的"先检查后执行"
          if(i > upper.get()){
              throw new IllegalArgumentException("不能设置lower > upper");
          }
          lower.set(i);
      }
  
      public void setUpper(int i){
      //注意： 不安全的"先检查后执行"
          if(i < lower.get()){
              throw new IllegalArgumentException("不能设置upper < lower");
          }
          upper.set(i);
      }   
  }
  ```

- **volatile变量规则：** 仅当一个变量参与到包含其他变量的不变性条件时，才可以声明为 volatile 类型

- **发布底层的状态变量**： 如果一个状态变量是线程安全的，并且没有任何不变性条件来约束它的值，在变量的操作上也不存在任何不允许的状态转换，那么就可以安全的发布这个变量

```java
//线程安全且可变的 Point 类
@ThreadSafe
public class SafePoint { 
    @GuardedBy("this") private int x, y;

    public SafePoint(int x, int y) {
        this.x = x;
        this.y = y;
    }

    private SafePoint(int[] a){
        this(a[0], a[1]);
    }

    public SafePoint(SafePoint safePoint){
        this(safePoint.get());
    }

    public synchronized int[] get(){
        return new int[] {x, y};
    }

    public synchronized void set(int x, int y){
        this.x = x;
        this.y = y;
    }
}

//安全发布底层状态的车辆追踪器
@ThreadSafe
public class PublishingVehicleTracker {
    private final Map<String, SafePoint> locations;
    private final Map<String, SafePoint> unmodifiableMap;

    public PublishingVehicleTracker(Map<String, SafePoint> locations) {
        this.locations = new ConcurrentHashMap<>(locations);
        this.unmodifiableMap = Collections.unmodifiableMap(this.locations);
    }

    public Map<String, SafePoint> getLocations() {
        return unmodifiableMap;
    }

    public SafePoint getLocations(String id){
        return locations.get(id);
    }

    public void setLocations(String id, int x, int y){
        if (!locations.containsKey(id)){
            throw new IllegalArgumentException("invalid vehicle name: " + id);
        }
        locations.get(id).set(x, y);
    }
}
```

## 4. 在现有的线程安全类中添加功能

添加新功能的方法：

- **修改原始类**
- **扩展类**： 可使用继承来实现

- **客户端加锁机制**： 对于使用某个对象 X 的客户端代码，使用 X 本身保护其状态的锁来保护这段客户端代码

  ```java
  //通过客户端加锁来实现“若没有则添加”
  public class ListHelper<E> {
      public List<E> list = 
          (List<E>) Collections.synchronizedCollection(new ArrayList<E>());
  
      public boolean putIfAbsent(E x){
          synchronized(list){
              boolean absent = !list.contains(x);
              if(absent){
                  list.add(x);
              }
              return absent;
          }
      }
  }
  ```

  - 通过添加一个原子操作来扩展类是脆弱的，因为它将类的加锁代码分布到多个类中
  - 然而，客户端加锁却更加脆弱，因为它将类 C 的加锁代码当道与 C 完全无关的其他类中

- **组合**： 

  ```java
  //通过组合实现“若没有则添加”
  @ThreadSafe
  public class ImprovedList<T> implements List<T> {
  
      private final List<T> list;
      public ImprovedList(List<T> list){
          this.list = list;
      }
  
      public synchronized boolean putIfAbsent(E x){
          boolean absent = !list.contains(x);
          if(absent){
              list.add(x);
          }
          return absent;
  
      }
      public synchronized boolean add(T arg0) {
          return list.add(arg0);
      }
      // ... 按照类似的方式委托list其他方法
  }
  ```

# 五、基础构建模块

## 1. 同步容器类

- 同步容器类包括： `Vector, Hashtable`

  > 这些类实现的方法是：将它们的状态封装起来，并对每个共有的方法进行同步，使得每次只有一个线程能访问它们

- **同步容器类的问题**： 同步类容器都是线程安全的，但在某些场景下可能需要加锁来保护复合操作，如： 

  - **迭代**(反复访问元素，遍历完容器中的所有元素)
  - **跳转**(根据指定的顺序找到当前元素的下一个元素)
  - **条件运算**

  这些复合操作在多线程并发的修改容器时，可能会表现出意外的行为

  > 创建新的复合操作时，进行加锁操作便可实现同步策略

- 当容器在迭代过程中被修改，就会抛出 `ConcurrentModificationException` 异常

  > **加锁**或**克隆一个副本**，在副本上进行迭代，就可避免抛出 ConcurrentModificationException 异常

- **隐藏迭代器**： 容器的 `toString(), hashCode()，equals(),containsAll()` 等方法，都会对容器进行迭代，这些间接的迭代都可能会抛出 ConcurrentModificationException 异常

  > - 如果状态与保护它的同步代码之间相隔越远，则开发人员就越容易忘记在访问状态时使用正确的同步
  >
  > - 正如封装对象的状态有助于维持不变性条件一样，封装对象的同步机制同样有助于确保实施同步策略

```java
//Vetor 上可能导致混乱结果的复合操作
public static Object getLast(Vector list){
     int lastIndex = list.size() - 1;
     return list.get(lastIndex);
}
public static void deleteLast(Vector list){
     int lastIndex = list.size() - 1;
     list.remove(lastIndex);
}

//对 vector 加锁的复合操作
public static Object getLast(Vector list){
     synchronized(list){ //***
          int lastIndex = list.size() - 1;
          return list.get(lastIndex);
     }
}
public static Object getLast(Vector list){
     synchronized(list){ //***
          int lastIndex = list.size() - 1;
          list.remove(lastIndex);
     }
}
```

## 2. 并发容器

- **同步容器**： 将所有对容器状态的访问都串行化来实现它们的安全性，但会严重降低并发性，当多个容器竞争容器的锁时，吞吐量将严重降低
- **并发容器**： 针对多个线程并发访问设计，通过并发容器代替同步容器，可以极大的提高伸缩性并降低风险

- 两个新的容器：
  - `Queue`： 用于临时保存一组等待处理的元素
  - `BlockingQueue`： 增加可阻塞的插入和获取操作

- `ConcurrentHashMap`： 使用一种粒度更细的**分段锁**来实现更大程度的共享

  > - ConcurrentHashMap 返回的迭代器具有**弱一致性**
  > - ConcurrentHashMap 不能被加锁来执行独占访问

- `CopyOnWriteArrayList`： 当往一个容器添加元素时，先将容器进行Copy，然后往新的容器里添加元素，添加完元素之后，再将原容器的引用指向新容器

  > - **安全性在于**： 只要正确的发布一个事实不可变对象，那么在访问该对象时就不再需要进一步的同步
  >
  > - 在每次修改时，都会创建并重新发布一个新的容器副本，从而实现可变性
  >
  > - 仅当迭代操作远远多于修改操作时，才应该使用“写入时复制”容器
  >
  > - **好处**： 可以对 CopyOnWrite 容器进行并发的读，而不需要加锁，是一种读写分离的思想

## 3. 阻塞队列

- **阻塞队列**： 提供了可阻塞的 put 和 take 方法，以及支持定时的 offer 和 poll 方法

  > **适用于生产者-消费者模式**，所有消费者共享一个工作队列

  - 如果队列已满，则 put 方法将阻塞直到有可用空间为止
  - 如果队列为空，则 take 方法将会阻塞直到有元素可用
  - offer 方法：如果数据项不能被添加到队列中，那么将返回一个失败状态；用于处理**负荷过载**

- **有界队列**：能抑制并防止产生过多的工作项，使应用程序在符合过载的情况下变得更加健壮

- **双端队列(Deque, BlockingDeque)**： 实现了在队列头和队列尾的高效插入和移除

  > - 适用于**工作密取**，即每个消费者有各自的双端队列
  > - 当双端队列为空时，会在另一个线程的队列尾查找新的任务，从而确保每个线程都保持忙碌的状态

```java
//实现的生产者与消费者
class Producer implements Runnable {  
    private String name;  
    private BlockingDeque<Integer> deque;  

    public Producer(String name, BlockingDeque<Integer> deque) {  
        this.name = name;  
        this.deque = deque;  
    }  

    public synchronized void run() {  
        for (int i = 0; i < 10; i++) {  
            try {  
                deque.putFirst(i);  
                System.out.println(name + " puts " + i);  
                Thread.sleep(300);  
            } catch (InterruptedException e) {  
                e.printStackTrace();  
            }  
        }  
    }  
}  

class Consumer implements Runnable {  
    private String name;  
    private BlockingDeque<Integer> deque;  

    public Consumer(String name, BlockingDeque<Integer> deque) {  
        this.name = name;  
        this.deque = deque;  
    }  

    public synchronized void run() {  
        for (int i = 0; i < 10; i++) {  
            try {  
                int j = deque.takeLast();  
                System.out.println(name + " takes " + j);  
                Thread.sleep(3000);  
            } catch (InterruptedException e) {  
                e.printStackTrace();  
            }  
        }  
    }  
}  

public class BlockingDequeTester {  
    public static void main(String[] args) {  
        BlockingDeque<Integer> deque = new LinkedBlockingDeque<Integer>(5);  
        Runnable producer = new Producer("Producer", deque);  
        Runnable consumer = new Consumer("Consumer", deque);  
        new Thread(producer).start();  
        try {  
            Thread.sleep(500);  
        } catch (InterruptedException e) {  
            e.printStackTrace();  
        }  
        new Thread(consumer).start();  
    }  
} 
```

## 4. 阻塞方法与中断方法

- 阻塞方法： 方法抛出 `interruptedException` 错误

- **Thread 提供的 interrupt 方法**： 用于终端线程或查询线程是否已被中断

  > 每个线程都有一个布尔类型的属性，用于表示线程的中断状态，当线程中断时将设置这个状态

- 处理中断的方式： 

  - 传递 `interruptedException`： 
    - 不捕获该异常
    - 捕获该异常，然后再执行某种简单的清理工作后再次抛出异常
  - **恢复中断**

## 5. 同步工具类

- **同步工具类**： 可以是任何对象，只要根据其自身的状态来协调线程的控制流

### 1. 阻塞队列

- **阻塞队列**： 不仅能作为保存对象的容器，还能协调生产者和消费者之间的控制流

  > `take, put`： 该方法将阻塞直到队列达到期望的状态

### 2. 闭锁

- **闭锁**： 可以延迟线程的进度直到其达到终止状态
  - 闭锁到达终止状态==前==，任何线程不能通过
  - 闭锁到达终止状态==时==，允许所有线程通过
  - 闭锁到达终止状态==后==，不会再改变状态

- `CountDownLatch`： 一种灵活的闭锁实现

  > 实现过程： 
  >
  > - **计数器**： 被初始化一个正数，表示需要等待的事件数量
  > - **递减计数器**： 表示有一个事件已发生
  > - `await() 方法`： 等待计数器到达 0，表示所有需等待事件都已发生

- `FutureTask`： 实现了 `Future` 语义，表示一种抽象的可生成结果的计算，该计算通过 `Callable` 来实现

  > - 可处于三种状态： **等待运行、正在运行、运行完成** 
  > - 当 FutureTask 进入完成状态后，会永远停止在该状态
  > - Future 表示一个任务的生命周期，并提供了相应的方法来判断是否已经完成或取消，以及获取任务的结果或取消任务
  > - `FutureTask.get()`： 若任务完成，立即返回结果；否则将阻塞直到任务完成，然后返回结果或抛出异常

```java
//计数测试中，使用 CountDownLatch 来启动和停止线程
public class TestHarness {  
    public long timeTasks(int n, final Runnable task) throws Exception {  
        final CountDownLatch startGate = new CountDownLatch(1);  
        final CountDownLatch endGate = new CountDownLatch(n);  
        for (int i = 0; i < n; i++) {  
            Thread t = new Thread() {  
                public void run() {  
                    try {  
                        startGate.await(); // 所有线程运行到此被暂停, 等待一起被执行  
                        try {  
                            task.run();  
                        } finally {  
                            endGate.countDown();  
                        }  
                    } catch (Exception e) {  
                    }  
                };  
            };  
            t.start();  
        }  
        long start = System.nanoTime();  
        startGate.countDown(); // 启动所有被暂停的线程  
        endGate.await(); // 等待所有线程执行完  
        long end = System.nanoTime();  
        return end - start;  
    }  

    //测试
    public static void main(String[] args) {  
        TestHarness th = new TestHarness();  
        Runnable r = new Runnable() {  
            public void run() {  
                System.out.println("running");  
            }  
        };  
        try {  
            th.timeTasks(10, r);  
        } catch (Exception e) {  
            e.printStackTrace();  
        }  
    }
}
```

### 3. 信号量(Semaphore)

- **计数信号量**： 用来控制同时访问某个特定资源的操作数量，或同时执行某个指定操作的数量

  > 可用来**实现某种资源池**，或**对容器施加边界**

- **虚拟许可**： 执行操作时，先获得许可，在使用后释放许可

  > - 许可初始数量可通过构造函数来指定
  > - 当 `acquire()` 方法无许可时，将阻塞到有许可为止
  > - `release()` 方法将返回一个许可给信号量

- **二值信号量**： 初始值为 1 的信号量，可用作**互斥体**，即谁拥有该唯一许可，谁就拥有了互斥锁

```java
//使用 Semaphore 为容器设置边界
public class BoundedHashSet <T>{  
    private final Set<T> set;  
    private final Semaphore sem;  

    public BoundedHashSet(int n) {  
        set = Collections.synchronizedSet(new HashSet<T>());  
        sem = new Semaphore(n);  
    }  

    public boolean add(T element) {  
        sem.acquire();  //请求信号量
        boolean result = false;  
        try {  
            result = set.add(element);  
        }finally {  
            sem.release();  
        }  
        return result;  
    }  

    public void remove(T o) {  
        boolean result = set.remove(o);  
        if (result) {  
            sem.release();  //返回信号量
        }  
    }  

    //测试
    public static void main(String[] args) {  
        final BoundedHashSet<String> bhs = new BoundedHashSet<String>(3);  
        for (int i = 0; i < 4; i++) {  
            Thread t = new Thread() {  
                @Override  
                public void run() {  
                    bhs.add(System.currentTimeMillis() + "");  
                };  
            };  
            t.start();  
        }  
    }  
} 
```

### 4. 栅栏

- **栅栏**： 能阻塞一组线程直到某个事件发生，类似于闭锁

- **与闭锁区别**： 所有线程必须同时到达栅栏位置，才能继续执行

  > - 闭锁用于等待事件
  > - 栅栏用于等待其他线程

- `CyclicBarrier`： 可使一定数量的参与方反复地在栅栏位置汇集，适用于迭代运算

  > - 当线程到达栅栏时，将调用 `await()` 方法，该方法将阻塞直到所有线程都到达栅栏位置
  > - 若所有线程都到达栅栏位置，则栅栏将打开，所有线程将被释放，栅栏被重置以便下次使用
  > - 若 `await()` 调用超时或 `await()` 阻塞线程被中断，所有阻塞 `await` 的调用都将终止并抛出 `BrokenBarrierException`
  > - 若成功通过栅栏，`await` 将为每个线程返回一个唯一的到达索引号，可利用这些索引来选举产生一个领导线程，并在下次迭代中由该领导线程执行一些特殊工作

- `Exchanger`： 一种**两方栅栏**，各方在栅栏位置交换数据，**适用于两方执行不对称操作**

  > 如： 缓冲区的读写线程可使用 `Exchanger` 来汇合，并将满缓冲区与空缓冲区交换

```java
public class Cellular {  
    private CyclicBarrier cb;  
    private Worker[] workers;  

    public Cellular() {  
        int count = Runtime.getRuntime().availableProcessors();  
        workers = new Worker[count];  
        for (int i = 0; i < count; i++) {  
            workers[i] = new Worker();  
        }  
        cb = new CyclicBarrier(count, new Runnable() {  
            public void run() {  
                System.out.println("the workers is all end...");  
            }  
        });  
    }  

    public void start() {  
        for (Worker worker : workers) {  
            new Thread(worker).start();  
        }  
    }  

    private class Worker implements Runnable {  
        public void run() {  
            System.out.println("working...");  
            try {  
                cb.await();//在这里线程阻塞，等待其他线程。  
            } catch (InterruptedException e) {  
                e.printStackTrace();  
            } catch (BrokenBarrierException e) {  
                e.printStackTrace();  
            }  
        }  
    }  

    //测试
    public static void main(String[] args) {  
        Cellular c = new Cellular();  
        c.start();  
    }  
}
```

# 

# # 结构化并发应用程序

# 六、任务执行

## 1. 线程中执行任务

- **显示为任务创建线程的注意事项**： 

  - 任务处理过程从主线程中分离出来，使得主循环能更快的重新等待下一个到来的连接

    > 这使得程序在完成前面的请求之前可以接受新的请求，从而提高响应性

  - 任务可以并行处理，从而能同时服务多个请求

  - 任务处理代码必须是线程安全的，因为当有多个任务时会并发的调用这段代码

- **无限创建线程的不足**：

  - 线程生命周期的开销很高
  - 资源消耗
  - 稳定性

## 2. Executor 框架

###1. 概述

- **任务是一组逻辑工作单元，而线程是使任务异步执行的机制**
- 两种通过线程来执行任务的策略：
  - **所有任务在单个线程中穿行执行**： 吞吐量与响应性低
  - **每个任务在各自的线程中执行**： 资源管理比较复杂

- `Executor 框架`： 提供了一种标准的方法将任务的**提交过程与执行过程解耦**，并用 `Runnable` 来表示任务

  > - Executor 的实现提供了对生命周期的支持、统计信息收集、应用程序管理机制、性能监视等功能
  > - 提交予执行解耦，有助于在部署阶段选择与可用硬件资源最匹配的执行策略
  > - 代码 `new Thread(runnable).start()` 可用 `Executor` 来代替

- 要使用 `Executor`，须将任务表述为一个 `Runnable`
- 在 `Executor` 框架中，已提交但尚未开始的任务可以取消

```java
//为每个请求启动一个新线程的 Executor
public class ThreadPerTaskExecutor implements Executor {  
    public void execute(Runnable command) {  
        new Thread(command).start();  
    }  
} 

//在调用线程中以同步方式执行所有任务的 Executor
public class WithinThreadExecutor implements Executor {  
    public void execute(Runnable command) {  
        command.run();  
    }  
} 
```



### 2. 线程池

- **线程池**： 指管理一组同构工作线程的资源池，与工作队列相关，在工作队列中保存了所有等待执行的任务
- **工作者线程**： 从工作队列中获取一个任务，执行任务，然后返回线程池并等待下一个任务

Executor 可创建的线程池：

- `newFixedThreadPool`： 创建一个**固定长度的线程池**

  > 每当提交一个任务便创建一个线程，直到达到线程池的最大数量

- `newCachedThreadPool`： 创建一个**可缓存的线程池**

  > - 若线程池的当前规模超过了处理需求，则回收空闲线程
  >
  > - 当需求增加，则添加新的线程，此时线程池规模不限

- `newSingleThreadExecutor`： 创建**单个工作者线程**来执行任务

  > - 若线程异常结束，会创建另一个线程来代替
  > - 能确保依照任务在队列中的顺序来串行执行

- `newScheduledThreadPool`： 创建一个**固定长度的线程池**，且**以延迟或定时方式来执行任务**

### 3. 生命周期

Executor 扩展 `ExecutorService` 接口，添加了用于生命周期管理的方法，其生命周期的状态为： 运行、关闭、已终止

- `ExecutorService` 在初始创建时处于运行状态

- 方法 `shutdown()` 会执行**平缓**的关闭过程： 不再接受新的任务，同时等待已提交的任务执行完成

  方法 `shutdownNow()` 会执行**粗暴**的关闭过程： 将尝试取消所有运行中的任务，且不再启动队列中尚未开始执行的任务

  > `ExecutorService` 关闭后提交的任务将由**拒绝执行处理器**来处理：
  >
  > - 其会关闭任务
  > - 或使 execute 方法抛出一个未检查的 `RejectedExecutionExeception`

- 方法 `awaitTermination` 可用于等待 `ExecutorService` 到达终止状态

  方法 `isTermination` 可用来轮询 `ExecutorService` 是否已经终止

> - `ExecutorService` 中的所有 submit 方法都将返回一个 Future，从而将一个 Runnable 或 Callable 提交给 Executor，并得到一个 Future 用来获得任务的执行结果或取消任务

### 4. 延迟任务与周期任务

- `Timer` 类负责管理延迟任务与周期任务
- `Timer` 在执行所有定时任务时只会创建一个线程
- `Timer` 支持基于绝对时间的调度机制
- **线程泄漏**： 当出现异常，Timer 不会恢复线程的执行，而是会错误的将整个 Timer 都取消；已调度但未执行的周期不再执行，新的任务也不会被调度

## 3. 找出可利用的并行性

- `CompletionService`： 将 `Executor 和 BlockingQueue` 的功能融合在一起

  - 可将 `Callable` 任务提交给它来执行
  - 然后使用类似于队列操作的 take 和 poll 等方法来获得已完成的结果
  - 而这些结果会在完成时被封装为  Future

  `ExecutorCompletionService` 实现了 `CompletionService`，并将计算部分委托给 `Executor`

```java
//使用 Future 等待图像下载
public class FutureRenderer {
    private final ExecutorService executor = Executors.newCachedThreadPool();

    void renderPage(data);(String source) {
        final List<ImageInfo> imageInfos = scanForImageInfo(source);
        //下载图片
        Callable<List<ImageData>> task = new Callable<List<ImageData>>() {
            @Override
            public List<ImageData> call() throws Exception {
                List<ImageData> list = new ArrayList<ImageData>();
                for (ImageInfo imageInfo : imageInfos) {
                    list.add(imageInfo.download());
                }
            }
        };
        //处理图片
        //将一个Runnable或者Callable任务传递给ExecutorService的submit方法
        //将返回一个Future用于获得任务的执行结果或者取消任务
        Future<List<ImageData>> future = executor.submit(task);
        renderText(source);
        try {
            ////get()方法会一直阻塞，直到callable的任务全部完成
            List<ImageData> list = future.get();
            for(ImageData data : future){
                renderPage(data);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            // 由于不需要结果，因此取消任务
            future.cancel(true);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }
}

//使用 CompletionService，使页面元素在下载完成后立即显示出来
public class Test {
    private final ExecutorService executor = Executors.newCachedThreadPool();

    void renderPage(String source) {
        final List<ImageInfo> imageInfos = scanForImageInfo(source);
        CompletionService<ImageData> service = 
            new ExecutorCompletionService<ImageData>(executor);
        for (final ImageInfo imageInfo : imageInfos) {
            service.submit(new Callable<ImageData>() {
                public ImageData call() throws Exception {
                    return imageInfo.downloadImage();
                }
            });
        }
        renderText(source);
        for (int i = 0; i < imageInfos.size(); i++) {
            Future<ImageData> f;
            try {
                f = service.take();
                ImageData imageData = f.get();
                renderImage(imageData);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
    }
}
```

# 七、取消与关闭

## 1. 任务取消

- 如果外部代码能在某个操作正常完成之前将其置入 “完成” 状态，则该操作称为 **可取消的**

- 操作取消原因：
  - **用户请求取消**
  - **有时间限制的操作**
  - **应用程序事件**
  - **错误**
  - **关闭**

```java
//使用 volatile 类型的域来保存取消状态
public class PrimeGenerator implements Runnable {  
    private final List<BigInteger> primes = new ArrayList<BigInteger>();  
    private volatile boolean cancelled;  
    @Override  
    public void run() {  
        BigInteger p = BigInteger.ONE;  
        while (!cancelled) {  
            p = p.nextProbablePrime();  
            synchronized (this) {  
                primes.add(p);  
            }  
        }  
    }  

    public void cancel() {  
        this.cancelled = true;  
    }  

    public synchronized List<BigInteger> get() {  
        return new ArrayList<BigInteger>(primes);  
    }
}
```

### 1. 中断

- 响应中断时执行的操作：
  - 清除中断状态
  - 抛出 `InterruptedException`
  - 阻塞操作由于中断而提前结束

- **中断操作**： 不会真正的中断一个正在运行的线程，而只是发出中断请求，然后由线程在下一个合适的时刻中断自己，该时刻也被称为**取消点**，如： `interrupt` 方法

  > 中断时实现取消的最合理方式，可使用中断来替代 `boolean` 来请求取消

- **中断策略**： 规定线程如何解释某个中断请求

  > - 最合理的中断策略是某种形式的**线程级取消操作或服务级取消操作**
  > - 线程只能由其所有者中断

- 任务不会在其自己拥有的线程中执行，而是在某个服务拥有的线程中执行

  > 因此采用的最合理取消策略： 尽快推出执行流程，并把中断信息传递给调用者，从而使调用栈中的上层代码可以采取进一步的操作

- **响应中断**： 处理 `InterruptedException` 的策略如下
  - **传递异常**： 
  - **恢复中断状态**： 

- 只有实现了线程中断策略的代码才可以屏蔽中断请求
- 对于不持支取消但仍可调用可中断阻塞方法的操作，应在本地保存中断状态并在返回前恢复状态

### 2. Future 实现取消

- `cancel(bool)` 方法： 表示取消操作是否成功
  - 若 bool 为 true 且任务当前正在某个线程中运行，则线程能被中断
  - 若 bool 为 false，则意味着任务还未启动，不要运行

## 2. 停止基于线程的服务

- **线程封装原则**： 除非拥有某个线程，否则不能对该线程进行操控
- **线程所有权不可传递**： 应用程序可以拥有服务，服务也可拥有工作者线程，但应用程序不可拥有工作者线程

- 当持有线程的服务的存在时间大于创建线程的方法的存在时间，则应该提供生命周期方法

## 3. 处理正常的线程终止

- 在运行时间较长的应用程序中，通常会为所有线程的未捕获异常指定同一个异常处理器，且该处理器至少会将异常信息记录到日志中

- 通过 execute 提交的任务，才能将抛出的异常交给未捕获异常处理器

- 通过 submit 提交的任务，其异常将被任务是任务返回状态的一部分

  > 若 submit 提交的任务由于抛出异常而结束，则异常将被 Future.get 封装在 ExecutionException 中重新抛出

## 4. JVM 关闭

**JVM 关闭过程**：

- 正常关闭中，JVM 会首先调用所有已注册的关闭钩子
- 在关闭应用程序时，守护线程将与关闭线程并发执行
- 当所有关闭钩子都执行结束时，若 runFinalizersOnExit 为 true，则 JVM 将运行终结器，然后再停止

- JVM 最终结束时，以上线程将被强行结束

  > - 若关闭钩子或终结器未执行完成，则正常关闭进程挂起，JVM 被强行关闭
  > - 当强行关闭时，只关闭 JVM，而不会运行关闭钩子



- **关闭钩子**： 指通过 Runtime.addShutdownHook 注册的但尚未开始的线程，可以用于实现服务或应用程序的清理工作

  > 关闭钩子不应依赖于被应用程序或其他钩子关闭的服务，可通过**对所有服务使用同一个关闭钩子**来实现

- **守护线程**： 当守护线程退出时，JVM 会正常退出操作

  > 当 JVM 停止时，所有存在的守护线程将会被抛弃，即不会执行 finally 代码块，也不执行回卷栈

- **终结器(finalize)**： 由垃圾回收器定义的释放持久化资源的方法

  > 避免使用终结器

# 八、线程池的使用

## 1. 任务与执行策略之间的隐形耦合

需明确指定执行策略的任务：

- 依赖性任务
- 使用线程封闭机制的任务
- 对响应时间敏感的任务
- 使用 ThreadLocal 的任务



- 只有当任务类型相同且独立时，线程池的性能才能达到最佳
- 线程池中，若任务依赖于其他任务，可能产生死锁
- **线程饥饿死锁**： 线程池中，所有执行任务的线程都由于等待其他仍处于工作队列中的任务而阻塞

##2. 线程池大小 

- 线程池理想大小取决于被提交任务的类型以及所部署系统的特性

  > - 线程池最优大小： $N_{threads}=N_{cpu} * U_{cpu} * (1 + W/C)$
  >
  >   其中： $N_{cpu}$ 为 CPU 数量，$U_{cpu}$ 为 CPU 利用目标，$W/C$ 为等待时间与完成时间之比
  >
  > - 通过 Runtime 来获得 CPU 数目： `int N_CPUS = Runtime.getRuntime().availableProcessors();`

## 3. ThreadPoolExecutor

###1. 配置 ThreadPoolExecutor

- `ThreadPoolExecutor`： 是一个灵活的、稳定的线程池，允许进行各种定制

- 线程池的基本大小、最大大小、存活时间等因素共同负责线程的创建于销毁

  > 通过调节基本大小与存活时间，可以帮助线程池回收空闲线程占有的资源

  - **基本大小**： 线程池的目标大小
  - **最大大小**： 表示可同时活动的线程数量的上限

- `newFixedThreadPool`： 将线程池的基本大小和最大大小设置为参数中指定的值，且创建的线程池不会超时

  `newCachedThreadPool`： 将线程池的基本大小设置为 0，最大大小设置为 Integer.MAX_VALUE，超时时间设置为 1 分钟

  > 该方法创建出的线程池可被无限扩展，且当需求降低时会自动收缩

- `ThreadPoolExecutor` 允许一个 BlockingQueue 来保存等待执行的任务，基本任务排队方法有无界队列、有界队列、同步移交

- 执行策略：

  - **饱和策略**： 默认为中止策略，该策略将抛出未检查的 RejectedExecutionException

    > 当有界队列被填满后，执行饱和策略
    >
    > 可通过调用 setRejectedExecutionHandler 来修改

  - **抛弃策略**： 抛弃下一个将被执行的任务，然后尝试重新提交新的任务

    > 当新提交的任务无法保存到队列中等待执行时，执行饱和策略

  - **调用者运行策略**： 实现了一种调节机制，该策略既不会抛弃任务，也不会抛出异常，而是将某些任务回退到调用者，从而降低新任务的流量

```java
//使用 Semaphore 来控制任务的提交速率
public class BoundedExecutor {  
    private final Executor exec;  
    private final Semaphore semahpore;  

    public BoundedExecutor(Executor exec, int bound) {  
        this.exec = exec;  
        this.semahpore = new Semaphore(bound);  
    }  

    public void submitTask(final Runnable command) throws InterruptedException {  
        semahpore.acquire();  //请求资源
        try {  
            exec.execute(new Runnable() {  
                @Override  
                public void run() {  
                    try {  
                        command.run();  
                    } finally {  
                        semahpore.release();  
                    }  
                }  
            });  
        } catch (RejectedExecutionException e) {  
            semahpore.release();  
        }  
    }  
} 
```

###2. 扩展 ThreadPoolExecutor

- `beforeExecute, afterExecute, terminated` 等方法用于扩展 ThreadPoolExecutor 的行为

  - 当任务从 run 中正常返回或抛出异常，则 afterExecute 将被调用

  - 当 beforeExecute 抛出一个 RuntimeException，则任务将不被执行，且 afterExecute 也不会被调用

  - 当线程池完成关闭操作时调用 terminated

    > - terminated 可用于释放 Executor 在其生命周期里分配的各种资源
    > - terminated 还可执行发送通知、记录日志、收集 finalize 统计信息等操作

```java
//增加日志和计时等功能的线程池
public class TimingThreadPool extends ThreadPoolExecutor {  
    private final ThreadLocal<Long> startTime = new ThreadLocal<Long>();  
    private final Logger log = Logger.getLogger("TimingThreadPool");  
    private final AtomicLong numTask = new AtomicLong();  
    private final AtomicLong totalTime = new AtomicLong();  

    @Override  
    protected void beforeExecute(Thread t, Runnable r) {  
        super.beforeExecute(t, r);  
        log.fine(String.format("Thread %s: start %s", t, r));  
        startTime.set(System.nanoTime());  
    }  

    @Override  
    protected void afterExecute(Runnable r, Throwable t) {  
        try {  
            long endTime = System.nanoTime();  
            long taskTime = endTime - startTime.get();  
            totalTime.addAndGet(taskTime);  
            numTask.incrementAndGet();  
        } finally {  
            super.afterExecute(r, t);  
        }  
    }  

    @Override  
    protected void terminated() {  
        try {  
            log.info(String.format("Terminated: avg time=%dns", 
                                   totalTime.get() / numTask.get()));  
        }finally {  
            super.terminated();  
        }  
    }  
} 
```

## 4. 递归算法的并行化

- 若循环中的迭代操作是独立的，且不需要等待所有的迭代操作都完成在继续执行，则 Executor 会将串行循环转化为并行循环

```java
//树节点的串行递归
public <T> void sequentialRecursive(List<Node<T>> nodes, Collection<T> results) {  
    for (Node<T> node : nodes) {  
        results.add(node.compute());  
        sequentialRecursive(node.getChildren(), results);  
    }  
}  

//树节点的并行递归
public <T> void parallelRecursive(final Executor exec, List<Node<T>> nodes, 
                                  final Collection<T> results) {  
    for (final Node<T> node : nodes) {  
        exec.execute(new Runnable() {  
            @Override  
            public void run() {  
                results.add(node.compute());  
            }  
        });  
        parallelRecursive(exec, nodes, results);  
    }  
}  

public <T> Collection<T> getParallelResult(List<Node<T>> nodes) throws InterruptedException {  
    ExecutorService exec = Executors.newCachedThreadPool();  
    Queue<T> resultQueue = new ConcurrentLinkedQueue<T>();  
    parallelRecursive(exec, nodes, resultQueue);  
    exec.shutdown();  
    exec.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);  
    return resultQueue;  
} 
```

# 

# # 活跃性、性能、测试

# 十、避免活跃性危险

## 1. 死锁

### 1. 锁顺序死锁

- 若所有线程以固定的顺序来获得锁，则在程序中就不会出现锁顺序死锁问题

```java
//简单的锁顺序死锁
public class Test {
    private final Object left = new Object();
    private final Object right = new Object();

    public void leftRight(){
        synchronized (left) {
            synchronized (right) {
                //do
            }
        }
    }
    public void rightLeft(){
        synchronized (right) {
            synchronized (left) {
                //do
            }
        }
    }
}
```

### 2. 动态的锁顺序死锁

```java
/**动态的锁顺序死锁
 * 当出现下列情况时，或死锁：
 * 用户 A：tansfer(myAccount,yourAccount,10)
 * 用户 B：tansfer(yourAccount,myAccount,20)
 */
public void tansfer(Account fromAccount,Account toAccount,DollarAmout amout){
    synchronized(fromAccount){
        synchronized(toAccount){
            if(fromAccount.getBalance().compareTo(amout) < 0){
                throw new Exception();
            }else{
                fromAccount.debit(amount);
                toAccount.credit(mount);
            }
        }
    }
}


//通过 System.identityHashCode 来定义锁顺序，避免动态死锁
public void transferMoney(final Account fromAccount, 
                          final Account toAccount, final DollarAmount amount) {  
    class Helper {  
        public void transfer() {  
            if (fromAccount.getBalance().compareTo(amount) < 0) {  
                throw new RuntimeException();  
            } else {  
                fromAccount.debit(amount);  
                toAccount.credit(amount);  
            }  
        }  
    }  
    
   // 通过唯一hashcode来统一锁的顺序, 如果account具有唯一键, 可以采用该键来作为顺序.  
    int fromHash = System.identityHashCode(fromAccount);  
    int toHash = System.identityHashCode(toAccount);  
    if (fromHash < toHash) {  
        synchronized (fromAccount) {  
            synchronized (toAccount) {  
                new Helper().transfer();  
            }  
        }  
    } else if (fromHash > toHash) {  
        synchronized (toAccount) {  
            synchronized (fromAccount) {  
                new Helper().transfer();  
            }  
        }  
    } else {  
        synchronized (tieLock) { // 针对fromAccount和toAccount具有相同的hashcode  
            synchronized (fromAccount) {  
                synchronized (toAccount) {  
                    new Helper().transfer();  
                }  
            }  
        }  
    }  
}  
```

### 3. 协作对象之间发生死锁

- 若在持有锁时调用外部方法，则会出现活跃性问题

```java
//协作对象之间的锁顺序死锁
class Taxi{
    private Point location,destination;
    private final Dispather dispather;

    public Taxi(Dispather dispather){
        this.dispather = dispather;
    }

    public synchronized Point getLocation(){
        return location;
    }

    public synchronized void setLocation(Point location){
        this.location = location;
        if(location.equals(destination)){
            dispather.notifyAvailable(this);
        }
    }
}

class Dispather{
    private final Set<Taxi> taxi;
    private final Set<Taxi> availableTaxis;

    public Dispather(){
        taxi = new HashSet<Taxi>();
        availableTaxis = new HashSet<Taxi>();
    }

    public synchronized void notifyAvailable(Taxi taxi){
        availableTaxis.add(taxi);
    }
}
```

### 4. 开放性调用

- **开放调用**： 在调用某个方法时不需要持有锁

```java
//通过公开调用来比卖你在相互协作对象之间产生死锁
class Taxi{
    private Point location,destination;
    private final Dispather dispather;

    public Taxi(Dispather dispather){
        this.dispather = dispather;
    }

    public synchronized Point getLocation(){
        return location;
    }

    public synchronized void setLocation(Point location){
        boolean reached;
        synchronized(this){//加锁***
            this.location = location;
            reached = location.equals(destination);
        }
        if(reached){
            dispather.notifyAvailable(this);
        }
    }
}

class Dispather{
    private final Set<Taxi> taxi;
    private final Set<Taxi> availableTaxis;

    public Dispather(){
        taxi = new HashSet<Taxi>();
        availableTaxis = new HashSet<Taxi>();
    }

    public synchronized void notifyAvailable(Taxi taxi){
        availableTaxis.add(taxi);
    }
}
```

### 5. 资源死锁

- 当多个线程相互持有彼此正在等待的锁而又不释放自己持有的锁时，会发生死锁

## 2. 死锁避免与诊断

- 有界线程池/资源池与相互依赖的任务不能一起使用
- 如果要获取多个锁，必须考虑锁的顺序，尽可能使用开放调用
- 使用显示锁检测死锁，并且可以从死锁中恢复过来
- 通过线程转储信息来分析死锁

## 3. 其他活跃性危险

- **饥饿**：由于线程无法访问它所需要的资源而不能继续执行时，就发生了饥饿
- **糟糕的响应性也很影响活跃性**
- **活锁**：尽管没有被阻塞, 线程却仍然不能继续, 因为它不断重试相同的操作, 却总是失败，活锁通常发生在消息处理应用程序中

# 十一、性能与可伸缩性

##1. 概述

- 应用程序的性能可以用多个指标来衡量：服务时间，延迟时间，吞吐率，效率，可伸缩性，容量

- **可伸缩性**：当增加计算机资源时(内存，CPU，存储容量或IO带宽)，程序的吞吐量或者处理能力要相应得提高

- **Amdahl 定律**： 在增加计算资源的情况下，程序在理论上能够实现最高加速比，该值取决于可并行组件与串行组件所占比重

  > $Speedup \leq \frac{1}{F + \frac{1 - F}{N}}$
  >
  > - `F`： 为须被串行执行的部分
  > - `N`： 处理器的数量

## 2. 线程引入开销

### 1. 上下文切换

- **上下文切换**： 若可运行的线程数大于 CPU 数，则操作系统会将某个正在运行的线程调度出来，从而使其他线程能使用 CPU

  >  该过程将保存当前线程执行的上下文，并将新调度的线程的上下文设置为当前的上下文

### 2. 内存同步

- **内存栅栏**： 可以刷新缓存，使缓存无效，刷新硬件的写缓存，以及停止执行管道

  > - 可能会对性能带来影响，因为它将抑制一些编译器优化
  >
  > - 在内存栅栏中，大多数操作不能重排序

- JVM 能通过优化去掉一些不会发生竞争的锁，以减少不必要的性能开销
- **锁消除优化**： 如果一个锁对象只能由当前线程访问，则 JVM 会去掉这个锁的获取操作

### 3. 阻塞

- **非竞争的同步**可在 JVM 中进行处理，**竞争的同步**则需要操作系统的介入，从而增加开销

- JVM 在实现阻塞行为时，采用**自旋等待**或通过操作系统**挂起被阻塞的线程**

  > - **自旋等待**： 通过循环不断的尝试获取锁，直到成功
  > - 两种方式的效率高低： 取决于上下文切换的开销和等待时间

## 3. 减少锁竞争

- 影响锁竞争的可能性因素：
  - **锁的请求频率**
  - **每次持有锁的时间**
- 降低锁的竞争程度的方式： 
  - **减少锁的持有时间** 
  - **降低锁的请求频率** 
  - **使用其它带有协调机制的锁**

### 1. 缩小锁的范围

- **尽可能的缩短锁的持有时间**

### 2. 减小锁的粒度

- **降低线程请求锁的频率**，可通过锁分解和锁分段来实现

#### 1. 锁分解

- **锁分解**： 若一个锁需要保护多个相互独立的状态变量，则可以将该锁分解为多个锁，且每个锁只保护一个变量，从而提高可伸缩性，并最终降低每个锁被请求的频率
- 对竞争适中的锁进行分解，则是将该锁转换为非竞争的锁，从而有效提高性能和可伸缩性

#### 2. 锁分段

- **锁分段**： 对一组独立对象上的锁进行分解

- **锁分段缺点**： 要获取多个锁来实现独占访问，因而导致难度更高，开销更大

## 4. 放弃使用独占锁

- **替代方式： 并发容器，读-写锁，不可变对象，原子变量**

#  

# # 高级主题

# 十三、显示锁

- 与内置锁不同，`Lock` 提供了一种**无条件的、可轮询的、定时的、可中断的**锁获取操作，所有加锁和解锁都是**显式的**

  ```java
  public interface Lock {
      void lock();
      void lockInterruptibly() throws InterruptedException;
      boolean tryLock();
      boolean tryLock(long time, TimeUnit unit) throws InterruptedException;
      void unlock();
      Condition newCondition();
  }
  ```

## 1. `ReentrantLock`

- `ReentrantLock` 实现了Lock接口，提供了与 synchronized 同样的**互斥性和内存可见性**

- **内置锁的限制**：无法中断一个正在等待获取锁的线程，无法获取一个锁时无限得等待下去

  ReentrantLock 更加灵活，能提供更好的活跃性和性能

  > - 仅当内置锁不能满足需求时，才考虑使用 ReentrantLock
  > - 内置锁优点： 在线程转储中能给出在哪些调用帧中获得了哪些锁，并能检测和识别发生死锁的线程

- 内置锁的释放时自动的，而 ReentrantLock 的释放**必须在finally手动释放**

  ```java
  Lock lock = new ReentrantLock();
  lock.lock();
  try{
  
  } finally{
      lock.unlock();
  }
  ```

**可中断的锁获取操作**： 

- `lockInterruptibly`： 能在获得锁的同时保持对响应的中断
- 定时的 `tryLock` 也能响应中断

- **非块结构加锁**： 为每个节点使用独立的锁，使不同线程能独立进行访问等操作

  > 内置锁中，锁的获取和释放都是基于代码块的，即锁的释放与获取操作处于同一代码块中

## 2. 公平锁与非公平锁

- **公平锁**：线程按照它们发出请求的顺序来获得锁 
- **非公平锁**：允许插队，当一个线程请求非公平锁时，如果在发出请求的同时该锁的状态变为可用，那么这个线程将跳过队列中所有等待线程并获得该锁

- **非公平锁的性能高于公平锁**，因为恢复一个被挂起的线程与该线程真正开始运行之间存在着严重的延迟

## 3. 读写锁

- **读写锁**：一个资源可以被多个读操作访问，或者被一个写操作访问，但两者不能同时访问

- `ReadWriteLock` 的可选实现：

  - **释放优先**： 当一个写线程释放锁，且队列中同时存在一个写线程和读线程，哪个应该先被调度：
    - 在公平锁中，等待时间最长的线程优先获得锁
    - 在非公平锁中，线程调度的顺序是不确定的

  - **读线程插队**： 如果锁由读线程持有，但有写线程正在等待，则新到达的读线程能否插队立即获得锁的访问权： 
    - 在公平锁中，如果持有锁的是读线程，而写线程正在请求写入锁，那么其他读线程都不能获取读取锁，直到写线程使用完并释放了写入锁

  - **重入性**： 读取锁和写入锁是否可重入： 
    - 读取所和写入锁都可重入

  - **降级**： 如果一个线程持有写入锁，则它能否在不释放写入锁的同时获取读取锁： 
    - 写入锁允许降级为读取锁

  - **升级**： 读取锁能否优先于其它正在等待的写入锁和读取锁而升级为一个写入锁： 
    - 读取锁不允许升级为写入锁，因为这可能造成死锁

# 十四、构建自定义的同步工具

## 1. 条件队列

- **条件队列**： 使一组线程能通过某种方式来等待特定的条件变成真，条件队列中的元素是一个个正在等待相关条件的线程

  > 每个对象都可以作为一个条件队列，且 Object 的 `wait(), notify(), notifyAll()` 构成了内部条件队列的 API
  >
  > - `wait()`： 调用 wait 会自动释放锁，并请求系统挂起当前线程，从而使其他线程能够获得这个锁 
  >
  > - `notify()`： 发出通知，解除阻塞条件，JVM 会从这个条件队列上等待的多个线程选择一个来唤醒
  >
  > - `notifyAll()`： 发出通知，解除阻塞条件，JVM 会唤醒所有在这个条件队列上等待的线程 
  >
  > - `notify() 与 notifyAll() 的区别`： 
  >
  >   - 多数情况下，优先选择 `notifyAll()`
  >
  >   - 满足两个条件时，使用 `notify()`：
  >
  >     > - **所有等待线程的类型都相同**： 只有一个条件谓词与条件队列相关，且每个线程在从 wait 返回后将执行相同的操作
  >     > - **单进单出**：在条件变量上的每次通知，最多只能唤醒一个线程来执行 

- **条件谓语**：线程等待的条件

- **条件等待的三元关系**： 加锁、wait() 方法、条件谓词

  > 使用条件等待的注意事项：
  >
  > - 首先有一个条件谓词，线程执行前必须先通过这些测试
  > - 调用 wait() 之前测试条件谓词，并从 wait() 中返回时再次进行测试
  > - 在一个循环中调用 wait()
  > - 确保使用于条件队列相关的锁来保护构成条件谓词的各个状态变量
  > - 当调用 `wait(), notify(), notifyAll()` 时，一定要持有与条件队列相关的锁
  > - 在检查条件谓词后及开始执行相应操作前，不要释放锁

- **丢失的信号**： 一种活跃性故障，指线程必须等待一个已为真的条件，但在开始前没有检查条件谓词

```java
//条件谓词： not-empty
public synchronized V take() throws InterruptedException{
    while(isEmpty()){ //阻塞直到非空
        wait();
    }
    V v = get();
    notifyAll();
    return v;
}
```

## 2. Condition 对象

- `Condition`： 一种广义的内置条件队列，可在关联的 Lock 上调用 `Lock.newCondition` 来创建

  > - **内置条件队列的缺陷**： 每个内置锁只能有一个相关联的条件队列
  > - `Lock 与 Condition`： 可用于实现带有多个条件谓语的并发对象
  > - Condition 中，`await, signal, signalAll` 分别对应 `wait, notify, notifyAll`

```java
//使用显示条件变量实现的有界缓存
public class ConditionBoundedBuffer<T> {  
    private static final int BUFFER_SIZE = 2;  
    private final Lock lock = new ReentrantLock();  
    //条件谓词： notFull
    private final Condition notFull = lock.newCondition();  
    //条件谓词： notEmpty
    private final Condition notEmpty = lock.newCondition();  
    private final T[] items = (T[]) new Object[BUFFER_SIZE];  
    private int tail, head, count;  

    //阻塞并直到： notFull
    public void put(T x) throws InterruptedException {  
        lock.lock();  
        try {  
            while (count == items.length) {  
                notFull.await();  //***
            }  
            items[tail] = x;  
            if (++tail == items.length) {  
                tail = 0;  
            }  
            count++;  
            notEmpty.signal();  //***
        } finally {  
            lock.unlock();  
        }  
    }  

    //阻塞并直到： notEmpty
    public T take() throws InterruptedException {  
        lock.lock();  
        try {  
            while (count == 0) {  
                notEmpty.await(); //*** 
            }  
            T x = items[head];  
            items[head] = null;  
            if (++head == items.length) {  
                head = 0;  
            }  
            count--;  
            notFull.signal();  //***
            return x;  
        } finally {  
            lock.unlock();  
        }  
    }  
} 
```

## 3. AbstractQueuedSynchronizer(AQS)

- 同步的实现都是基于 `AbstractQueuedSynchronizer(AQS)`，AQS 是一个用于构建锁和同步器的框架

  > `ReentrantLock，Semaphore，CountDownLatch` 等都是基于AQS构建的

- AQS 构建的容器中，最基本的就是获取操作和释放操作

  > - `CountDownLatch`： 获取意味着等待并直到闭锁到达结束状态
  > - `FutureTask`： 获取意味着等待直到任务已经完成

- AQS 负责同步容器类中的状态，它管理了一个整数状态信息，可以通过 `getState, setState, compareAndSetState` 来设置和获取

  > 例如： 
  >
  > - `ReentrantLock` 用它来表示线程已经重复获取该锁的次数
  > - `Semaphore` 用它来表示剩余的许可数量
  > - `FutureTask` 用它来表示任务的状态（尚未开始，正在运行，已完成以及以取消）

```java
//AQS 中获取和释放操作的标准形式
boolean acquire() throws InterruptedException {
    while(当前状态不允许获取操作){
        if(需要阻塞获取请求){
            若当前线程不在队列中，则将其插入队列
            阻塞当前队列
        }else {
            返回失败
        }
    }
    可能更新同步器的状态
    若线程处于队列中，则将其移出队列
    返回成功
}

void release(){
    更新同步器的状态
    if(新的状态允许某个被阻塞的线程获取成功){
        解除队列中的一个或多个线程的阻塞状态
    }
}
```

## 4. `java.util.concurrent` 同步容器类中的 AQS

- `ReentrantLock`： 只支持独占方式的获取操作
- `Semaphore`： 将 AQS 的同步状态用于保存当前可用许可的数量
- `CountDownLatch`： 在同步状态中保存的是当前的计数值
- `FutureTask`： AQS 同步状态被用来保存任务的状态
- `ReentrantReadWriteLock`： 单个 AQS 子类将同时管理读取加锁和写入加锁，并以两个 16 位的状态来分别表示读取/写入锁的计数	

# 十五、原子变量与非阻塞同步机制

## 1. 比较并交换(CAS)

- CAS 包含 3 个操作数： **需要读写的内存位置 V，进行比较的值 A，拟写入的新值 B** 

  > - 仅当 `V == A` 时, CAS 才会用新值 B 原子化地更新 V 的值
  > - 无论位置 V 的值是否等于 A，CAS 都会返回 V 原有的值

- **CAS 是一项乐观技术**，它抱着成功的希望进行更新, 并且如果另一个线程在上次检查后更新了该变量, 它能够发现错误

- 当多个线程试图使用 CAS 同时更新相同的变量时, 只有一个线程会更新变量的值

  > - 失败的线程不会被挂起，但会被告知失败，同时允许重试
  > - 对于锁： 当锁获取失败时，线程将会被挂起

- **CAS 的典型使用模式**： 首先从 V 中读取值 A，根据 A 计算值 B，然后通过 CAS 以原子操作将 V 的值 A 变为 B

```java
//模拟 CAS 操作
public class SimulateCAS {  
    private int value;  

    public synchronized int get() {  
        return value;  
    }  

    public synchronized int compareAndSwap(int expectedValue, int newValue) {  
        int oldValue = value;  
        if (expectedValue == oldValue) {  
            value = newValue;  
        }  
        return oldValue;  //无论能否修改，都返回旧值
    }  

    public synchronized boolean compareAndSet(int expectedValue, int newValue) {  
        return (expectedValue == compareAndSwap(expectedValue, newValue));  
    }  
}

//基于 CAS 实现的线程安全的计数器
public class CasCounter {  
    private SimulateCAS value = new SimulateCAS();  

    public int getValue() {  
        return value.get();  
    }  

    public int increment() {  
        int v;  
        do {  
            v = value.get();  
        } while (v != value.compareAndSwap(v, v + 1));  
        return v + 1;  
    }  
}  
```

## 2. 非阻塞算法

- **非阻塞算法**：一个线程的失败者挂起不会导致其他线程也失败或挂起
- **无锁算法**：在算法的每一个步骤都存在某个线程能够执行下去

- **无阻塞，无锁算法**：算法中仅将 CAS 用于协调线程之间的操作，并且能够正确地实现



**栈**： 一种最简单的链式结构，每个元素仅指向一个元素，且每个元素也被一个元素引用

```java
//构造的非阻塞栈
public class ConcurrentStack<E> {  
    AtomicReference<Node<E>> top = new AtomicReference<Node<E>>(); // 对栈顶的一个引用 

    public void push(E item) {  
        Node<E> newHead = new Node<E>(item);  
        Node<E> oldHead;  
        do {  
            oldHead = top.get();  
            newHead.next = oldHead;  
        } while (!top.compareAndSet(oldHead, newHead));  
    }  

    public E pop() {  
        Node<E> oldHead;  
        Node<E> newHead;  
        do {  
            oldHead = top.get();  
            if (oldHead == null) {  
                return null;  
            }  
            newHead = oldHead.next;  
        } while (!top.compareAndSet(oldHead, newHead));  
        return oldHead.item;  
    }  
    
    private static class Node<E> {  
        final E item;  
        Node<E> next;  
        public Node(E item) {  
            this.item = item;  
        }  
    } 
}
```



```java
//非阻塞链表
public class LinkedQueue<E> {  
    static class Node<E> {  
        final E item;  
        final AtomicReference<Node<E>> next;  

        public Node(E item, Node<E> next) {  
            this.item = item;
            this.next = new AtomicReference<Node<E>>(next);
        }  
    }  

    private final Node<E> dummy = new Node<E>();  
    private final AtomicReference<Node<E>> head = 
        new AtomicReference<Node<E>>(dummy);  
    private final AtomicReference<Node<E>> tail = 
        new AtomicReference<Node<E>>(dummy);  

    public boolean put(E item) {  
        Node<E> newNode = new Node<E>(item, null);  
        while (true) {  
            Node<E> curTailNode = tail.get();  
            Node<E> tailNextNode = curTailNode.next.get();  
            if (curTailNode == tail.get()) {  
                if (tailNextNode == null) {  
                    // 更新尾节点下一个节点  
                    if (curTailNode.next.compareAndSet(null, newNode)) {  
                        // 更新成功, 将尾节点指向下一个节点  
                        tail.compareAndSet(curTailNode, newNode);  
                        return true;  
                    }  
                } else {  
                    // 在更新过程中, 发现尾节点的下一个节点被更新了, 将尾节点指向下一个节点  
                    tail.compareAndSet(curTailNode, tailNextNode);  
                }  
            }  
        }  
    }  

    //测试
    public static void main(String[] args) {  
        final LinkedQueue<String> queue = new LinkedQueue<String>();  
        new Thread(new Runnable() {  
            @Override  
            public void run() {  
                queue.put("item1");  
            }  
        }).start();  
        new Thread(new Runnable() {  
            @Override  
            public void run() {  
                queue.put("item2");  
            }  
        }).start();  
    }  
}
```