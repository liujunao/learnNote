![](../../pics/concurrent/Java并发知识图谱.png)

强烈推荐： 

- ==[Java并发知识点总结](<https://github.com/CL0610/Java-concurrency>)== 
- [死磕Java](<http://cmsblogs.com/?cat=189>) 

# 一、基础知识

## 1. 并发编程的优缺点

### 1. 优点

- 并发编程的形式可以将多核CPU的计算能力发挥到极致，性能得到提升

  > 充分利用多核CPU的计算能力

- 面对复杂业务模型，并行程序会比串行程序更适应业务需求，而并发编程更能吻合这种业务拆分

  > 方便进行业务拆分，提升应用性能

### 2. 缺点

- CPU 不断通过切换线程，让人感觉多个线程同时执行，每次切换时，需要保存当前的状态起来，以便下次能够恢复先前状态，而这个切换非常损耗性能，过于频繁反而无法发挥出多线程编程的优势

  > 时间片是 CPU 分配给各个线程的时间，一般是几十毫秒

**上下文切换的方式**： 

- **无锁并发编程**：参照 concurrentHashMap 分段锁的思想，不同的线程处理不同段的数据，这样在多线程竞争的条件下，可以减少上下文切换的时间

- **CAS 算法**： 参照 Atomic 中的 CAS 算法来更新数据，使用乐观锁，可以有效的减少一部分不必要的锁竞争带来的上下文切换

- **使用最少线程**：避免创建不需要的线程，比如任务很少，但创建了很多的线程，这样会造成大量的线程都处于等待状态

- **协程**：在单线程里实现多任务的调度，并在单线程里维持多个任务间的切换

  > 推荐阅读： [进程，线程，协程与并行，并发](<https://www.jianshu.com/p/f11724034d50>) 与 [线程与协程](<https://www.jianshu.com/p/25cb5a6a17f6>) 
  >
  > - **进程是为了更好的利用 CPU 资源使并发成为可能**
  >
  > - **线程是为了降低上下文切换的消耗，提高系统的并发性，使进程内并发成为可能**
  >
  > - **协程通过在线程中实现调度，避免了陷入内核级别的上下文切换造成的性能损失，进而突破了线程在IO上的性能瓶颈**
  >
  >   > 协程的好处： 
  >   >
  >   > - 切换协程在用户态进行 ，无需系统调用，更快
  >   > - 调用者能够自主控制协程切换，更自由
  >   > - 没有锁的概念，协程安全，不担心死锁等状况，省去锁的开销

### 3. 线程安全

**避免死锁的情况**： 

- 避免一个线程同时获得多个锁
- 避免一个线程在锁内部占有多个资源，尽量保证每个锁只占用一个资源
- 尝试使用定时锁，使用 lock.tryLock(timeOut)，当超时等待时当前线程不会阻塞
- 对于数据库锁，加锁和解锁必须在一个数据库连接里，否则会出现解锁失败的情况

### 4. 相关概念

- **同步和异步**： 

  - 同步调用： 指调用者必须等待被调用的方法结束后，调用者后面的代码才能执行
  - 异步调用： 指调用者不用管被调用方法是否完成，都会继续执行后面的代码，当被调用的方法完成后会通知调用者

- **并发和并行**： 

  - 并发： 指多个任务交替进行
  - 并行： 指真正意义上的“同时进行”

  > - 若系统只有一个 CPU，而使用多线程，则只能通过切换时间片的方式交替进行，并发执行任务
  >
  > - 真正的并行只能出现在拥有多个 CPU 的系统中

- **阻塞和非阻塞**： 通常用来形容多线程间的相互影响

  - 阻塞： 一个线程占有了临界区资源，则其他线程需要等待该资源的释放，进而导致等待的线程挂起
  - 非阻塞： 强调没有一个线程可以阻塞其他线程，所有的线程都会尝试地往前运行

- **临界区**： 用来表示一种公共资源或者说是共享数据，可以被多个线程使用

  >  注意： 一旦临界区资源被一个线程占有，那么其他线程必须等待

## 2. 线程的状态转换

### 1. 线程的创建

```java
public class CreateThreadDemo {
    public static void main(String[] args) {
        //1.继承Thread
        Thread thread = new Thread() {
            @Override
            public void run() {
                System.out.println("继承Thread");
                super.run();
            }
        };
        thread.start();
        
        //2.实现runable接口
        Thread thread1 = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("实现runable接口");
            }
        });
        thread1.start();
        
        //3.实现callable接口
        //实现callable接口，提交给ExecutorService返回的是异步执行的结果
        ExecutorService service = Executors.newSingleThreadExecutor();
        Future<String> future = service.submit(new Callable() {
            @Override
            public String call() throws Exception {
                return "通过实现Callable接口";
            }
        });
        try {
            String result = future.get();
            System.out.println(result);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }
}
```

- 可以利用 `FutureTask(Callable<V> callable)` 将 callable 进行包装然后 FeatureTask 提交给ExecutorsService
- 可以通过 Executors 将 Runable 转换成 Callable，具体方法是：`Callable<T> callable(Runnable task, T result), Callable<Object> callable(Runnable task)` 

### 2. 线程的切换

> - 当线程进入到 synchronized 方法或者 synchronized 代码块时，线程切换到的是BLOCKED状态
> - 使用 java.util.concurrent.locks 下 lock 进行加锁时，线程切换的是 WAITING 或 TIMED_WAITING 状态，因为 lock 会调用 LockSupport 方法

![](../../pics/concurrent/concurrent_1.png)

六种线程状态： 

![](../../pics/concurrent/concurrent_2.png)

### 3. 线程的操作

- `interrupted`： 表示了一个运行中的线程是否被其他线程进行了中断操作

  > 注意： 当抛出 InterruptedException 时，会清除中断标志位，即调用 isInterrupted 会返回 false
  >
  > ![](../../pics/concurrent/concurrent_3.png)
  >
  > 实例：
  >
  > ```java
  > public class InterruptDemo {
  >     public static void main(String[] args) throws InterruptedException {
  >         //sleepThread睡眠1000ms
  >         final Thread sleepThread = new Thread() {
  >             @Override
  >             public void run() {
  >                 try {
  >                     Thread.sleep(1000);
  >                 } catch (InterruptedException e) {
  >                     e.printStackTrace();
  >                 }
  >                 super.run();
  >             }
  >         };
  >         //busyThread一直执行死循环
  >         Thread busyThread = new Thread() {
  >             @Override
  >             public void run() {
  >                 while (true) ;
  >             }
  >         };
  >         
  >         sleepThread.start();
  >         busyThread.start();
  >         //执行中断操作
  >         sleepThread.interrupt();
  >         busyThread.interrupt();
  >         //当返回 false，后续代码才会继续执行
  >         while (sleepThread.isInterrupted()) ;
  >         
  >         System.out.println("sleepThread isInterrupted: " 
  >                            + sleepThread.isInterrupted());
  >         System.out.println("busyThread isInterrupted: " 
  >                            + busyThread.isInterrupted());
  >     }
  > }
  > 
  > //输出
  > sleepThread isInterrupted: false
  > busyThread isInterrupted: true
  > ```

- `join`： 

  > threadB.join()： 当前线程 A 会等待 threadB 线程终止后 threadA 才会继续执行
  >
  > ```java
  > public class JoinDemo {
  >     public static void main(String[] args) {
  >         Thread previousThread = Thread.currentThread();
  >         for (int i = 1; i <= 10; i++) {
  >             Thread curThread = new JoinThread(previousThread);
  >             curThread.start();
  >             previousThread = curThread;
  >         }
  >     }
  > 
  >     static class JoinThread extends Thread {
  >         private Thread thread;
  > 
  >         public JoinThread(Thread thread) {
  >             this.thread = thread;
  >         }
  > 
  >         @Override
  >         public void run() {
  >             try {
  >                 thread.join();
  >                 System.out.println(thread.getName() + " terminated.");
  >             } catch (InterruptedException e) {
  >                 e.printStackTrace();
  >             }
  >         }
  >     }
  > }
  > 
  > //输出： 
  > main terminated.
  > Thread-0 terminated.
  > Thread-1 terminated.
  > Thread-2 terminated.
  > Thread-3 terminated.
  > Thread-4 terminated.
  > Thread-5 terminated.
  > Thread-6 terminated.
  > Thread-7 terminated.
  > Thread-8 terminated.
  > ```

- `sleep`： 

  > sleep  VS  wait： 
  >
  > - sleep() 方法是Thread的静态方法，而 wait() 是Object实例方法
  >
  > - wait() 方法必须在同步方法或同步块中调用，即必须获得对象锁
  >
  >   sleep() 方法可以在任何地方种使用
  >
  > - wait() 方法会释放占有的对象锁，使得该线程进入等待池中，等待下一次获取资源
  > - sleep() 方法只是让出 CPU 并不会释放掉对象锁
  >
  > - sleep() 方法在休眠时间达到后，如果再次获得 CPU 时间片就会继续执行
  > - wait() 方法必须等待 Object.notift/Object.notifyAll 通知后，才会离开等待池，并且再次获得CPU时间片才会继续执行

- `yield`： 使当前线程让出 CPU，让出的时间片只会分配**给当前线程优先级相同或更高**的线程

  > 注意： 让出 CPU 并不代表当前线程不再运行，若在下次竞争中，又获得了CPU 时间片，当前线程依然会继续运行
  >
  > > 在Java程序中，通过一个**整型成员变量Priority**来控制优先级，优先级的范围从1~10.在构建线程的时候可以通过**setPriority(int)**方法进行设置，默认优先级为 5

## 3. 守护线程Daemon

```java
public class DaemonDemo {
    public static void main(String[] args) {
        Thread daemonThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        System.out.println("i am alive");
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        System.out.println("finally block");
                    }
                }
            }
        });
        //设置守护线程要先于start()方法
        daemonThread.setDaemon(true);//设为守护线程
        daemonThread.start();
        //确保main线程结束前能给daemonThread能够分到时间片
        try {
            Thread.sleep(800);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

//输出： 会一直打印,但当 main 线程结束后,daemonThread 就会退出所以不会出现死循环的情况
i am alive
finally block
i am alive
```

> 注意： 守护线程在退出时，并不会执行 finnally 块中的代码

# 二、JMM

- 线程间的通信：**共享内存**和**消息传递** 

  > java内存模型是**共享内存的并发模型**，线程之间主要通过**读-写共享变量**来完成隐式通信

## 1. JMM 内存模型

![](../../pics/concurrent/concurrent_4.png)

线程A和线程B之间完成通信需要经历的步骤： 

- 线程A从主内存中将共享变量读入线程A的工作内存后并进行操作，之后将数据重新写回到主内存中
- 线程B从主存中读取最新的共享变量

## 2. 重排序

编译器和处理器的三种重排序： **1 为编译器重排序； 2，3 为处理器重排序** 

> - 针对编译器重排序，JMM 的编译器重排序规则会**禁止一些特定类型的编译器重排序**
> - 针对处理器重排序，编译器在生成指令序列时会通过**插入内存屏障**来禁止某些特殊的处理器重排序

![](../../pics/concurrent/concurrent_5.png)

- **编译器优化重排序**： 编译器在不改变单线程程序语义的前提下，重新**安排语句的执行顺序**

- **指令级并行重排序**： **将多条指令重叠执行**

  > 如果**不存在数据依赖性**，处理器可以改变语句对应机器指令的执行顺序
  >
  > > 数据依赖性： 若两个操作访问同一个变量，且有一个为写操作，此时这两个操作就存在数据依赖性
  >
  > 注意： 编译器和处理器在重排序时，会遵守数据依赖性，不会改变存在数据依赖性关系的两个操作的执行顺序

- **内存系统重排序**： 由于处理器使用缓存和读/写缓冲区，这使得加载和存储操作看上去可能是在乱序执行的

## 3. happens-before 规则

**happens-before 规则定义**： 

- 如果一个操作 happens-before 另一个操作，那么第一个操作的执行结果将对第二个操作可见，而且第一个操作的执行顺序排在第二个操作之前
- 两个操作之间存在happens-before关系，并不意味着一定要按照happens-before原则制定的顺序来执行。如果重排序之后的执行结果与按照happens-before关系来执行的结果一致，那么这种重排序并不非法

**六项具体规则**： 

- **程序次序规则**：一个线程内，按照代码顺序，书写在前面的操作先行发生于书写在后面的操作

- **锁定规则**：一个 unLock 操作先行发生于后面对同一个锁额 lock 操作

- **volatile 变量规则**：对一个变量的写操作先行发生于后面对这个变量的读操作

- **传递规则**：如果操作A先行发生于操作B，而操作B又先行发生于操作C，则操作A先行发生于操作C

- **线程启动规则**：Thread 对象的 start() 方法先行发生于此线程的每个一个动作

- **线程中断规则**：对线程 interrupt() 方法的调用先行发生于被中断线程的代码检测到中断事件的发生

- **线程终结规则**：线程中所有的操作都先行发生于线程的终止检测

  > 可以通过 Thread.join() 方法结束、Thread.isAlive() 的返回值手段检测到线程已经终止执行

- **对象终结规则**：一个对象的初始化完成先行发生于他的finalize()方法的开始

上述规则的推论：

- 将一个元素放入一个线程安全的队列的操作 Happens-Before 从队列中取出这个元素的操作
- 将一个元素放入一个线程安全容器的操作Happens-Before从容器中取出这个元素的操作
- 在 CountDownLatch 上的倒数操作 Happens-Before CountDownLatch#await()操作
- 释放Semaphore许可的操作Happens-Before获得许可操作
- Future表示的任务的所有操作Happens-Before Future#get()操作
- 向Executor提交一个Runnable或Callable的操作Happens-Before任务开始执行操作

## 4. DCL(Double Check Lock)





# 三、并发关键字

## 1. synchronized

### 1. 实现方式

![](../../pics/concurrent/concurrent_6.png)

### 2. 实现原理

#### 1. Java 对象头

> synchronized 用的锁是存在 Java 对象头中

对象头包括：

- Mark Word(标记字段)： 用于存储对象自身的运行时数据，是实现轻量级锁和偏向锁的关键
- Klass Pointer(类型指针)： 是对象指向它的类元数据的指针，虚拟机通过这个指针来确定对象是哪个类的实例

##### Mark Word

- Java  对象头一般占有两个机器码，但是如果对象是数组类型，则需要三个机器码

  > JVM 虚拟机通过 Java 对象的元数据信息确定 Java 对象的大小，用一块来记录数组长度

  ![](../../pics/concurrent/concurrent_8.png)

- Mark Word 被设计成一个非固定的数据结构以便在极小的空间存储更多数据，它会根据对象的状态复用自己的存储空间

  ![](../../pics/concurrent/concurrent_9.png)

#### 2. monitor

- Monitor 是线程私有的数据结构，每一个线程都有一个可用 monitor record 列表，同时还有一个全局的可用列表

- 每一个被锁住的对象都会和一个monitor关联，同时monitor中有一个Owner字段存放拥有该锁的线程的唯一标识，表示该锁被这个线程占用

  > 对象头的MarkWord中的LockWord指向monitor的起始地址

**monitor 结构**： 

![](../../pics/concurrent/concurrent_10.png)

- **Owner**： 

  - NULL 表示当前没有任何线程拥有该 monitor record
  - 当线程成功拥有该锁后保存线程唯一标识

- **EntryQ**： 关联一个系统互斥锁(semaphore)，阻塞所有试图锁住 monitor record 失败的线程

- **RcThis**： 表示 blocked 或 waiting 在该 monitor record 上的所有线程的个数

- **Nest**： 用来实现重入锁的计数

- **HashCode**： 保存从对象头拷贝过来的 HashCode 值

- **Candidate**： 用来避免不必要的阻塞或等待线程唤醒，因为每一次只有一个线程能够成功拥有锁，如果每次前一个释放锁的线程唤醒所有正在阻塞或等待的线程，会引起不必要的上下文切换从而导致性能严重下降

  > Candidate只有两种可能的值： 
  >
  > - 0 表示没有需要唤醒的线程
  > - 1 表示要唤醒一个继任线程来竞争锁

对象，对象监视器，同步队列以及执行线程状态之间的关系：

> 线程对 Object 的访问，首先要获得 Object 的监视器，如果获取失败，该线程就进入同步状态，线程状态变为BLOCKED，当Object的监视器占有者释放后，在同步队列中得线程就会有机会重新获取该监视器

![](../../pics/concurrent/concurrent_7.png)

### 3. 锁优化

#### 1. CAS 

CAS 比较交换： 包含三个值：**V 内存地址存放的实际值；O 预期的值（旧值）；N 更新的新值**

> - 当 V 和 O 相同时，表明该值没有被其他线程更改过，则可以将新值 N 赋值给 V
> - 若 V 和 O 不相同，表明该值已被其他线程更改，则不能将新值 N 赋给 V，返回 V 即可
>
> 当多个线程使用 CAS 操作一个变量时，只有一个线程会成功，并成功更新，其余会失败。失败的线程会重新尝试，当然也可以选择挂起线程

CAS 问题：

- **ABA 问题**： 旧值 A 变为 B，然后再变成 A，但在做 CAS 时检查发现旧值并没有变化依然为A，但是实际上的确发生了变化

  > 解决方案： 沿袭数据库中常用的乐观锁方式，**添加版本号**

- **自旋时间过长**： 使用 CAS 会发生非阻塞同步，即不会将线程挂起，会自旋进行下一次尝试，自旋时间过长对性能是很大的消耗

- **只能保证一个共享变量的原子操作**： 当对一个共享变量执行操作时CAS能保证其原子性，如果对多个共享变量进行操作，CAS就不能保证其原子性

  > 解决方案： 利用对象整合多个共享变量，即构造一个包含多个共享变量的类，然后将这个对象做CAS操作就可以保证其原子性
  >
  > > atomic中提供了AtomicReference来保证引用对象之间的原子性

#### 2. 自旋锁

- 自旋锁： 通过自旋让该线程等待一段时间，不会被立即挂起，看持有锁的线程是否会很快释放锁

  > 自旋等待的时间(自旋的次数)必须有一个限度，如果超过了定义的时间仍然没有获取到锁，则应该被挂起

#### 3. 适应自旋锁

- 自适应： 即自旋次数不固定，它由前一次在同一个锁上的自旋时间及锁的拥有者的状态来决定

  > - 线程如果自旋成功，则下次自旋的次数会更加多，因为虚拟机认为既然上次成功了，那么此次自旋也很有可能会再次成功，那么它就会允许自旋等待持续的次数更多
  > - 反之，如果对于某个锁，很少有自旋能够成功的，那么在以后要获得这个锁时，自旋的次数会减少甚至省略掉自旋过程，以免浪费处理器资源

#### 4. 锁消除

- **锁消除**： 当 JVM 检测到不可能存在共享数据竞争，则 JVM 会对这些同步锁进行锁消除

- **锁消除依据**： 逃逸分析的数据支持

  > 使用数据流分析来确定数据是否逃逸

#### 5. 锁粗化

- 锁粗化： 将多个连续的加锁、解锁操作连接在一起，扩展成一个范围更大的锁

#### 6. 轻量级锁

**轻量级锁的主要目的**： 在没有多线程竞争的前提下，减少传统的重量级锁使用操作系统互斥量产生的性能消耗

> 当关闭偏向锁功能或者多个线程竞争偏向锁导致偏向锁升级为轻量级锁，则会尝试获取轻量级锁

步骤如下：

- **获取锁**： 
  1. 判断当前对象是否处于无锁状态： 
     - 若是，则 JVM 首先将在当前线程的栈帧中建立一个名为锁记录（Lock Record）的空间，用于存储锁对象目前的 Mark Word 的拷贝（即Displaced Mark Word）
     - 否则执行步骤 3
  2. JVM 利用 CAS 操作尝试将对象的 Mark Word 更新为指向 Lock Record 的指针： 
     - 如果成功表示竞争到锁，则将锁标志位变成00（表示此对象处于轻量级锁状态），执行同步操作
     - 如果失败则执行步骤 3
  3. 判断当前对象的 Mark Word 是否指向当前线程的栈帧：
     - 如果是则表示当前线程已经持有当前对象的锁，则直接执行同步代码块
     - 否则只能说明该锁对象已经被其他线程抢占了，这时轻量级锁需要膨胀为重量级锁，锁标志位变成10，后面等待的线程将会进入阻塞状态
- **释放锁**： 
  1. 取出在获取轻量级锁保存在 Displaced Mark Word 中的数据
  2. 用 CAS 操作将取出的数据替换当前对象的 Mark Word 中：
     - 如果成功，则说明释放锁成功
     - 否则执行 3
  3. 如果 CAS 操作替换失败，说明有其他线程尝试获取该锁，则需要在释放锁的同时唤醒被挂起的线程

![](../../pics/concurrent/concurrent_11.png)

#### 7. 偏向锁

**偏向锁主要目的**：为了在无多线程竞争的情况下尽量减少不必要的轻量级锁执行路径

步骤如下：

- **获取锁**： 

  1. 检测 Mark Word 是否为可偏向状态，即是否为偏向锁，锁标识位为 01
  2. 若为可偏向状态，则测试线程 ID 是否为当前线程 ID：
     - 如果是，则执行步骤 5
     - 否则执行步骤 3
  3. 如果线程 ID 不为当前线程 ID，则通过 CAS 操作竞争锁：
     - 竞争成功，则将 Mark Word 的线程 ID 替换为当前线程 ID
     - 否则执行线程 4
  4. 通过 CAS 竞争锁失败，证明存在多线程竞争，当到达全局安全点，获得偏向锁的线程被挂起，偏向锁升级为轻量级锁，然后被阻塞在安全点的线程继续往下执行同步代码块
  5. 执行同步代码块

- **释放锁**： 只有竞争才会释放锁的机制，线程是不会主动去释放偏向锁，需要等待其他线程来竞争

  > **偏向锁的撤销需要等待全局安全点**(这个时间点上没有正在执行的代码)

  1. 暂停拥有偏向锁的线程，判断锁对象是否还处于被锁定状态
  2. 撤销偏向锁，恢复到无锁状态（01）或者轻量级锁的状态

![](../../pics/concurrent/concurrent_12.png)

#### 8. 重量级锁

重量级锁通过对象内部的监视器（monitor）实现，其中monitor的本质是依赖于底层操作系统的Mutex Lock实现，操作系统实现线程之间的切换需要从用户态到内核态的切换，切换成本非常高

## 2. volatile

### 1. 实现原理

在 volatile 修饰的共享变量进行写操作时，会多出**Lock前缀的指令** 

- Lock前缀的指令会引起处理器缓存写回内存

- 一个处理器的缓存回写到内存会导致其他处理器的缓存失效

- 当处理器发现本地缓存失效后，就会从内存中重读该变量数据，即可以获取当前最新值

  > 缓存一致性： 每个处理器通过嗅探在总线上传播的数据来检查自己缓存的值是不是过期

### 2. 内存语义

- volatile 通过**内存屏障**阻止重排序

JMM内存屏障分为四类： 

![](../../pics/concurrent/concurrent_13.png)

JMM会针对编译器制定volatile重排序规则表：NO 表示禁止重排序

![](../../pics/concurrent/concurrent_14.png)

> JMM 采取的策略：
>
> - 在每个volatile写操作的**前面**插入一个StoreStore屏障
> - 在每个volatile写操作的**后面**插入一个StoreLoad屏障
> - 在每个volatile读操作的**后面**插入一个LoadLoad屏障
> - 在每个volatile读操作的**后面**插入一个LoadStore屏障
>
> 注意：volatile写是在前面和后面**分别插入内存屏障**，而volatile读操作是在**后面插入两个内存屏障** 
>
> - **StoreStore屏障**：禁止上面的普通写和下面的volatile写重排序
>
> - **StoreLoad屏障**：防止上面的volatile写与下面可能有的volatile读/写重排序
>
> - **LoadLoad屏障**：禁止下面所有的普通读操作和上面的volatile读重排序
>
> - **LoadStore屏障**：禁止下面所有的普通写操作和上面的volatile读重排序

![](../../pics/concurrent/concurrent_15.png)

![](../../pics/concurrent/concurrent_16.png)

## 3. final 域重排序

> final 全局变量定义时必须进行初始化且不更更改
>
> final 局部变量，在未初始化时可以进行初始化，但初始化后不能修改

### 1. final 域为基本类型

```java
public class FinalDemo {
    private int a;  //普通域
    private final int b; //final域
    private static FinalDemo finalDemo;

    public FinalDemo() {
        a = 1; // 1. 写普通域
        b = 2; // 2. 写final域
    }

    public static void writer() {
        /**
         * 1. 构造了一个 FinalDemo 对象
         * 2. 把这个对象赋值给成员变量finalDemo
         */
        finalDemo = new FinalDemo();
    }

    public static void reader() {
        FinalDemo demo = finalDemo; // 3.读对象引用
        int a = demo.a;    //4.读普通域
        int b = demo.b;    //5.读final域
    }
}
```

- **写 final 域重排序规则**： 禁止对 final 域的写重排序到构造函数之外

  > - JMM 禁止编译器把 final 域的写重排序到构造函数之外
  > - 编译器会在 final 域写之后，构造函数 return 之前，插入一个 storestore 屏障
  >
  > ![](../../pics/concurrent/concurrent_17.png)
  >
  > - 由于a,b之间没有数据依赖性，普通域 a 可能会被重排序到构造函数之外，线程B就有可能读到的是普通变量a初始化之前的值（零值），这样就可能出现错误
  > - 而 final 域变量 b，根据重排序规则，会禁止 final 修饰的变量 b 重排序到构造函数之外，从而 b 能够正确赋值，线程 B 就能够读到 final 变量初始化后的值
  >
  > **写 final 域的重排序规则可以确保**：在对象引用为任意线程可见之前，对象的 final 域已被正确初始化，而普通域就不具有这个保障

- **读 final 域重排序规则**： 在一个线程中，初次读对象引用和初次读该对象包含的 final 域，JMM会禁止这两个操作的重排序

  > - 处理器会在读final域操作的前面插入一个 LoadLoad 屏障
  >
  > ![](../../pics/concurrent/concurrent_18.png)
  >
  > - 读对象的普通域被重排序到了读对象引用的前面，会出现线程 B 还未读到对象引用就在读取该对象的普通域变量，这显然是错误的操作
  > - 而 final 域的读操作就“限定”了在读final域变量前已经读到了该对象的引用，从而避免这种情况
  >
  > **读 final 域的重排序规则可以确保**：在读一个对象的 final 域之前，一定会先读这个包含这个final域的对象的引用

### 2. final 域为引用类型

```java
public class FinalReferenceDemo {
    final int[] arrays; //final 是引用类型
    private FinalReferenceDemo finalReferenceDemo;

    public FinalReferenceDemo() {
        arrays = new int[1];  //1
        arrays[0] = 1;        //2
    }

    public void writerOne() {
        finalReferenceDemo = new FinalReferenceDemo(); //3
    }

    public void writerTwo() {
        arrays[0] = 2;  //4
    }

    public void reader() {
        if (finalReferenceDemo != null) {  //5
            int temp = finalReferenceDemo.arrays[0];  //6
        }
    }
}
```

- **对 final 修饰的对象的成员域写操作**： 在构造函数内对**一个final修饰的对象的成员域的写入，与随后在构造函数之外把这个被构造的对象的引用赋给一个引用变量**，这两个操作是不能被重排序的

  > 线程线程 A 执行 wirterOne 方法，执行完后线程 B 执行 writerTwo 方法，然后线程 C 执行 reader 方法
  >
  > ![](../../pics/concurrent/concurrent_19.png)

- **对 final 修饰的对象的成员域读操作**： 

  > - JMM 可以确保： 线程 C 至少能看到写线程 A 对 final 引用的对象的成员域的写入，即能看到 `arrays[0] = 1`，而写线程 B 对数组元素的写入可能看到可能看不到
  >
  > - JMM不保证线程 B 的写入对线程 C 可见，线程 B 和线程 C 之间存在数据竞争，此时的结果是不可预知的。如果可见的，可使用锁或者 volatile

**final 域总结**： 

- **基本数据类型**:

  - **final 域写**：禁止**final域写**与**构造方法**重排序，即禁止final域写重排序到构造方法之外，从而保证该对象对所有线程可见时，该对象的final域全部已经初始化过

  - **final 域读**：禁止初次**读对象的引用**与**读该对象包含的final域**的重排序

- **引用数据类型**：

  额外增加约束：禁止在构造函数对**一个final修饰的对象的成员域的写入**与随后将**这个被构造的对象的引用赋值给引用变量** 重排序

## 4. 三大特性

### 1. 原子性

- **原子性**： 指一个操作是不可中断的，要么全部执行成功要么全部执行失败

- JMM 定义的八种原子操作：
  - **lock(锁定)**：作用于主内存中的变量，把一个变量标识为一个线程独占的状态
  - **unlock(解锁)**： 作用于主内存中的变量，把一个处于锁定状态的变量释放出来，释放后的变量才可以被其他线程锁定
  - **read(读取)**：作用于主内存的变量，把一个变量的值从主内存传输到线程的工作内存中，以便后面的load动作使用
  - **load(载入)**：作用于工作内存的变量，把 read 操作从主内存中得到的变量值放入工作内存中的变量副本
  - **use(使用)**：作用于工作内存的变量，把工作内存中一个变量的值传递给执行引擎，每当虚拟机遇到一个需要使用到变量的值的字节码指令时将会执行这个操作
  - **assign(赋值)**：作用于工作内存的变量，把一个从执行引擎接收到的值赋给工作内存的变量，每当虚拟机遇到一个给变量赋值的字节码指令时执行这个操作
  - **store(存储)**：作用于工作内存的变量，把工作内存中一个变量的值传送给主内存中以便随后的 write 操作使用
  - **write(操作)**：作用于主内存的变量，把 store 操作从工作内存中得到的变量的值放入主内存的变量中

### 2. 有序性

- **有序性**： 在本线程内观察，所有的操作都是有序的；在一个线程观察另一个线程，所有的操作都是无序的

### 3. 可见性

- **可见性**： 指当一个线程修改了共享变量后，其他线程能够立即得知这个修改

# 四、Lock 体系

![](../../pics/concurrent/concurrent_20.png)

## 1. Lock 与 synchronized 比较





## 2. AQS(AbstractQueuedSynchronizer)

### 1. 基本方法

- AQS 可重写的方法： 

  ![](../../pics/concurrent/concurrent_21.png)

- 在实现同步组件时 AQS 提供的模板方法： 

  ![](../../pics/concurrent/concurrent_22.png)

  > AQS 提供的模板方法可以分为 3 类：
  >
  > - **独占式**获取与释放同步状态
  >
  > - **共享式**获取与释放同步状态
  >
  > - 查询同步队列中等待线程情况

### 2. CLH 同步队列

> AQS 内部维护着一个同步队列，即 CLH 同步队列： 
>
> - CLH 同步队列是一个 FIFO 双向队列，AQS 依赖它来完成同步状态的管理
> - 当前线程如果获取同步状态失败，AQS 则会将当前线程已经等待状态等信息构造成一个节点（Node）并将其加入到CLH同步队列，同时会阻塞当前线程，当同步状态释放时，会把首节点唤醒（公平锁），使其再次尝试获取同步状态

![](../../pics/concurrent/concurrent_23.png)

### 3. 同步状态的获取与释放

#### 1. 独占式

- **独占锁获取(acquire 方法)**： AQS 提供的模板方法，独占式获取同步状态，但对中断不敏感，即由于线程获取同步状态失败加入到 CLH 同步队列中，后续对线程进行中断操作时，线程不会从同步队列中移除

  > ```java
  > public final void acquire(int arg) {
  >     if (!tryAcquire(arg) && acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
  >         selfInterrupt();
  > }
  > ```
  >
  > - `tryAcquire`：去尝试获取锁，获取成功则设置锁状态并返回 true，否则返回 false
  >
  >   > 该方法自定义同步组件自己实现，必须要保证线程安全的获取同步状态
  >
  > - `addWaiter`：如果 tryAcquire 返回 FALSE（获取同步状态失败），则调用该方法将当前线程加入到CLH同步队列尾部
  >
  > - `acquireQueued`：当前线程会根据公平性原则来进行阻塞等待（自旋）,直到获取锁为止；并且返回当前线程在等待过程中有没有中断过
  >
  > - `selfInterrupt`：产生一个中断
  >
  > acquire 方法的执行流程： 
  >
  > ![](../../pics/concurrent/concurrent_24.png)

- **可中断式获取锁(acquireInterruptibly 方法)**： 该方法在等待获取同步状态时，如果当前线程被中断了，会立刻**响应中断**抛出异常 InterruptedException

  > ```java
  > public final void acquireInterruptibly(int arg) throws InterruptedException {
  >     if (Thread.interrupted()) //校验该线程是否已经中断
  >         throw new InterruptedException(); //如果是则抛出InterruptedException
  >     if (!tryAcquire(arg))//执行tryAcquire(int arg)方法获取同步状态
  >         //如果获取成功，则直接返回，否则执行 doAcquireInterruptibly(int arg)
  >         doAcquireInterruptibly(arg);
  > }
  > 
  > private void doAcquireInterruptibly(int arg) throws InterruptedException {
  > 	//将节点插入到同步队列中
  >     final Node node = addWaiter(Node.EXCLUSIVE);
  >     boolean failed = true;
  >     try {
  >         for (;;) {
  >             final Node p = node.predecessor();
  >             //获取锁出队
  > 			if (p == head && tryAcquire(arg)) {
  >                 setHead(node);
  >                 p.next = null; // help GC
  >                 failed = false;
  >                 return;
  >             }
  >             if (shouldParkAfterFailedAcquire(p, node) &&
  >                 parkAndCheckInterrupt())
  > 				//线程中断抛异常
  >                 throw new InterruptedException();
  >         }
  >     } finally {
  >         if (failed)
  >             cancelAcquire(node);
  >     }
  > }
  > ```

- **超时等待获取锁(tryAcquireNanos 方法)**： 除了响应中断外，还有超时控制，即如果当前线程没有在指定时间内获取同步状态，则会返回false，否则返回true

  > ```java
  > public final boolean tryAcquireNanos(int arg, long nanosTimeout)
  >             throws InterruptedException {
  >     if (Thread.interrupted())
  >         throw new InterruptedException();
  >     return tryAcquire(arg) || doAcquireNanos(arg, nanosTimeout);
  > }
  > 
  > private boolean doAcquireNanos(int arg, long nanosTimeout)
  >         throws InterruptedException {
  >     if (nanosTimeout <= 0L)
  >         return false;
  > 	//1. 根据超时时间和当前时间计算出截止时间
  >     final long deadline = System.nanoTime() + nanosTimeout;
  >     final Node node = addWaiter(Node.EXCLUSIVE);
  >     boolean failed = true;
  >     try {
  >         for (;;) {
  >             final Node p = node.predecessor();
  > 			//2. 当前线程获得锁出队列
  >             if (p == head && tryAcquire(arg)) {
  >                 setHead(node);
  >                 p.next = null; // help GC
  >                 failed = false;
  >                 return true;
  >             }
  > 			// 3.1 重新计算超时时间
  >             nanosTimeout = deadline - System.nanoTime();
  >             // 3.2 已经超时返回false
  > 			if (nanosTimeout <= 0L)
  >                 return false;
  > 			// 3.3 线程阻塞等待 
  >             if (shouldParkAfterFailedAcquire(p, node) &&
  >                 	nanosTimeout > spinForTimeoutThreshold)
  >                 LockSupport.parkNanos(this, nanosTimeout);
  >             // 3.4 线程被中断抛出被中断异常
  > 			if (Thread.interrupted())
  >                 throw new InterruptedException();
  >         }
  >     } finally {
  >         if (failed)
  >             cancelAcquire(node);
  >     }
  > }
  > ```
  >
  > ![](../../pics/concurrent/concurrent_25.png)

- **独占锁释放(release 方法)**： 

  > ```java
  > public final boolean release(int arg) {
  >     if (tryRelease(arg)) { //如果同步状态释放成功
  >         Node h = head;
  >         //当head指向的头结点不为null 且 该节点的状态值不为 0
  >         if (h != null && h.waitStatus != 0)
  >             unparkSuccessor(h); //执行unparkSuccessor()方法
  >         return true;
  >     }
  >     return false;
  > }
  > 
  > private void unparkSuccessor(Node node) {
  >     int ws = node.waitStatus;
  >     if (ws < 0)
  >         compareAndSetWaitStatus(node, ws, 0);
  > 
  > 	//头节点的后继节点
  >     Node s = node.next;
  >     if (s == null || s.waitStatus > 0) {
  >         s = null;
  >         for (Node t = tail; t != null && t != node; t = t.prev)
  >             if (t.waitStatus <= 0)
  >                 s = t;
  >     }
  >     if (s != null)
  > 		//后继节点不为null时唤醒该线程
  >         LockSupport.unpark(s.thread);
  > }
  > ```

总结： 

- 线程获取锁失败，线程被封装成 Node 进行入队操作，核心方法在于 addWaiter() 和 enq()，同时 enq() 完成对同步队列的头结点初始化工作以及 CAS 操作失败的重试

- 线程获取锁是一个自旋的过程，当且仅当当前节点的前驱节点是头结点并且成功获得同步状态时，节点出队即该节点引用的线程获得锁，否则，当不满足条件时就会调用 LookSupport.park() 方法使得线程阻塞

- 释放锁的时候会唤醒后继节点

  > **总的来说**： 
  >
  > - 在获取同步状态时，AQS 维护一个同步队列，获取同步状态失败的线程会加入到队列中进行自旋
  > - 移除队列（或停止自旋）的条件是前驱节点是头结点并且成功获得了同步状态
  > - 在释放同步状态时，同步器会调用 unparkSuccessor() 方法唤醒后继节点

#### 2. 共享式

- **共享式锁的获取(acquireShared 方法)**： 

  > ```java
  > public final void acquireShared(int arg) {
  >     if (tryAcquireShared(arg) < 0)
  >         doAcquireShared(arg); ////获取失败，自旋获取同步状态
  > }
  > 
  > private void doAcquireShared(int arg) {
  >     //共享式节点
  >     final Node node = addWaiter(Node.SHARED);
  >     boolean failed = true;
  >     try {
  >         boolean interrupted = false;
  >         for (;;) {
  >             //前驱节点
  >             final Node p = node.predecessor();
  >             //如果其前驱节点，获取同步状态
  >             if (p == head) {
  >                 //尝试获取同步
  >                 int r = tryAcquireShared(arg);
  >                 if (r >= 0) {
  >                     setHeadAndPropagate(node, r);
  >                     p.next = null; // help GC
  >                     if (interrupted)
  >                         selfInterrupt();
  >                     failed = false;
  >                     return;
  >                 }
  >             }
  >             if (shouldParkAfterFailedAcquire(p, node) &&
  >                     parkAndCheckInterrupt())
  >                 interrupted = true;
  >         }
  >     } finally {
  >         if (failed)
  >             cancelAcquire(node);
  >     }
  > }
  > ```
  >
  > **自旋退出的条件**： 当前节点的前驱节点是头结点并且 tryAcquireShared(arg) 返回值 >=0 即能成功获得同步状态

- **可中断(acquireSharedInterruptibly 方法)**： 

- **超时等待(tryAcquireSharedNanos 方法)**： 

- **共享式锁的释放(releaseShared 方法)**： 

  > ```java
  > public final boolean releaseShared(int arg) {
  >     if (tryReleaseShared(arg)) {
  >         doReleaseShared();
  >         return true;
  >     }
  >     return false;
  > }
  > 
  > private void doReleaseShared() {
  >     for (;;) {
  >         Node h = head;
  >         if (h != null && h != tail) {
  >             int ws = h.waitStatus;
  >             if (ws == Node.SIGNAL) {
  >                 if (!compareAndSetWaitStatus(h, Node.SIGNAL, 0))
  >                     continue;            // loop to recheck cases
  >                 unparkSuccessor(h);
  >             }
  >             else if (ws == 0 &&
  >                      !compareAndSetWaitStatus(h, 0, Node.PROPAGATE))
  >                 continue;                // loop on failed CAS
  >         }
  >         if (h == head)                   // loop if head changed
  >             break;
  >     }
  > }
  > ```

## 3. ReentrantLock 可重入锁

- **ReentrantLock 可重入锁**： 是一种递归无阻塞的同步机制。提供了比 synchronized 更强大、灵活的锁机制，可以减少死锁发生的概率

### 1. 获取与释放锁

- **获取锁**： 

  > **非公平锁**： 
  >
  > ```java
  > public void lock() {
  >     sync.lock();
  > }
  > 
  > final void lock() {
  >     //首先会第一次尝试快速获取锁，如果获取失败，则调用acquire(int arg)方法
  >     if (compareAndSetState(0, 1))
  >         setExclusiveOwnerThread(Thread.currentThread());
  >     else
  >         acquire(1); //获取失败，调用AQS的acquire(int arg)方法
  > }
  > 
  > public final void acquire(int arg) {
  >     if (!tryAcquire(arg) && acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
  >         selfInterrupt();
  > }
  > 
  > protected final boolean tryAcquire(int acquires) {
  >     return nonfairTryAcquire(acquires);
  > }
  > 
  > final boolean nonfairTryAcquire(int acquires) {
  >     //当前线程
  >     final Thread current = Thread.currentThread();
  >     //获取同步状态
  >     int c = getState();
  >     //state == 0,表示没有该锁处于空闲状态
  >     if (c == 0) {
  >         //获取锁成功，设置为当前线程所有
  >         if (compareAndSetState(0, acquires)) {
  >             setExclusiveOwnerThread(current);
  >             return true;
  >         }
  >     }
  >     //线程重入
  >     //判断锁持有的线程是否为当前线程
  >     else if (current == getExclusiveOwnerThread()) {
  >         int nextc = c + acquires;
  >         if (nextc < 0) // overflow
  >             throw new Error("Maximum lock count exceeded");
  >         setState(nextc);
  >         return true;
  >     }
  >     return false;
  > }
  > ```

- **释放锁**： 

  > ```java
  > public void unlock() {
  >     sync.release(1);
  > }
  > 
  > public final boolean release(int arg) {
  >     if (tryRelease(arg)) {
  >         Node h = head;
  >         if (h != null && h.waitStatus != 0)
  >             unparkSuccessor(h);
  >         return true;
  >     }
  >     return false;
  > }
  > 
  > protected final boolean tryRelease(int releases) {
  >     //减掉releases
  >     int c = getState() - releases;
  >     //如果释放的不是持有锁的线程，抛出异常
  >     if (Thread.currentThread() != getExclusiveOwnerThread())
  >         throw new IllegalMonitorStateException();
  >     boolean free = false;
  >     //state == 0 表示已经释放完全了，其他线程可以获取同步状态了
  >     if (c == 0) {
  >         free = true;
  >         setExclusiveOwnerThread(null);
  >     }
  >     setState(c);
  >     return free;
  > }
  > ```

### 2. 公平锁与非公平锁

- 公平锁与非公平锁的区别在于获取锁的时候**是否按照 FIFO 的顺序来**

公平锁的 tryAcquire 方法： 

```java
protected final boolean tryAcquire(int acquires) {
    final Thread current = Thread.currentThread();
    int c = getState();
    if (c == 0) {
        //公平锁较非公平锁多了 hasQueuedPredecessors() 方法
        if (!hasQueuedPredecessors() && compareAndSetState(0, acquires)) {
            setExclusiveOwnerThread(current);
            return true;
        }
    } else if (current == getExclusiveOwnerThread()) {
        int nextc = c + acquires;
        if (nextc < 0)
            throw new Error("Maximum lock count exceeded");
        setState(nextc);
        return true;
    }
    return false;
}

//用于判断当前线程是否位于 CLH 同步队列中的第一个： 如果是则返回 true，否则返回 false
public final boolean hasQueuedPredecessors() {
    Node t = tail;  //尾节点
    Node h = head;  //头节点
    Node s;

    //头节点 != 尾节点
    //同步队列第一个节点不为null
    //当前线程是同步队列第一个节点
    return h != t && ((s = h.next) == null || s.thread != Thread.currentThread());
}
```

> ReentrantLock 与 synchronized的区别： 
>
> - 与 synchronized 相比，ReentrantLock 提供了更多，更加全面的功能，具备更强的扩展性
>
>   > 例如：时间锁等候，可中断锁等候，锁投票
>
> - ReentrantLock 还提供了条件 Condition，对线程的等待、唤醒操作更加详细和灵活，所以在多个条件变量和高度竞争锁的地方，ReentrantLock 更加适合
>
> - ReentrantLock 提供了可轮询的锁请求，会尝试着去获取锁，如果成功则继续，否则等到下次运行时处理，而 synchronized 则一旦进入锁请求要么成功要么阻塞
>
> - ReentrantLock 支持更加灵活的同步代码块，但是使用 synchronized 时，只能在同一个 synchronized块结构中获取和释放
>
>   > 注：ReentrantLock 的锁释放一定要在 finally 中处理，否则可能会产生严重的后果
>
> - ReentrantLock 支持中断处理，且性能较 synchronized 会好些

## 4. ReentrantReadWriteLock 读写锁

### 1. 简介

- 读写锁维护着一对锁，一个读锁和一个写锁。通过分离读锁和写锁，使得并发性比一般的排他锁有了较大的提升：在同一时间可以允许多个读线程同时访问，但是在写线程访问时，所有读线程和写线程都会被阻塞

- **读写锁的主要特性**： 

  - **公平性**：支持公平性和非公平性

  - **重入性**：读写锁最多支持 65535 个递归写入锁和 65535 个递归读取锁

    > - 在 ReentrantLock 中使用一个 int 类型的 state 来表示同步状态，该值表示锁被一个线程重复获取的次数
    >
    > - 读写锁 ReentrantReadWriteLock 内部维护着两个一对锁，需要用一个变量维护多种状态
    >
    >   > 读写锁采用“**按位切割使用**”的方式来维护这个变量，将其切分为两部分，**高16为表示读，低16为表示写**

  - **锁降级**：遵循获取写锁、获取读锁在释放写锁的次序，**写锁能够降级成为读锁** 

```java
/** 内部类  读锁 */
private final ReentrantReadWriteLock.ReadLock readerLock;
/** 内部类  写锁 */
private final ReentrantReadWriteLock.WriteLock writerLock;

final Sync sync;

/** 使用默认（非公平）的排序属性创建一个新的 ReentrantReadWriteLock */
public ReentrantReadWriteLock() {
    this(false);
}

/** 使用给定的公平策略创建一个新的 ReentrantReadWriteLock */
public ReentrantReadWriteLock(boolean fair) {
    sync = fair ? new FairSync() : new NonfairSync();
    readerLock = new ReadLock(this);
    writerLock = new WriteLock(this);
}

/** 返回用于写入操作的锁 */
public ReentrantReadWriteLock.WriteLock writeLock() { return writerLock; }
/** 返回用于读取操作的锁 */
public ReentrantReadWriteLock.ReadLock  readLock()  { return readerLock; }

abstract static class Sync extends AbstractQueuedSynchronizer {
    /**
     * 省略其余源代码
     */
}
public static class WriteLock implements Lock, java.io.Serializable{
    /**
     * 省略其余源代码
     */
}

public static class ReadLock implements Lock, java.io.Serializable {
    /**
     * 省略其余源代码
     */
}
```

### 2. 写锁

- **写锁的获取**： 终会调用tryAcquire(int arg)

  > ```java
  > protected final boolean tryAcquire(int acquires) {
  >     Thread current = Thread.currentThread();
  >     int c = getState(); //当前锁个数
  >     int w = exclusiveCount(c); //写锁
  >     if (c != 0) {
  >         //c != 0 && w == 0 表示存在读锁(判断读锁是否存在)
  >         //当前线程不是已经获取写锁的线程
  >         if (w == 0 || current != getExclusiveOwnerThread())
  >             return false;
  >         //超出最大范围
  >         if (w + exclusiveCount(acquires) > MAX_COUNT)
  >             throw new Error("Maximum lock count exceeded");
  >         setState(c + acquires);
  >         return true;
  >     }
  >     //是否需要阻塞
  >     if (writerShouldBlock() || !compareAndSetState(c, c + acquires))
  >         return false;
  >     //设置获取锁的线程为当前线程
  >     setExclusiveOwnerThread(current);
  >     return true;
  > }
  > ```

- **写锁的释放**： 

  > ```java
  > public void unlock() {
  >     sync.release(1);
  > }
  > 
  > public final boolean release(int arg) {
  >     if (tryRelease(arg)) {
  >         Node h = head;
  >         if (h != null && h.waitStatus != 0)
  >             unparkSuccessor(h);
  >         return true;
  >     }
  >     return false;
  > }
  > 
  > protected final boolean tryRelease(int releases) {
  >     //释放的线程不为锁的持有者
  >     if (!isHeldExclusively())
  >         throw new IllegalMonitorStateException();
  >     int nextc = getState() - releases;
  >     //若写锁的新线程数为0，则将锁的持有者设置为null
  >     boolean free = exclusiveCount(nextc) == 0;
  >     if (free)
  >         setExclusiveOwnerThread(null);
  >     setState(nextc);
  >     return free;
  > }
  > ```

### 3. 读锁

> 读锁为一个可重入的共享锁，它能够被多个线程同时持有，在没有其他写线程访问时，读锁总是或获取成功

- **读锁的获取**： 

  > ```java
  > public void lock() {
  >     sync.acquireShared(1);
  > }
  > 
  > public final void acquireShared(int arg) {
  >     if (tryAcquireShared(arg) < 0)
  >         doAcquireShared(arg);
  > }
  > 
  > //用于获取共享式同步状态
  > /** 执行流程： 
  >  * 1. 因为存在锁降级情况，如果存在写锁且锁的持有者不是当前线程则直接返回失败，否则继续
  >  * 2. 依据公平性原则，判断读锁是否需要阻塞： 
  >  *   - 读锁持有线程数小于最大值（65535），且设置锁状态成功，执行以下代码，并返回1
  >  *   - 如果不满足改条件，执行fullTryAcquireShared()
  >  */
  > protected final int tryAcquireShared(int unused) {
  >     //当前线程
  >     Thread current = Thread.currentThread();
  >     int c = getState();
  >     //exclusiveCount(c)计算写锁
  >     //如果存在写锁，且锁的持有者不是当前线程，直接返回-1
  >     //存在锁降级问题，后续阐述
  >     if (exclusiveCount(c) != 0 && getExclusiveOwnerThread() != current)
  >         return -1;
  >     //读锁
  >     int r = sharedCount(c);
  > 
  >     /*
  >      * readerShouldBlock():读锁是否需要等待（公平锁原则）
  >      * r < MAX_COUNT：持有线程小于最大数（65535）
  >      * compareAndSetState(c, c + SHARED_UNIT)：设置读取锁状态
  >      */
  >     if (!readerShouldBlock() && r < MAX_COUNT &&
  >             compareAndSetState(c, c + SHARED_UNIT)) {
  >         /*
  >          * holdCount部分后面讲解
  >          */
  >         if (r == 0) {
  >             firstReader = current;
  >             firstReaderHoldCount = 1;
  >         } else if (firstReader == current) {
  >             firstReaderHoldCount++;
  >         } else {
  >             HoldCounter rh = cachedHoldCounter;
  >             if (rh == null || rh.tid != getThreadId(current))
  >                 cachedHoldCounter = rh = readHolds.get();
  >             else if (rh.count == 0)
  >                 readHolds.set(rh);
  >             rh.count++;
  >         }
  >         return 1;
  >     }
  >     return fullTryAcquireShared(current);
  > }
  > 
  > final int fullTryAcquireShared(Thread current) {
  >     HoldCounter rh = null;
  >     for (;;) {
  >         int c = getState();
  >         //锁降级
  >         if (exclusiveCount(c) != 0) {
  >             if (getExclusiveOwnerThread() != current)
  >                 return -1;
  >         }
  >         //读锁需要阻塞
  >         else if (readerShouldBlock()) {
  >             //列头为当前线程
  >             if (firstReader == current) {
  >             }
  >             //HoldCounter后面讲解
  >             else {
  >                 if (rh == null) {
  >                     rh = cachedHoldCounter;
  >                     if (rh == null || rh.tid != getThreadId(current)) {
  >                         rh = readHolds.get();
  >                         if (rh.count == 0)
  >                             readHolds.remove();
  >                     }
  >                 }
  >                 if (rh.count == 0)
  >                     return -1;
  >             }
  >         }
  >         //读锁超出最大范围
  >         if (sharedCount(c) == MAX_COUNT)
  >             throw new Error("Maximum lock count exceeded");
  >         //CAS设置读锁成功
  >         if (compareAndSetState(c, c + SHARED_UNIT)) {
  >             //如果是第1次获取“读取锁”，则更新firstReader和firstReaderHoldCount
  >             if (sharedCount(c) == 0) {
  >                 firstReader = current;
  >                 firstReaderHoldCount = 1;
  >             }
  >             //如果想要获取锁的线程(current)是第1个获取锁(firstReader)的线程
  >             //则将firstReaderHoldCount+1
  >             else if (firstReader == current) {
  >                 firstReaderHoldCount++;
  >             } else {
  >                 if (rh == null)
  >                     rh = cachedHoldCounter;
  >                 if (rh == null || rh.tid != getThreadId(current))
  >                     rh = readHolds.get();
  >                 else if (rh.count == 0)
  >                     readHolds.set(rh);
  >                 //更新线程的获取“读取锁”的共享计数
  >                 rh.count++;
  >                 cachedHoldCounter = rh; // cache for release
  >             }
  >             return 1;
  >         }
  >     }
  > }
  > ```

- **读锁的释放**： 

  > ```java
  > public void unlock() {
  >     sync.releaseShared(1);
  > }
  > 
  > public final boolean releaseShared(int arg) {
  >     if (tryReleaseShared(arg)) {
  >         doReleaseShared();
  >         return true;
  >     }
  >     return false;
  > }
  > 
  > protected final boolean tryReleaseShared(int unused) {
  >     Thread current = Thread.currentThread();
  >     //如果想要释放锁的线程为第一个获取锁的线程
  >     if (firstReader == current) {
  >         //仅获取了一次，则需要将firstReader 设置null，否则 firstReaderHoldCount - 1
  >         if (firstReaderHoldCount == 1)
  >             firstReader = null;
  >         else
  >             firstReaderHoldCount--;
  >     }
  >     //获取rh对象，并更新“当前线程获取锁的信息”
  >     else {
  >         HoldCounter rh = cachedHoldCounter;
  >         if (rh == null || rh.tid != getThreadId(current))
  >             rh = readHolds.get();
  >         int count = rh.count;
  >         if (count <= 1) {
  >             readHolds.remove();
  >             if (count <= 0)
  >                 throw unmatchedUnlockException();
  >         }
  >         --rh.count;
  >     }
  >     //CAS更新同步状态
  >     for (;;) {
  >         int c = getState();
  >         int nextc = c - SHARED_UNIT;
  >         if (compareAndSetState(c, nextc))
  >             return nextc == 0;
  >     }
  > }
  > ```

## 5. Condition 机制

### 1. 简介

![](../../pics/concurrent/concurrent_26.png)

Contition 提供的方法： 

- **await()** ：造成当前线程在接到信号或被中断之前一直处于等待状态

- **await(long time, TimeUnit unit) **：造成当前线程在接到信号、被中断或到达指定等待时间之前一直处于等待状态

- **awaitNanos(long nanosTimeout) **：造成当前线程在接到信号、被中断或到达指定等待时间之前一直处于等待状态

  > 返回值表示剩余时间，如果在 nanosTimesout 之前唤醒，那么返回值 = nanosTimeout – 消耗时间
  >
  > 如果返回值 <= 0，则可以认定它已超时

- **awaitUninterruptibly() **：造成当前线程在接到信号之前一直处于等待状态

  > 该方法对中断不敏感

- **awaitUntil(Date deadline) **：造成当前线程在接到信号、被中断或到达指定最后期限前一直处于等待状态

  > 如果没有到指定时间就被通知，则返回true，否则表示到了指定时间，返回返回false

- **signal()**：唤醒一个等待线程，该线程从等待方法返回前必须获得与Condition相关的锁

- **signal()All**：唤醒所有等待线程，能够从等待方法返回的线程必须获得与 Condition 相关的锁

### 2. 实现原理

- **等待队列**： 对象 Object 对象监视器上只能拥有一个同步队列和一个等待队列，而并发包中的 Lock 拥有一个同步队列和多个等待队列

  > ![](../../pics/concurrent/concurrent_27.png)

- `await()`： 调用 Condition 的 await() 方法会使当前线程进入等待状态，同时会加入到 Condition 等待队列同时释放锁

  > 当从 await() 方法返回时，当前线程一定是获取了 Condition 相关连的锁
  >
  > ```java
  > /**实现逻辑： 
  >  * 首先将当前线程新建一个节点同时加入到条件队列中，然后释放当前线程持有的同步状态
  >  * 然后则是不断检测该节点代表的线程是否出现在 CLH 同步队列中
  >  *（收到signal信号之后就会在AQS队列中检测到），如果不存在则一直挂起，否则参与竞争同步状态
  >  */
  > public final void await() throws InterruptedException {
  >     // 当前线程中断
  >     if (Thread.interrupted())
  >         throw new InterruptedException();
  >     //当前线程加入等待队列
  >     Node node = addConditionWaiter();
  >     //释放锁
  >     long savedState = fullyRelease(node);
  >     int interruptMode = 0;
  >     /**
  >      * 检测此节点的线程是否在同步队上，如果不在，则说明该线程还不具备竞争锁的资格
  >      * 则继续等待直到检测到此节点在同步队列上
  >      */
  >     while (!isOnSyncQueue(node)) {
  >         //线程挂起
  >         LockSupport.park(this);
  >         //如果已经中断了，则退出
  >         if ((interruptMode = checkInterruptWhileWaiting(node)) != 0)
  >             break;
  >     }
  >     //竞争同步状态
  >     if (acquireQueued(node, savedState) && interruptMode != THROW_IE)
  >         interruptMode = REINTERRUPT;
  >     //清理下条件队列中的不是在等待条件的节点
  >     if (node.nextWaiter != null) // clean up if cancelled
  >         unlinkCancelledWaiters();
  >     if (interruptMode != 0)
  >         reportInterruptAfterWait(interruptMode);
  > }
  > //将当前线程加入到 Condition 条件队列中
  > //当然在加入到尾节点之前会清楚所有状态不为 Condition 的节点
  > private Node addConditionWaiter() {
  >     Node t = lastWaiter;    //尾节点
  >     //Node的节点状态如果不为CONDITION，则表示该节点不处于等待状态，需要清除节点
  >     if (t != null && t.waitStatus != Node.CONDITION) {
  >         //清除条件队列中所有状态不为Condition的节点
  >         unlinkCancelledWaiters();
  >         t = lastWaiter;
  >     }
  >     //当前线程新建节点，状态CONDITION
  >     Node node = new Node(Thread.currentThread(), Node.CONDITION);
  >     /**
  >      * 将该节点加入到条件队列中最后一个位置
  >      */
  >     if (t == null)
  >         firstWaiter = node;
  >     else
  >         t.nextWaiter = node;
  >     lastWaiter = node;
  >     return node;
  > }
  > 
  > //释放该线程持有的锁
  > final long fullyRelease(Node node) {
  >     boolean failed = true;
  >     try {
  >         //节点状态--其实就是持有锁的数量
  >         long savedState = getState();
  >         //释放锁
  >         if (release(savedState)) {
  >             failed = false;
  >             return savedState;
  >         } else {
  >             throw new IllegalMonitorStateException();
  >         }
  >     } finally {
  >         if (failed)
  >             node.waitStatus = Node.CANCELLED;
  >     }
  > }
  > 
  > //如果一个节点刚开始在条件队列上，现在在同步队列上获取锁则返回 true
  > final boolean isOnSyncQueue(Node node) {
  >     //状态为Condition，获取前驱节点为null，返回false
  >     if (node.waitStatus == Node.CONDITION || node.prev == null)
  >         return false;
  >     //后继节点不为null，肯定在CLH同步队列中
  >     if (node.next != null)
  >         return true;
  >     return findNodeFromTail(node);
  > }
  > 
  > //负责将条件队列中状态不为 Condition 的节点删除
  > private void unlinkCancelledWaiters() {
  >     Node t = firstWaiter;
  >     Node trail = null;
  >     while (t != null) {
  >         Node next = t.nextWaiter;
  >         if (t.waitStatus != Node.CONDITION) {
  >             t.nextWaiter = null;
  >             if (trail == null)
  >                 firstWaiter = next;
  >             else
  >                 trail.nextWaiter = next;
  >             if (next == null)
  >                 lastWaiter = trail;
  >         }
  >         else
  >             trail = t;
  >         t = next;
  >     }
  > }
  > ```

- `signal/signalAll`： 会唤醒在等待队列中等待最长时间的节点（条件队列里的首节点），在唤醒节点前，会将节点移到CLH同步队列中

  > ```java
  > public final void signal() {
  >     //检测当前线程是否为拥有锁的独
  >     if (!isHeldExclusively())
  >         throw new IllegalMonitorStateException();
  >     //头节点，唤醒条件队列中的第一个节点
  >     Node first = firstWaiter;
  >     if (first != null)
  >         doSignal(first);    //唤醒
  > }
  > //唤醒头节点
  > private void doSignal(Node first) {
  >     do {
  >         //修改头结点，完成旧头结点的移出工作
  >         if ( (firstWaiter = first.nextWaiter) == null)
  >             lastWaiter = null;
  >         first.nextWaiter = null;
  >     } while (!transferForSignal(first) && (first = firstWaiter) != null);
  > }
  > 
  > final boolean transferForSignal(Node node) {
  >     //将该节点从状态CONDITION改变为初始状态0,
  >     if (!compareAndSetWaitStatus(node, Node.CONDITION, 0))
  >         return false;
  > 
  >     //将节点加入到syn队列中去，返回的是syn队列中node节点前面的一个节点
  >     Node p = enq(node);
  >     int ws = p.waitStatus;
  >     //如果结点p的状态为cancel 或者修改waitStatus失败，则直接唤醒
  >     if (ws > 0 || !compareAndSetWaitStatus(p, ws, Node.SIGNAL))
  >         LockSupport.unpark(node.thread);
  >     return true;
  > }
  > ```
  >
  > 整体流程： 
  >
  > - 判断当前线程是否已经获取了锁，如果没有获取则直接抛出异常，因为获取锁为通知的前置条件
  > - 如果线程已经获取了锁，则将唤醒条件队列的首节点
  > - 唤醒首节点是先将条件队列中的头节点移出，然后调用 AQS 的 enq(Node node) 方法将其安全地移到 CLH 同步队列中
  > - 最后判断如果该节点的同步状态是否为 Cancel，或者修改状态为 Signal 失败时，则直接调用LockSupport 唤醒该节点的线程

![](../../pics/concurrent/concurrent_28.png)

## 6. LockSupport

![](../../pics/concurrent/concurrent_29.png)

> - `park(Object blocker)` 方法的 blocker 参数，主要是用来标识当前线程在等待的对象，该对象主要用于问题排查和系统监控
> - park 和 unpark(Thread thread) 成对出现，且 unpark 在 park 执行之后执行

- `park()` 源码： 

  ```java
  public static void park() {
      UNSAFE.park(false, 0L);
  }
  ```

- `unpark(Thread thread)` 源码： 

  ```java
  public static void unpark(Thread thread) {
      if (thread != null)
          UNSAFE.unpark(thread);
  }
  ```

# 五、并发容器

## 1. concurrentHashMap

### 1. 重要概念

```java
// 最大容量：2^30=1073741824
private static final int MAXIMUM_CAPACITY = 1 << 30;
// 默认初始值，必须是2的幕数
private static final int DEFAULT_CAPACITY = 16;
//
static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
//
private static final int DEFAULT_CONCURRENCY_LEVEL = 16;
//
private static final float LOAD_FACTOR = 0.75f;
// 链表转红黑树阀值,> 8 链表转换为红黑树
static final int TREEIFY_THRESHOLD = 8;
//树转链表阀值，小于等于6（tranfer时，lc、hc=0两个计数器分别++记录原bin、新binTreeNode数量，<=UNTREEIFY_THRESHOLD 则untreeify(lo)）
static final int UNTREEIFY_THRESHOLD = 6;
//
static final int MIN_TREEIFY_CAPACITY = 64;
//
private static final int MIN_TRANSFER_STRIDE = 16;
//
private static int RESIZE_STAMP_BITS = 16;
// 2^15-1，help resize的最大线程数
private static final int MAX_RESIZERS = (1 << (32 - RESIZE_STAMP_BITS)) - 1;
// 32-16=16，sizeCtl中记录size大小的偏移量
private static final int RESIZE_STAMP_SHIFT = 32 - RESIZE_STAMP_BITS;
// forwarding nodes的hash值
static final int MOVED = -1;
// 树根节点的hash值
static final int TREEBIN   = -2;
// ReservationNode的hash值
static final int RESERVED  = -3;
// 可用处理器数量
static final int NCPU = Runtime.getRuntime().availableProcessors();
```

- **table**：用来存放 Node 节点数据，默认为null，默认大小为 16 的数组，每次扩容时大小总是2的幂次方

- **nextTable**：扩容时新生成的数据，数组为 table 的两倍

- **Node**：节点，保存 key-value 的数据结构

- **ForwardingNode**：特殊的 Node 节点，hash 值为 -1，存储 nextTable 的引用

  > - table 发生扩容时，ForwardingNode 才会发挥作用
  > - 作为一个占位符放在 table 中，表示当前节点为 null 或已经被移动

- **sizeCtl**：控制标识符，用来控制 table 初始化和扩容操作

  - 负数代表正在进行初始化或扩容操作
  - -1 代表正在初始化
  - -N 表示有 N-1 个线程正在进行扩容操作
  - 正数或0代表hash表还没有被初始化，这个数值表示初始化或下一次进行扩容的大小

### 2. 重要内部类

#### 1. Node

- Node：存放 key-value 键值对，所有插入ConcurrentHashMap 的中数据都将会包装在 Node 中

  ```java
  static class Node<K,V> implements Map.Entry<K,V> {
      final int hash;
      final K key;
      volatile V val;             //带有volatile，保证可见性
      volatile Node<K,V> next;    //下一个节点的指针
  
      Node(int hash, K key, V val, Node<K,V> next) {
          this.hash = hash;
          this.key = key;
          this.val = val;
          this.next = next;
      }
  
      public final K getKey()       { return key; }
      public final V getValue()     { return val; }
      public final int hashCode()   { return key.hashCode() ^ val.hashCode(); }
      public final String toString(){ return key + "=" + val; }
      /** 不允许修改value的值 */
      public final V setValue(V value) {
          throw new UnsupportedOperationException();
      }
  
      public final boolean equals(Object o) {
          Object k, v, u; 
          Map.Entry<?,?> e;
          return ((o instanceof Map.Entry) &&
                  (k = (e = (Map.Entry<?,?>)o).getKey()) != null &&
                  (v = e.getValue()) != null &&
                  (k == key || k.equals(key)) &&
                  (v == (u = val) || v.equals(u)));
      }
  
      /**  赋值get()方法 */
      Node<K,V> find(int h, Object k) {
          Node<K,V> e = this;
          if (k != null) {
              do {
                  K ek;
                  if (e.hash == h &&
                          ((ek = e.key) == k || (ek != null && k.equals(ek))))
                      return e;
              } while ((e = e.next) != null);
          }
          return null;
      }
  }
  ```

#### 2. TreeNode

- 链表转红黑树： **将链表的节点包装成 TreeNode 放在 TreeBin 对象中，然后由 TreeBin 完成红黑树的转换**

  ```java
  static final class TreeNode<K,V> extends Node<K,V> {
      TreeNode<K,V> parent;  // red-black tree links
      TreeNode<K,V> left;
      TreeNode<K,V> right;
      TreeNode<K,V> prev;    // needed to unlink next upon deletion
      boolean red;
  
      TreeNode(int hash, K key, V val, Node<K,V> next, TreeNode<K,V> parent) {
          super(hash, key, val, next);
          this.parent = parent;
      }
  
      Node<K,V> find(int h, Object k) {
          return findTreeNode(h, k, null);
      }
  
      //查找hash为h，key为k的节点
      final TreeNode<K,V> findTreeNode(int h, Object k, Class<?> kc) {
          if (k != null) {
              TreeNode<K,V> p = this;
              do  {
                  int ph, dir; K pk; TreeNode<K,V> q;
                  TreeNode<K,V> pl = p.left, pr = p.right;
                  if ((ph = p.hash) > h)
                      p = pl;
                  else if (ph < h)
                      p = pr;
                  else if ((pk = p.key) == k || (pk != null && k.equals(pk)))
                      return p;
                  else if (pl == null)
                      p = pr;
                  else if (pr == null)
                      p = pl;
                  else if ((kc != null ||
                          (kc = comparableClassFor(k)) != null) &&
                          (dir = compareComparables(kc, k, pk)) != 0)
                      p = (dir < 0) ? pl : pr;
                  else if ((q = pr.findTreeNode(h, k, kc)) != null)
                      return q;
                  else
                      p = pl;
              } while (p != null);
          }
          return null;
      }
  }
  ```

#### 3. TreeBin

- 作用： **用于在链表转换为红黑树时包装 TreeNode 节点**

  > 即 ConcurrentHashMap 红黑树存放是 TreeBin

#### 4. ForwardingNode

- 仅仅存活在 ConcurrentHashMap 扩容操作时，只是一个标志节点，且指向 nextTable，提供 find 方法而已

- 该类也是集成 Node 节点，其hash为 -1，key、value、next 均为null

  ```java
  static final class ForwardingNode<K,V> extends Node<K,V> {
      final Node<K,V>[] nextTable;
      ForwardingNode(Node<K,V>[] tab) {
          super(MOVED, null, null, null);
          this.nextTable = tab;
      }
  
      Node<K,V> find(int h, Object k) {
          // loop to avoid arbitrarily deep recursion on forwarding nodes
          outer: for (Node<K,V>[] tab = nextTable;;) {
              Node<K,V> e; int n;
              if (k == null || tab == null || (n = tab.length) == 0 ||
                      (e = tabAt(tab, (n - 1) & h)) == null)
                  return null;
              for (;;) {
                  int eh; K ek;
                  if ((eh = e.hash) == h &&
                          ((ek = e.key) == k || (ek != null && k.equals(ek))))
                      return e;
                  if (eh < 0) {
                      if (e instanceof ForwardingNode) {
                          tab = ((ForwardingNode<K,V>)e).nextTable;
                          continue outer;
                      }
                      else
                          return e.find(h, k);
                  }
                  if ((e = e.next) == null)
                      return null;
              }
          }
      }
  }
  ```

### 3. 构造函数

```java
public ConcurrentHashMap() {
}

public ConcurrentHashMap(int initialCapacity) {
    if (initialCapacity < 0)
        throw new IllegalArgumentException();
    int cap = ((initialCapacity >= (MAXIMUM_CAPACITY >>> 1)) ?
               MAXIMUM_CAPACITY :
               tableSizeFor(initialCapacity + (initialCapacity >>> 1) + 1));
    this.sizeCtl = cap;
}

public ConcurrentHashMap(Map<? extends K, ? extends V> m) {
    this.sizeCtl = DEFAULT_CAPACITY;
    putAll(m);
}

public ConcurrentHashMap(int initialCapacity, float loadFactor) {
    this(initialCapacity, loadFactor, 1);
}

public ConcurrentHashMap(int initialCapacity,
                         float loadFactor, int concurrencyLevel) {
    if (!(loadFactor > 0.0f) || initialCapacity < 0 || concurrencyLevel <= 0)
        throw new IllegalArgumentException();
    if (initialCapacity < concurrencyLevel)   // Use at least as many bins
        initialCapacity = concurrencyLevel;   // as estimated threads
    long size = (long)(1.0 + (long)initialCapacity / loadFactor);
    int cap = (size >= (long)MAXIMUM_CAPACITY) ?
        MAXIMUM_CAPACITY : tableSizeFor((int)size);
    this.sizeCtl = cap;
}
```

### 4. 初始化 initTable()

```java
private final Node<K,V>[] initTable() {
    Node<K,V>[] tab; int sc;
    while ((tab = table) == null || tab.length == 0) {
        //sizeCtl < 0 表示有其他线程在初始化，该线程必须挂起
        if ((sc = sizeCtl) < 0)
            Thread.yield();
        // 如果该线程获取了初始化的权利，则用CAS将sizeCtl设置为-1，表示本线程正在初始化
        else if (U.compareAndSwapInt(this, SIZECTL, sc, -1)) {
                // 进行初始化
            try {
                if ((tab = table) == null || tab.length == 0) {
                    int n = (sc > 0) ? sc : DEFAULT_CAPACITY;
                    @SuppressWarnings("unchecked")
                    Node<K,V>[] nt = (Node<K,V>[])new Node<?,?>[n];
                    table = tab = nt;
                    // 下次扩容的大小
                    sc = n - (n >>> 2); ///相当于0.75*n 设置一个扩容的阈值  
                }
            } finally {
                sizeCtl = sc;
            }
            break;
        }
    }
    return tab;
}
```

初始化方法 initTable() 的关键就在于 sizeCtl，该值默认为 0： 

- 如果在构造函数时，有参数传入该值则为 2 的幂次方

  > 表示将要进行初始化或扩容的大小

- 该值如果 < 0，表示有其他线程正在初始化，则必须暂停该线程

- 如果线程获得了初始化的权限则先将 sizeCtl 设置为-1，防止有其他线程进入，最后将sizeCtl设置0.75 * n，表示扩容的阈值

### 5. 操作函数

#### 1. put 操作

- **核心思想**： 根据 hash 值计算节点插入 table 的位置，如果该位置为空，则直接插入，否则插入到链表或树中

执行流程：

- 判空： ConcurrentHashMap的key、value都不允许为null

- 计算hash： 利用方法计算 hash 值

  ```java
  static final int spread(int h) {
      return (h ^ (h >>> 16)) & HASH_BITS;
  }
  ```

- 遍历 table，进行节点插入操作： 

  - 如果 table 为空，则表示 ConcurrentHashMap 还没有初始化，则进行初始化操作：initTable()

  - 根据 hash 值获取节点的位置 i，若该位置为空，则直接插入，不需要加锁

    > 计算 f 位置：`i=(n – 1) & hash`

  - 如果检测到 `fh = f.hash == -1`，则 f 是 ForwardingNode 节点，表示有其他线程正在进行扩容操作，则帮助线程一起进行扩容操作

  - 如果 `f.hash >= 0` 表示是链表结构，则遍历链表，如果存在当前 key 节点则替换 value，否则插入到链表尾部。如果 f 是 TreeBin 类型节点，则按照红黑树的方法更新或者增加节点

  - 若链表长度` > TREEIFY_THRESHOLD`(默认是8)，则将链表转换为红黑树结构

- 调用 addCount 方法，ConcurrentHashMap 的 size + 1

```java
public V put(K key, V value) {
    return putVal(key, value, false);
}

final V putVal(K key, V value, boolean onlyIfAbsent) {
    //key、value均不能为null
    if (key == null || value == null) throw new NullPointerException();
    //计算hash值
    int hash = spread(key.hashCode());
    int binCount = 0;
    for (Node<K,V>[] tab = table;;) {
        Node<K,V> f; int n, i, fh;
        // table为null，进行初始化工作
        if (tab == null || (n = tab.length) == 0)
            tab = initTable();
        //如果i位置没有节点，则直接插入，不需要加锁
        else if ((f = tabAt(tab, i = (n - 1) & hash)) == null) {
            if (casTabAt(tab, i, null, new Node<K,V>(hash, key, value, null)))
                break;                   // no lock when adding to empty bin
        }
        // 有线程正在进行扩容操作，则先帮助扩容
        else if ((fh = f.hash) == MOVED)
            tab = helpTransfer(tab, f);
        else {
            V oldVal = null;
            //对该节点进行加锁处理（hash值相同的链表的头节点），对性能有点儿影响
            synchronized (f) {
                if (tabAt(tab, i) == f) {
                    //fh > 0 表示为链表，将该节点插入到链表尾部
                    if (fh >= 0) {
                        binCount = 1;
                        for (Node<K,V> e = f;; ++binCount) {
                            K ek;
                            //hash 和 key 都一样，替换value
                            if (e.hash == hash && ((ek = e.key) == key ||
                                            (ek != null && key.equals(ek)))) {
                                oldVal = e.val;
                                //putIfAbsent()
                                if (!onlyIfAbsent)
                                    e.val = value;
                                break;
                            }
                            Node<K,V> pred = e;
                            //链表尾部  直接插入
                            if ((e = e.next) == null) {
                                pred.next = new Node<K,V>(hash, key, value, null);
                                break;
                            }
                        }
                    }
                    //树节点，按照树的插入操作进行插入
                    else if (f instanceof TreeBin) {
                        Node<K,V> p;
                        binCount = 2;
                        if ((p = ((TreeBin<K,V>)f).putTreeVal(hash, key, value)) 
                            		!= null) {
                            oldVal = p.val;
                            if (!onlyIfAbsent)
                                p.val = value;
                        }
                    }
                }
            }
            if (binCount != 0) {
                // 如果链表长度已经达到临界值8 就需要把链表转换为树结构
                if (binCount >= TREEIFY_THRESHOLD)
                    treeifyBin(tab, i);
                if (oldVal != null)
                    return oldVal;
                break;
            }
        }
    }
    //size + 1  
    addCount(1L, binCount);
    return null;
}
```

#### 2. get 操作

执行流程： 

- 计算 hash 值
- 判断 table 是否为空，如果为空，直接返回 null
- 根据 hash 值获取 table 中的 Node 节点（tabAt(tab, (n – 1) & h)）
- 然后根据链表或者树形方式找到相对应的节点，返回其 value 值

```java
public V get(Object key) {
    Node<K,V>[] tab; Node<K,V> e, p; int n, eh; K ek;
    // 计算hash
    int h = spread(key.hashCode());
    if ((tab = table) != null && (n = tab.length) > 0 &&
            (e = tabAt(tab, (n - 1) & h)) != null) {
        // 搜索到的节点key与传入的key相同且不为null,直接返回这个节点
        if ((eh = e.hash) == h) {
            if ((ek = e.key) == key || (ek != null && key.equals(ek)))
                return e.val;
        }
        // 树
        else if (eh < 0)
            return (p = e.find(h, key)) != null ? p.val : null;
        // 链表，遍历
        while ((e = e.next) != null) {
            if (e.hash == h &&
                    ((ek = e.key) == key || (ek != null && key.equals(ek))))
                return e.val;
        }
    }
    return null;
}
```

#### 3. size 操作

- size() 返回的是一个不精确的值，因为在进行统计的时候有其他线程正在进行插入和删除操作

```java
public int size() {
    long n = sumCount();
    return ((n < 0L) ? 0 :
            (n > (long)Integer.MAX_VALUE) ? Integer.MAX_VALUE :
            (int)n);
}
//sumCount()就是迭代counterCells来统计sum的过程
final long sumCount() {
    CounterCell[] as = counterCells; 
    CounterCell a;
    long sum = baseCount;
    if (as != null) {
        for (int i = 0; i < as.length; ++i) {
            //遍历，所有counter求和
            if ((a = as[i]) != null)
                sum += a.value;     
        }
    }
    return sum;
}

@sun.misc.Contended static final class CounterCell {
    volatile long value;
    CounterCell(long x) { value = x; }
}

//ConcurrentHashMap中元素个数,但返回的不一定是当前Map的真实元素个数。基于CAS无锁更新
private transient volatile long baseCount;

private transient volatile CounterCell[] counterCells;

/**put 操作最后调用的 addCount 方法
 * 1. 更新baseCount的值
 * 2. 检测是否进行扩容
 */
private final void addCount(long x, int check) {
    CounterCell[] as; long b, s;
    // s = b + x，完成baseCount++操作；
    if ((as = counterCells) != null ||
        !U.compareAndSwapLong(this, BASECOUNT, b = baseCount, s = b + x)) {
        CounterCell a; long v; int m;
        boolean uncontended = true;
        if (as == null || (m = as.length - 1) < 0 ||
            (a = as[ThreadLocalRandom.getProbe() & m]) == null ||
           !(uncontended = U.compareAndSwapLong(a, CELLVALUE, v = a.value, v + x))) {
            //  多线程CAS发生失败时执行
            fullAddCount(x, uncontended);
            return;
        }
        if (check <= 1)
            return;
        s = sumCount();
    }
    // 检查是否进行扩容...
}
```

#### 4. 扩容操作

**扩容步骤**： 

- 为每个内核分任务，并保证其不小于16
- 检查 nextTable 是否为null，如果是，则初始化 nextTable，使其容量为 table 的两倍
- 死循环遍历节点，直到 finished：节点从 table 复制到 nextTable 中，支持并发：
  - 如果节点 f 为 null，则插入 ForwardingNode（采用Unsafe.compareAndSwapObjectf方法实现），这个是触发并发扩容的关键
  - 如果 f 为链表的头节点（fh >= 0）,则先构造一个反序链表，然后把他们分别放在 nextTable 的 i 和 i + n位置，并将 ForwardingNode 插入原节点位置，代表已经处理过了
  - 如果 f 为 TreeBin 节点，同样也是构造一个反序 ，同时判断是否需要进行 unTreeify() 操作，并把处理的结果分别插入到 nextTable 的 i  和 i+nw 位置，并插入 ForwardingNode 节点
- 所有节点复制完成后，则将 table 指向 nextTable，同时更新 sizeCtl = nextTable 的 0.75 倍，完成扩容过程

**扩容关键**： 

- 当一个线程遍历到的节点，如果是 ForwardingNode，则继续往后遍历，如果不是，则将该节点加锁，防止其他线程进入，完成后设置 ForwardingNode节点，以便要其他线程可以看到该节点已经处理过了，如此交叉进行，高效而又安全

![](../../pics/concurrent/concurrent_30.png)

### 6. 红黑树转换

#### 1. 红黑树

红黑树特点：

- 每个节点非红即黑
- 根节点为黑色
- 每个叶子节点为黑色。叶子节点为NIL节点，即空节点
- 如果一个节点为红色，那么它的子节点一定是黑色
- 从一个节点到该节点的子孙节点的所有路径包含相同个数的黑色节点

![](../../pics/concurrent/concurrent_34.png)

#### 2. treeifyBin

> ConcurrentHashMap 的链表转换为红黑树过程就是一个红黑树增加节点的过程

```java
if (binCount >= TREEIFY_THRESHOLD)
    treeifyBin(tab, i);

private final void treeifyBin(Node<K,V>[] tab, int index) {
    Node<K,V> b; int n, sc;
    if (tab != null) {
        if ((n = tab.length) < MIN_TREEIFY_CAPACITY)  //判断是否小于 
            tryPresize(n << 1);
        else if ((b = tabAt(tab, index)) != null && b.hash >= 0) {
            synchronized (b) {
                if (tabAt(tab, index) == b) {
                    TreeNode<K,V> hd = null, tl = null;
                    for (Node<K,V> e = b; e != null; e = e.next) {
                        TreeNode<K,V> p =
                            new TreeNode<K,V>(e.hash, e.key, e.val,
                                              null, null);
                        if ((p.prev = tl) == null)
                            hd = p;
                        else
                            tl.next = p;
                        tl = p;
                    }
                    setTabAt(tab, index, new TreeBin<K,V>(hd));
                }
            }
        }
    }
}
```

## 2. CopyOnWriteArrayList







## 3. ConcurrentSkipListMap







## 4. ConcurrentLinkedQueue

> ConcurrentLinkedQueue 是一个基于链接节点的无边界的线程安全队列： 
>
> - 采用 FIFO 原则对元素进行排序
> - 采用“wait-free”算法（即CAS算法）来实现

### 1. 简介

**CoucurrentLinkedQueue 规定的几个不变性**：

- 在入队的最后一个元素的 next 为 null

- 队列中所有未删除的节点的 item 都不能为 null 且都能从 head 节点遍历到

- 对于要删除的节点，不是直接将其设置为 null，而是先将其 item 域设置为 null

- 允许 head 和 tail 更新滞后，即 head、tail 不总是指向第一个元素和最后一个元素

**head 的不变性和可变性**：

- 不变性
  1. 所有未删除的节点都可以通过 head 节点遍历到
  2. head 不能为 null
  3. head 节点的 next 不能指向自身
- 可变性
  1. head 的 item 可能为 null，也可能不为 null
  2. 允许 tail 滞后 head，即调用 succc() 方法，从 head 不可达 tail

**tail 的不变性和可变性**： 

- 不变性： tail 不能为 null
- 可变性：
  1. tail 的 item 可能为 null，也可能不为 null
  2. tail 节点的 next 域可以指向自身
  3. 允许 tail 滞后 head，即调用 succc() 方法，从 head 不可达 tail

### 2. 源码分析

#### 1. 节点

CoucurrentLinkedQueue 的结构由head节点和tail节点组成： 

- 每个节点由节点元素 item 和指向下一个节点的 next 引用组成
- 节点与节点之间通过该 next 关联，从而组成一张链表的队列
- 节点 Node 为 ConcurrentLinkedQueue 的内部类

```java
private static class Node<E> {
    /** 节点元素域 */
    volatile E item;
    volatile Node<E> next;
    //初始化,获得item 和 next 的偏移量,为后期的CAS做准备
    Node(E item) {
        UNSAFE.putObject(this, itemOffset, item);
    }

    boolean casItem(E cmp, E val) {
        return UNSAFE.compareAndSwapObject(this, itemOffset, cmp, val);
    }

    void lazySetNext(Node<E> val) {
        UNSAFE.putOrderedObject(this, nextOffset, val);
    }

    boolean casNext(Node<E> cmp, Node<E> val) {
        return UNSAFE.compareAndSwapObject(this, nextOffset, cmp, val);
    }

    // Unsafe mechanics

    private static final sun.misc.Unsafe UNSAFE;
    /** 偏移量 */
    private static final long itemOffset;
    /** 下一个元素的偏移量 */
    private static final long nextOffset;

    static {
        try {
            UNSAFE = sun.misc.Unsafe.getUnsafe();
            Class<?> k = Node.class;
            itemOffset = UNSAFE.objectFieldOffset(k.getDeclaredField("item"));
            nextOffset = UNSAFE.objectFieldOffset(k.getDeclaredField("next"));
        } catch (Exception e) {
            throw new Error(e);
        }
    }
}
```

#### 2. 入列

- `offer(E e)`：将指定元素插入队列尾部

  ```java
  public boolean offer(E e) {
      //检查节点是否为null
      checkNotNull(e);
      // 创建新节点
      final Node<E> newNode = new Node<E>(e);
      //死循环直到成功为止
      for (Node<E> t = tail, p = t;;) {
          Node<E> q = p.next;
          // q == null 表示 p已经是最后一个节点了，尝试加入到队列尾
          // 如果插入失败，则表示其他线程已经修改了 p 的指向
          if (q == null) {                                // --- 1
              // casNext：t 节点的next指向当前节点
              if (p.casNext(null, newNode)) {             // --- 2
                  // node 加入节点后会导致tail距离最后一个节点相差大于一个，需要更新tail
                  if (p != t)                             // --- 3
                      // casTail：设置 tail 尾节点
                      casTail(t, newNode);                // --- 4
                  return true;
              }
          }
          // p == q 等于自身
          else if (p == q)                                // --- 5
              // p == q 代表着该节点已经被删除了
              // 由于多线程的原因，我们offer()的时候也会poll()
              //如果offer()的时候正好该节点已经poll()了
              // 则在poll()方法中的updateHead()方法会将head指向当前的q
              //而把p.next指向自己，即：p.next == p
              // 这样就会导致tail节点滞后head（tail位于head的前面），则需要重新设置p
              p = (t != (t = tail)) ? t : head;           // --- 6
          // tail并没有指向尾节点
          else
              // tail已经不是最后一个节点，将p指向最后一个节点
              p = (p != t && t != (t = tail)) ? t : q;    // --- 7
      }
  }
  ```

  详细过程： 

  - **初始化**： head、tail 存储的元素都为 null，且 head 等于 tail

    ![](../../pics/concurrent/concurrent_35.png)

  - **添加元素 A**：

    - 第一次插入元素 A，head = tail = dummyNode，所有 q = p.next = null
    - 直接走步骤2：p.casNext(null, newNode)，由于 p == t成立，所以不会执行步骤3：casTail(t, newNode)，直接 return

    ![](../../pics/concurrent/concurrent_36.png)

  - **添加元素 B**： 

    - q = p.next = A ,p = tail = dummyNode，所以直接跳到步骤7：p = (p != t && t != (t = tail)) ? t : q
    - 此时p = q，然后进行第二次循环 q = p.next = null，步骤2：p == null成立，将该节点插入，因为p = q，t = tail，所以步骤3：p != t 成立，执行步骤4：casTail(t, newNode)，然后return

    ![](../../pics/concurrent/concurrent_37.png)

  - **添加元素 C**： 此时t = tail ,p = t，q = p.next = null，和插入元素A无异

    ![](../../pics/concurrent/concurrent_38.png)

#### 3. 出列

```java
public E poll() {
    // 如果出现p被删除的情况需要从head重新开始
    restartFromHead:        // 这是什么语法？真心没有见过
    for (;;) {
        for (Node<E> h = head, p = h, q;;) {
            // 节点 item
            E item = p.item;
            // item 不为null，则将item 设置为null
            if (item != null && p.casItem(item, null)) {                    // --- 1
                // p != head 则更新head
                if (p != h)                                                 // --- 2
                    // p.next != null，则将head更新为p.next ,否则更新为p
                    updateHead(h, ((q = p.next) != null) ? q : p);          // --- 3
                return item;
            }
            // p.next == null 队列为空
            else if ((q = p.next) == null) {                                // --- 4
                updateHead(h, p);
                return null;
            }
            // 当一个线程在 poll 时，另一个线程已经把当前的 p 从队列中删除
            //将p.next = p，p已经被移除不能继续，需要重新开始
            else if (p == q)                                                // --- 5
                continue restartFromHead;
            else
                p = q;                                                      // --- 6
        }
    }
}
//用于 CAS 更新 head 节点
final void updateHead(Node<E> h, Node<E> p) {
    if (h != p && casHead(h, p))
        h.lazySetNext(h);
}
```

详解过程： 

- **原始链表**： 

  ![](../../pics/concurrent/concurrent_38.png)

- **poll A**：head = dumy，p = head， item = p.item = null，步骤1不成立，步骤4：(q = p.next) == null不成立，p.next = A，跳到步骤6，下一个循环，此时p = A，所以步骤1 item != null，进行p.casItem(item, null)成功，此时p == A != h，所以执行步骤3：updateHead(h, ((q = p.next) != null) ? q : p)，q = p.next = B != null，则将head CAS更新成B

  ![](../../pics/concurrent/concurrent_39.png)

- **poll B**：head = B ， p = head = B，item = p.item = B，步骤成立，步骤2：p != h 不成立，直接return

  ![](../../pics/concurrent/concurrent_40.png)

- **poll C**： head = dumy ，p = head = dumy，tiem = p.item = null，步骤1不成立，跳到步骤4：(q = p.next) == null，不成立，然后跳到步骤6，此时，p = q = C，item = C(item)，步骤1成立，所以讲C（item）设置为null，步骤2：p != h成立，执行步骤3：updateHead(h, ((q = p.next) != null) ? q : p)

  ![](../../pics/concurrent/concurrent_41.png)

## 5. ThreadLocal

ThreadLocal定义了四个方法：

- `get()`：返回此线程局部变量的当前线程副本中的值
- `initialValue()`：返回此线程局部变量的当前线程的“初始值”
- `remove()`：移除此线程局部变量当前线程的值
- `set(T value)`：将此线程局部变量的当前线程副本中的值设置为指定值

> ThreadLocalMap 提供了一种用键值对方式存储每一个线程的变量副本的方法： 
>
> - key 为当前 ThreadLoca l对象
> - value 则是对应线程的变量副本

注意： 

- ThreadLocal 实例本身是不存储值，只提供一个在当前线程中找到副本值得 key
- ThreadLocal 包含在 Thread 中，而不是 Thread 包含在 ThreadLocal 中

![](../../pics/concurrent/concurrent_31.png)

### 1. ThreadLocalMap

```java
//利用 Entry 来实现 key-value 的存储
//通过 WeakReference 避免内存泄漏
static class Entry extends WeakReference<ThreadLocal<?>> {
    Object value;

    Entry(ThreadLocal<?> k, Object v) {
        super(k);
        value = v;
    }
}
```

- `set(ThreadLocal key, Object value)`： 

  > - 采用**开放定址法**解决 hash 冲突
  > - replaceStaleEntry()和cleanSomeSlots() 可以清除掉 key == null 的实例，防止内存泄漏

  ```java
  private void set(ThreadLocal<?> key, Object value) {
      ThreadLocal.ThreadLocalMap.Entry[] tab = table;
      int len = tab.length;
      // 根据 ThreadLocal 的散列值，查找对应元素在数组中的位置
      int i = key.threadLocalHashCode & (len-1);
      // 采用“线性探测法”，寻找合适位置
      for (ThreadLocal.ThreadLocalMap.Entry e = tab[i];e != null;
          	e = tab[i = nextIndex(i, len)]) {
          ThreadLocal<?> k = e.get();
          // key 存在，直接覆盖
          if (k == key) {
              e.value = value;
              return;
          }
          // key == null，但 e != null，说明之前的ThreadLocal对象已经被回收
          if (k == null) {
              replaceStaleEntry(key, value, i); // 用新元素替换陈旧的元素
              return;
          }
      }
  
      // ThreadLocal对应的key实例不存在也没有陈旧元素，new 一个
      tab[i] = new ThreadLocal.ThreadLocalMap.Entry(key, value);
  
      int sz = ++size;
      // cleanSomeSlots 清楚陈旧的Entry（key == null）
      // 如果没有清理陈旧的 Entry 并且数组中的元素大于了阈值，则进行 rehash
      if (!cleanSomeSlots(i, sz) && sz >= threshold)
          rehash();
  }
  ```

- `getEntry(ThreadLocal<?> key)`： 

  ```java
  //因为采用开放定址法，所以要检测
  private Entry getEntry(ThreadLocal<?> key) {
      //首先取一个探测数(key的散列值)
      int i = key.threadLocalHashCode & (table.length - 1); 
      Entry e = table[i];
      if (e != null && e.get() == key) //如果所对应的key就是我们所要找的元素，则返回
          return e;
      else
          return getEntryAfterMiss(key, i, e); //否则调用getEntryAfterMiss()
  }
  
  private Entry getEntryAfterMiss(ThreadLocal<?> key, int i, Entry e) {
      Entry[] tab = table;
      int len = tab.length;
  
      while (e != null) {
          ThreadLocal<?> k = e.get();
          if (k == key)
              return e;
          if (k == null)
              expungeStaleEntry(i); //处理 k，防止内存泄漏
          else
              i = nextIndex(i, len);
          e = tab[i];
      }
      return null;
  }
  ```

### 2. get() 

- 作用： 返回当前线程所对应的线程变量

```java
public T get() {
    // 获取当前线程
    Thread t = Thread.currentThread();
    // 获取当前线程的成员变量 threadLocal
    ThreadLocalMap map = getMap(t);
    if (map != null) {
        // 从当前线程的ThreadLocalMap获取相对应的Entry
        ThreadLocalMap.Entry e = map.getEntry(this);
        if (e != null) {
            @SuppressWarnings("unchecked")
            T result = (T)e.value; // 获取目标值  
            return result;
        }
    }
    return setInitialValue();
}
//获取当前线程所对应的 ThreadLocalMap
ThreadLocalMap getMap(Thread t) {
    return t.threadLocals;
}
```

### 3. set(T value)

- 作用： 设置当前线程的线程局部变量的值

```java
public void set(T value) {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t); //获取当前线程所对应的 ThreadLocalMap
    if (map != null)  //不为空
        map.set(this, value); //调用ThreadLocalMap的set()方法，key就是当前ThreadLocal
    else //不存在
        createMap(t, value); //新建一个
}

void createMap(Thread t, T firstValue) {
    t.threadLocals = new ThreadLocalMap(this, firstValue);
}
```

### 4. initialValue()

- 作用： 返回该线程局部变量的初始值

```java
protected T initialValue() {
    return null;
}
```

注意： 

- 该方法定义为 protected 且返回为 null，很明显要子类实现，所以在使用 ThreadLocal 时应该覆盖该方法
- 该方法不能显示调用，只有在第一次调用 get() 或 set() 方法时才会被执行，并且仅执行 1 次

### 5. remove()

- 作用： 将当前线程局部变量的值删除，减少内存的占用

  > 不需要显示调用该方法，因为一个线程结束后，它所对应的局部变量就会被垃圾回收

```java
public void remove() {
    ThreadLocalMap m = getMap(Thread.currentThread());
    if (m != null)
        m.remove(this);
}
```

### 6. 内存泄漏

当 ThreadLocal的key == null 时，GC 就会回收这部分空间，但是 value 却不一定能够被回收，因为他还与Current Thread 存在一个强引用关系： 

> 由于存在这个强引用关系，会导致 value 无法回收

![](../../pics/concurrent/concurrent_32.png)

解决： 

- 在 ThreadLocalMap 中的 setEntry()、getEntry()，如果遇到 key == null 的情况，会对 value 设置为null
- 可以显示调用 ThreadLocal 的 remove() 方法进行处理

总结： 

- ThreadLocal 不是用于解决共享变量的问题的，也不是为了协调线程同步而存在，而是**为了方便每个线程处理自己的状态而引入的一个机制**

- 每个 Thread 内部都有一个 ThreadLocal.ThreadLocalMap 类型的成员变量，该成员变量用来存储实际的ThreadLocal 变量副本

- ThreadLocal 并不是为线程保存对象的副本，它仅仅只起到一个索引的作用

  > 主要目的： 为每一个线程隔离一个类的实例，这个实例的作用范围仅限于线程内部

## 6. Queue

### 1. ArrayBlockingQueue







### 2. PriorityBlockingQueue







### 3. DelayQueue







### 4. SynchronousQueue







### 5. LinkedTransferQueue







### 6. LinkedBlockingDeque







# 六、线程池(Executor)

## 1. ThreadPoolExecutor

### 1. 内部状态

```java
private final AtomicInteger ctl = new AtomicInteger(ctlOf(RUNNING, 0));
private static final int COUNT_BITS = Integer.SIZE - 3;
private static final int CAPACITY   = (1 << COUNT_BITS) - 1;

// runState is stored in the high-order bits
private static final int RUNNING    = -1 << COUNT_BITS;
private static final int SHUTDOWN   =  0 << COUNT_BITS;
private static final int STOP       =  1 << COUNT_BITS;
private static final int TIDYING    =  2 << COUNT_BITS;
private static final int TERMINATED =  3 << COUNT_BITS;

// Packing and unpacking ctl
private static int runStateOf(int c)     { return c & ~CAPACITY; }
private static int workerCountOf(int c)  { return c & CAPACITY; }
private static int ctlOf(int rs, int wc) { return rs | wc; }
```

> `ctl` 定义为 AtomicInteger ，记录了“**线程池中的任务数量**”和“**线程池的状态**”两个信息
>
> > 共32位，其中高3位表示”线程池状态”，低29位表示”线程池中的任务数量”
> >
> > ```
> > RUNNING        -- 对应的高3位值是111
> > SHUTDOWN       -- 对应的高3位值是000
> > STOP           -- 对应的高3位值是001
> > TIDYING        -- 对应的高3位值是010
> > TERMINATED     -- 对应的高3位值是011
> > ```

线程池的五种状态：

- **RUNNING**：处于 RUNNING 状态的线程池能够接受新任务，以及对新添加的任务进行处理

- **SHUTDOWN**：处于 SHUTDOWN 状态的线程池不可以接受新任务，但是可以对已添加的任务进行处理

- **STOP**：处于 STOP 状态的线程池不接收新任务，不处理已添加的任务，并且会中断正在处理的任务

- **TIDYING**：当所有的任务已终止，ctl 记录的”任务数量”为0，线程池会变为 TIDYING 状态

  > 当线程池变为 TIDYING 状态时，会执行钩子函数 `terminated()`
  >
  > > terminated() 在 ThreadPoolExecutor 类中是空的，需自己重载

- **TERMINATED**：线程池彻底终止的状态

![](../../pics/concurrent/concurrent_33.png)

### 2. 创建线程池

```java
public ThreadPoolExecutor(int corePoolSize, int maximumPoolSize, long keepAliveTime,
                          TimeUnit unit, BlockingQueue<Runnable> workQueue,
                          ThreadFactory threadFactory, 
                          RejectedExecutionHandler handler) {
    if (corePoolSize < 0 || maximumPoolSize <= 0 ||
        maximumPoolSize < corePoolSize || keepAliveTime < 0)
        	throw new IllegalArgumentException();
    if (workQueue == null || threadFactory == null || handler == null)
        throw new NullPointerException();
    this.corePoolSize = corePoolSize;
    this.maximumPoolSize = maximumPoolSize;
    this.workQueue = workQueue;
    this.keepAliveTime = unit.toNanos(keepAliveTime);
    this.threadFactory = threadFactory;
    this.handler = handler;
}
```

**各参数含义**： 

- `corePoolSize`： 线程池中核心线程的数量

  > - 当提交一个任务时，线程池会新建一个线程来执行任务，直到当前线程数等于 corePoolSize
  >
  > - 如果调用了线程池的 prestartAllCoreThreads() 方法，线程池会提前创建并启动所有基本线程

- `maximumPoolSize`： 线程池中允许的最大线程数

  > 线程池的阻塞队列满了之后，如果还有任务提交，如果当前的线程数小于 maximumPoolSize，则会新建线程来执行任务

- `keepAliveTime`： 线程空闲的时间，即**线程执行完任务后继续存活时间**

  > 默认情况下，该参数只有在线程数大于 corePoolSize 时才会生效

- `unit`： keepAliveTime 的单位，TimeUnit

- `workQueue`： 用来保存等待执行的任务的阻塞队列，等待的任务必须实现 Runnable 接口

  >  选择如下几种：
  >
  > - ArrayBlockingQueue：基于数组结构的有界阻塞队列，FIFO
  > - LinkedBlockingQueue：基于链表结构的有界阻塞队列，FIFO
  > - SynchronousQueue：不存储元素的阻塞队列，每个插入操作都必须等待一个移出操作，反之亦然
  > - PriorityBlockingQueue：具有优先级的阻塞队列

- `threadFactory`： 用于设置创建线程的工厂，该对象可以通过 Executors.defaultThreadFactory()

  > ```java
  > public static ThreadFactory defaultThreadFactory() {
  >     return new DefaultThreadFactory();
  > }
  > 
  > static class DefaultThreadFactory implements ThreadFactory {
  >     private static final AtomicInteger poolNumber = new AtomicInteger(1);
  >     private final ThreadGroup group;
  >     private final AtomicInteger threadNumber = new AtomicInteger(1);
  >     private final String namePrefix;
  > 
  >     DefaultThreadFactory() {
  >         SecurityManager s = System.getSecurityManager();
  >         group = (s != null) ? s.getThreadGroup() :
  >                               Thread.currentThread().getThreadGroup();
  >         namePrefix = "pool-" + poolNumber.getAndIncrement() + "-thread-";
  >     }
  > 	//newThread() 方法创建的线程都是“非守护线程”
  >     //线程优先级是 Thread.NORM_PRIORITY
  >     public Thread newThread(Runnable r) {
  >         Thread t = new Thread(group, r,
  >                               namePrefix + threadNumber.getAndIncrement(), 0);
  >         if (t.isDaemon())
  >             t.setDaemon(false);
  >         if (t.getPriority() != Thread.NORM_PRIORITY)
  >             t.setPriority(Thread.NORM_PRIORITY);
  >         return t;
  >     }
  > }
  > ```

- `handler`： 线程池的拒绝策略，指将任务添加到线程池中时，线程池拒绝该任务所采取的相应策略

  > 当向线程池中提交任务时，如果线程池中的线程已经饱和，且阻塞队列已经满了，则线程池会选择一种拒绝策略来处理该任务
  >
  >  
  >
  > 线程池提供了四种拒绝策略：
  >
  > 1. `AbortPolicy`：直接抛出异常，默认策略
  > 2. `CallerRunsPolicy`：用调用者所在的线程来执行任务
  > 3. `DiscardOldestPolicy`：丢弃阻塞队列中靠最前的任务，并执行当前任务
  > 4. `DiscardPolicy`：直接丢弃任务

![](../../pics/concurrent/concurrent_34.jpg)

![](../../pics/concurrent/concurrent_35.jpg)

### 3. 线程池

- `FixedThreadPool`： 可重用固定线程数的线程池

  > ```java
  > public static ExecutorService newFixedThreadPool(int nThreads) {
  >     return new ThreadPoolExecutor(nThreads, nThreads,0L, TimeUnit.MILLISECONDS,
  >                                   new LinkedBlockingQueue<Runnable>());
  > }
  > ```
  >
  > FixedThreadPool 使用的是“无界队列”LinkedBlockingQueue，则该线程池不会拒绝提交的任务

- `SingleThreadExecutor`： 使用单个 worker 线程的 Executor

  > ```java
  > public static ExecutorService newSingleThreadExecutor() {
  >     return new FinalizableDelegatedExecutorService
  >         (new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
  >                                 new LinkedBlockingQueue<Runnable>()));
  > }
  > ```

- `CachedThreadPool`： 

  > ```java
  > public static ExecutorService newCachedThreadPool() {
  >     return new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, TimeUnit.SECONDS,
  >                                   new SynchronousQueue<Runnable>());
  > }
  > ```
  >
  > - corePool 为 0 意味着所有的任务一提交就会加入到阻塞队列中
  >
  > - 阻塞队列采用的 SynchronousQueue 是一个没有元素的阻塞队列
  >
  >   > 问题： 
  >   >
  >   > - 如果主线程提交任务的速度远远大于 CachedThreadPool 的处理速度，则会不断地创建新线程来执行任务
  >   > - 进而可能导致系统耗尽 CPU 和内存资源，所以在**使用该线程池时，要注意控制并发的任务数，否则创建大量的线程可能导致严重的性能问题** 

### 4. 任务提交

> 线程池 ThreadPoolExecutor 任务提交的两种方式： `Executor.execute()、ExecutorService.submit()` 
>
> > ExecutorService.submit() 可以获取该任务执行的Future

以 Executor.execute() 为例，看看线程池的任务提交经历了哪些过程： 

- 定义： 

  ```java
  public interface Executor {
      void execute(Runnable command);
  }
  ```

- ThreadPoolExecutor 提供实现： 

  ```java
  public void execute(Runnable command) {
      if (command == null)
          throw new NullPointerException();
      int c = ctl.get();
      if (workerCountOf(c) < corePoolSize) {
          if (addWorker(command, true))
              return;
          c = ctl.get();
      }
      if (isRunning(c) && workQueue.offer(command)) {
          int recheck = ctl.get();
          if (! isRunning(recheck) && remove(command))
              reject(command);
          else if (workerCountOf(recheck) == 0)
              addWorker(null, false);
      }
      else if (!addWorker(command, false))
          reject(command);
  }
  ```

  > 执行流程：
  >
  > 1. 如果线程池当前线程数小于 corePoolSize，则调用 addWorker 创建新线程执行任务，成功返回true，失败执行步骤 2
  >
  > 2. 如果线程池处于 RUNNING 状态，则尝试加入阻塞队列，如果加入阻塞队列成功，则尝试进行Double Check，如果加入失败，则执行步骤 3
  >
  >    > Double Check 的目的： 判断加入到阻塞队中的线程是否可以被执行
  >    >
  >    > - 如果线程池不是 RUNNING 状态，则调用 remove() 方法从阻塞队列中删除该任务，然后调用reject() 方法处理任务
  >    > - 否则需要确保还有线程执行
  >
  > 3. 如果线程池不是 RUNNING 状态或者加入阻塞队列失败，则尝试创建新线程直到 maxPoolSize，如果失败，则调用 reject() 方法运行相应的拒绝策略

- `addWorker()`： 创建新线程执行任务

  > 当前线程数是根据**ctl**变量来获取的，调用 workerCountOf(ctl) 获取低29位即可
  >
  > ```java
  > private boolean addWorker(Runnable firstTask, boolean core) {
  >     retry:
  >     for (;;) {
  >         int c = ctl.get();
  >         // 获取当前线程状态
  >         int rs = runStateOf(c);
  >         if (rs >= SHUTDOWN &&  ! (rs == SHUTDOWN &&
  >             	firstTask == null && ! workQueue.isEmpty()))
  >             return false;
  >         // 内层循环，worker + 1
  >         for (;;) {
  >             // 线程数量
  >             int wc = workerCountOf(c);
  >             // 如果当前线程数大于线程最大上限CAPACITY  return false
  >             // 若core == true，则与corePoolSize 比较
  >             //否则与maximumPoolSize ，大于 return false
  >             if (wc >= CAPACITY || 
  >                 	wc >= (core ? corePoolSize : maximumPoolSize))
  >                 return false;
  >             // worker + 1,成功跳出retry循环
  >             if (compareAndIncrementWorkerCount(c))
  >                 break retry;
  >             // CAS add worker 失败，再次读取ctl
  >             c = ctl.get();
  >             // 如果状态不等于之前获取的state，跳出内层循环，继续去外层循环判断
  >             if (runStateOf(c) != rs)
  >                 continue retry;
  >         }
  >     }
  >     boolean workerStarted = false;
  >     boolean workerAdded = false;
  >     Worker w = null;
  >     try {
  >         // 新建线程：Worker
  >         w = new Worker(firstTask);
  >         // 当前线程
  >         final Thread t = w.thread;
  >         if (t != null) {
  >             // 获取主锁：mainLock
  >             final ReentrantLock mainLock = this.mainLock;
  >             mainLock.lock();
  >             try {
  >                 // 线程状态
  >                 int rs = runStateOf(ctl.get());
  >                 // rs < SHUTDOWN ==> 线程处于RUNNING状态
  >                 // 或者线程处于SHUTDOWN状态，且firstTask == null
  >                 //（可能是workQueue中仍有未执行完成的任务，创建没有初始任务的worker线程执行）
  >                 if (rs < SHUTDOWN || (rs == SHUTDOWN && firstTask == null)) {
  >                     // 当前线程已经启动，抛出异常
  >                     if (t.isAlive()) // precheck that t is startable
  >                         throw new IllegalThreadStateException();
  >                     // workers是一个HashSet<Worker>
  >                     workers.add(w);
  >                     // 设置最大的池大小largestPoolSize，workerAdded设置为true
  >                     int s = workers.size();
  >                     if (s > largestPoolSize)
  >                         largestPoolSize = s;
  >                     workerAdded = true;
  >                 }
  >             } finally {
  >                 mainLock.unlock(); // 释放锁
  >             }
  >             if (workerAdded) {
  >                 t.start(); // 启动线程
  >                 workerStarted = true;
  >             }
  >         }
  >     } finally {
  >         if (! workerStarted) // 线程启动失败
  >             addWorkerFailed(w);
  >     }
  >     return workerStarted;
  > }
  > ```
  >
  > 执行流程：
  >
  > - 判断当前线程是否可以添加任务，如果可以则进行下一步，否则 return false： 
  >
  >   - `rs >= SHUTDOWN`： 表示当前线程处于SHUTDOWN ，STOP、TIDYING、TERMINATED状态
  >
  >   - `rs == SHUTDOWN, firstTask != null`： 不允许添加线程，因为线程处于 SHUTDOWN 状态，不允许添加任务
  >
  >   - `rs == SHUTDOWN , firstTask == null，workQueue.isEmpty() == true`：不允许添加线程，因为firstTask == null是为了添加一个没有任务的线程然后再从workQueue中获取任务
  >
  > - 内嵌循环，通过 CAS worker + 1
  >
  > - 获取主锁 mailLock，如果线程池处于 RUNNING 状态获取处于 SHUTDOWN 状态且 firstTask == null，则将任务添加到workers Queue中，然后释放主锁mainLock，然后启动线程，然后return true，如果中途失败导致workerStarted= false，则调用 addWorkerFailed() 方法进行处理
  >
  > 在 execute() 方法中，有三处调用了该方法：
  >
  > - 第一次：`workerCountOf(c) < corePoolSize ==> addWorker(command, true)`
  > - 第二次：加入阻塞队列进行 Double Check 时，`else if (workerCountOf(recheck) == 0) ==>addWorker(null, false)`
  > - 第三次：线程池不是 RUNNING 状态或者加入阻塞队列失败：`else if (!addWorker(command, false))`

- `Woker` 内部类： 

  > ```java
  > private final class Worker extends AbstractQueuedSynchronizer
  >             implements Runnable {
  >     private static final long serialVersionUID = 6138294804551838833L;
  >     final Thread thread; // task 的thread
  >     Runnable firstTask; // 运行的任务task
  >     volatile long completedTasks;
  >     Worker(Runnable firstTask) {
  >         //设置 AQS 的同步状态 state，是一个计数器，大于0代表锁已经被获取
  >         setState(-1);
  >         this.firstTask = firstTask;
  >         // 利用ThreadFactory和 Worker这个Runnable创建的线程对象
  >         this.thread = getThreadFactory().newThread(this);
  >     }
  >     // 任务执行
  >     public void run() {
  >         runWorker(this);
  >     }
  > }
  > ```

- `runWorker()`： 

  > ```java
  > final void runWorker(Worker w) {
  >     Thread wt = Thread.currentThread(); // 当前线程
  >     Runnable task = w.firstTask; // 要执行的任务
  >     w.firstTask = null;
  >     // 释放锁，运行中断
  >     w.unlock(); // allow interrupts
  >     boolean completedAbruptly = true;
  >     try {
  >         while (task != null || (task = getTask()) != null) {
  >             // worker 获取锁
  >             w.lock();
  >             // 确保只有当线程是stoping时，才会被设置为中断，否则清楚中断标示
  >             // 如果线程池状态 >= STOP ,且当前线程没有设置中断状态，则wt.interrupt()
  >             // 如果线程池状态 < STOP，但是线程已经中断了，再次判断线程池是否 >= STOP，如果是 wt.interrupt()
  >             if ((runStateAtLeast(ctl.get(), STOP) ||
  >                     (Thread.interrupted() &&
  >                             runStateAtLeast(ctl.get(), STOP))) &&
  >                     		!wt.isInterrupted())
  >                 wt.interrupt();
  >             try {
  >                 // 自定义方法
  >                 beforeExecute(wt, task);
  >                 Throwable thrown = null;
  >                 try {
  >                     // 执行任务
  >                     task.run();
  >                 } catch (RuntimeException x) {
  >                     thrown = x; throw x;
  >                 } catch (Error x) {
  >                     thrown = x; throw x;
  >                 } catch (Throwable x) {
  >                     thrown = x; throw new Error(x);
  >                 } finally {
  >                     afterExecute(task, thrown);
  >                 }
  >             } finally {
  >                 task = null;
  >                 // 完成任务数 + 1
  >                 w.completedTasks++;
  >                 // 释放锁
  >                 w.unlock();
  >             }
  >         }
  >         completedAbruptly = false;
  >     } finally {
  >         processWorkerExit(w, completedAbruptly);
  >     }
  > }
  > ```
  >
  > 运行流程
  >
  > 1. 根据 worker 获取要执行的任务 task，然后调用 unlock() 方法释放锁
  >
  >    > 释放锁的目的： 在于中断，因为在new Worker时，设置的state为-1，调用unlock()方法可以将state设置为0，这里主要原因就在于interruptWorkers()方法只有在state >= 0时才会执行
  >
  > 2. 通过getTask()获取执行的任务，调用task.run()执行，当然在执行之前会调用worker.lock()上锁，执行之后调用worker.unlock()放锁
  >
  > 3. 在任务执行前后，可以根据业务场景自定义beforeExecute() 和 afterExecute()方法，则两个方法在ThreadPoolExecutor中是空实现
  >
  > 4. 如果线程执行完成，则会调用getTask()方法从阻塞队列中获取新任务，如果阻塞队列为空，则根据是否超时来判断是否需要阻塞
  >
  > 5. task == null或者抛出异常（beforeExecute()、task.run()、afterExecute()均有可能）导致worker线程终止，则调用processWorkerExit()方法处理worker退出流程

- `getTask()`： 

  > ```java
  > private Runnable getTask() {
  >     boolean timedOut = false; // Did the last poll() time out?
  >     for (;;) {
  >         // 线程池状态
  >         int c = ctl.get();
  >         int rs = runStateOf(c);
  >         // 线程池中状态 >= STOP 或者 线程池状态 == SHUTDOWN且阻塞队列为空，
  >         //则worker - 1，return null
  >         if (rs >= SHUTDOWN && (rs >= STOP || workQueue.isEmpty())) {
  >             decrementWorkerCount();
  >             return null;
  >         }
  >         int wc = workerCountOf(c);
  >         // 判断是否需要超时控制
  >         boolean timed = allowCoreThreadTimeOut || wc > corePoolSize;
  > 
  >         if ((wc > maximumPoolSize || (timed && timedOut)) && 
  >             	(wc > 1 || workQueue.isEmpty())) {
  >             if (compareAndDecrementWorkerCount(c))
  >                 return null;
  >             continue;
  >         }
  >         try {
  > 
  >             // 从阻塞队列中获取task
  >             // 如果需要超时控制，则调用poll()，否则调用take()
  >             Runnable r = timed ?
  >                     workQueue.poll(keepAliveTime, TimeUnit.NANOSECONDS) :
  >                     workQueue.take();
  >             if (r != null)
  >                 return r;
  >             timedOut = true;
  >         } catch (InterruptedException retry) {
  >             timedOut = false;
  >         }
  >     }
  > }
  > ```
  >
  > - timed == true，调用 poll() 方法，如果 在keepAliveTime 时间内还没有获取task的话，则返回null，继续循环
  > - timed == false，则调用take()方法，该方法为一个阻塞方法，没有任务时会一直阻塞挂起，直到有任务加入时对该线程唤醒，返回任务

- `processWorkerExit()`： 

  > ```java
  > private void processWorkerExit(Worker w, boolean completedAbruptly) {
  >     // true：用户线程运行异常,需要扣减
  >     // false：getTask方法中扣减线程数量
  >     if (completedAbruptly)
  >         decrementWorkerCount();
  >     // 获取主锁
  >     final ReentrantLock mainLock = this.mainLock;
  >     mainLock.lock();
  >     try {
  >         completedTaskCount += w.completedTasks;
  >         // 从HashSet中移出worker
  >         workers.remove(w);
  >     } finally {
  >         mainLock.unlock();
  >     }
  >     // 有worker线程移除，可能是最后一个线程退出需要尝试终止线程池
  >     tryTerminate();
  >     int c = ctl.get();
  >     // 如果线程为running或shutdown状态，即tryTerminate()没有成功终止线程池，则判断是否有必要一个worker
  >     if (runStateLessThan(c, STOP)) {
  >         // 正常退出，计算min：需要维护的最小线程数量
  >         if (!completedAbruptly) {
  >             // allowCoreThreadTimeOut 默认false：是否需要维持核心线程的数量
  >             int min = allowCoreThreadTimeOut ? 0 : corePoolSize;
  >             // 如果min ==0 或者workerQueue为空，min = 1
  >             if (min == 0 && ! workQueue.isEmpty())
  >                 min = 1;
  > 
  >             // 如果线程数量大于最少数量min，直接返回，不需要新增线程
  >             if (workerCountOf(c) >= min)
  >                 return; // replacement not needed
  >         }
  >         // 添加一个没有firstTask的worker
  >         addWorker(null, false);
  >     }
  > }
  > ```
  >
  > 首先completedAbruptly的值来判断是否需要对线程数-1处理，如果completedAbruptly == true，说明在任务运行过程中出现了异常，那么需要进行减1处理，否则不需要，因为减1处理在getTask()方法中处理了。然后从HashSet中移出该worker，过程需要获取mainlock。然后调用tryTerminate()方法处理，该方法是对最后一个线程退出做终止线程池动作。如果线程池没有终止，那么线程池需要保持一定数量的线程，则通过addWorker(null,false)新增一个空的线程。

### 5. 线程终止

线程池 ThreadPoolExecutor 提供的关闭方式： 

- `shutdown()`： 按过去执行已提交任务的顺序发起一个有序的关闭，但是不接受新任务

  > ```java
  > public void shutdown() {
  >     final ReentrantLock mainLock = this.mainLock;
  >     mainLock.lock();
  >     try {
  >         checkShutdownAccess();
  >         // 推进线程状态
  >         advanceRunState(SHUTDOWN);
  >         // 中断空闲的线程
  >         interruptIdleWorkers();
  >         // 交给子类实现
  >         onShutdown();
  >     } finally {
  >         mainLock.unlock();
  >     }
  >     tryTerminate();
  > }
  > ```

- `shutdownNow()`： 尝试停止所有的活动执行任务、暂停等待任务的处理，并返回等待执行的任务列表

  > ```java
  > public List<Runnable> shutdownNow() {
  >     List<Runnable> tasks;
  >     final ReentrantLock mainLock = this.mainLock;
  >     mainLock.lock();
  >     try {
  >         checkShutdownAccess();
  >         advanceRunState(STOP);
  >         // 中断所有线程
  >         interruptWorkers();
  >         // 返回等待执行的任务列表
  >         tasks = drainQueue();
  >     } finally {
  >         mainLock.unlock();
  >     }
  >     tryTerminate();
  >     return tasks;
  > }
  > 
  > private void interruptWorkers() {
  >     final ReentrantLock mainLock = this.mainLock;
  >     mainLock.lock();
  >     try {
  >         for (Worker w : workers)
  >             w.interruptIfStarted();
  >     } finally {
  >         mainLock.unlock();
  >     }
  > }
  > 
  > private List<Runnable> drainQueue() {
  >     BlockingQueue<Runnable> q = workQueue;
  >     ArrayList<Runnable> taskList = new ArrayList<Runnable>();
  >     q.drainTo(taskList);
  >     if (!q.isEmpty()) {
  >         for (Runnable r : q.toArray(new Runnable[0])) {
  >             if (q.remove(r))
  >                 taskList.add(r);
  >         }
  >     }
  >     return taskList;
  > }
  > ```

## 2. ScheduledThreadPoolExecutor

- `ScheduledThreadPoolExecutor`： 继承 ThreadPoolExecutor 且实现了 ScheduledExecutorService 接口

  > 相当于提供了“延迟”和“周期执行”功能的ThreadPoolExecutor

- 四个构造方法： 

  ```java
  public ScheduledThreadPoolExecutor(int corePoolSize) {
      super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,new DelayedWorkQueue());
  }
  
  public ScheduledThreadPoolExecutor(int corePoolSize,ThreadFactory threadFactory) {
      super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), threadFactory);
  }
  
  public ScheduledThreadPoolExecutor(int corePoolSize,
                                     RejectedExecutionHandler handler) {
      super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), handler);
  }
  
  public ScheduledThreadPoolExecutor(int corePoolSize, ThreadFactory threadFactory,
                                     RejectedExecutionHandler handler) {
      super(corePoolSize, Integer.MAX_VALUE, 0, NANOSECONDS,
              new DelayedWorkQueue(), threadFactory, handler);
  }
  ```

  

## 3. FutureTask

### 1. 简介

- 在 Executors 框架体系中，FutureTask 用来表示可获取结果的异步任务

- FutureTask 提供了启动和取消异步任务，查询异步任务是否计算结束及获取最终的异步任务结果的常用方法

- 通过 `get()` 方法来获取异步任务的结果，但会阻塞当前线程直至异步任务执行结束

  > 一旦任务执行结束，任务不能重新启动或取消，除非调用 `runAndReset()` 方法

  ```java
  private static final int NEW          = 0;
  private static final int COMPLETING   = 1;
  private static final int NORMAL       = 2;
  private static final int EXCEPTIONAL  = 3;
  private static final int CANCELLED    = 4;
  private static final int INTERRUPTING = 5;
  private static final int INTERRUPTED  = 6;
  ```

FutureTask 的三种状态： 

- **未启动**： 当创建一个 FutureTask，但没有执行 FutureTask.run() 方法之前，FutureTask 处于未启动状态

- **已启动**： FutureTask.run() 方法被执行的过程中，FutureTask 处于已启动状态

- **已完成**： FutureTask.run() 方法执行结束，或调用 FutureTask.cancel(...) 方法取消任务，或在执行任务期间抛出异常，这些情况都称之为 FutureTask 的已完成状态

  ![](../../pics/concurrent/concurrent_42.jpg)

### 2. 方法

- `get()` 方法： 
  - 当 FutureTask 处于**未启动或已启动状态**时，执行 FutureTask.get() 方法将导致调用线程阻塞
  - 当 FutureTask 处于**已完成状态**，调用 FutureTask.get() 方法将导致调用线程立即返回结果或抛出异常

- `cancel()` 方法： 

  - 当 FutureTask 处于**未启动状态**时，执行 FutureTask.cancel() 方法将使此任务永远不会执行

  - 当 FutureTask 处于**已启动状态**时，执行 FutureTask.cancel(true) 方法将以中断线程的方式来阻止任务继续进行

    > 如果执行 FutureTask.cancel(false) 将不会对正在执行任务的线程有任何影响

  - 当FutureTask处于**已完成状态**时，执行 FutureTask.cancel(...) 方法将返回 false

  ![](../../pics/concurrent/concurrent_43.jpg)

### 3. 应用场景

> - FutureTask 实现了 Future 接口与 Runnable 接口，因此 FutureTask 可以交给 Executor 执行，也可以由调用的线程直接执行（FutureTask.run()）
> - FutureTask 的获取也可以通过 ExecutorService.submit() 方法返回一个 FutureTask 对象，然后在通过FutureTask.get() 或 FutureTask.cancel 方法

**应用场景**： 当一个线程需要等待另一个线程把某个任务执行完后才能继续执行，可以使用 FutureTask

# 七、原子类(Atomic)

## 1. 原子更新基本类型

工具类：

- `AtomicBoolean`：以原子更新的方式更新 boolean
- `AtomicInteger`：以原子更新的方式更新 Integer
- `AtomicLong`：以原子更新的方式更新 Long

这几个类的用法基本一致，以 AtomicInteger 为例总结常用的方法： 

- `addAndGet(int delta)`：以原子方式将输入的数值与实例中原本的值相加，并返回最后的结果

- `incrementAndGet()`：以原子的方式将实例中的原值进行加 1 操作，并返回最终相加后的结果

- `getAndSet(int newValue)`：将实例中的值更新为新值，并返回旧值

- `getAndIncrement()`：以原子的方式将实例中的原值加1，返回的是自增前的旧值

源码分析：

```java
public final int getAndIncrement() {
    return unsafe.getAndAddInt(this, valueOffset, 1);
}

private static final Unsafe unsafe = Unsafe.getUnsafe();
```

## 2. 原子更新数组类型

工具类：

- `AtomicIntegerArray`：原子更新整型数组中的元素
- `AtomicLongArray`：原子更新长整型数组中的元素
- `AtomicReferenceArray`：原子更新引用类型数组中的元素

这几个类的用法一致，以 AtomicIntegerArray 总结常用的方法：

- `addAndGet(int i, int delta)`：以原子更新的方式将数组中索引为 `i` 的元素与输入值相加
- `getAndIncrement(int i)`：以原子更新的方式将数组中索引为 `i` 的元素自增加 1
- `compareAndSet(int i, int expect, int update)`：将数组中索引为 `i` 的位置的元素进行更新

## 3. 原子更新引用类型

工具类：

- `AtomicReference`：原子更新引用类型
- `AtomicReferenceFieldUpdater`：原子更新引用类型里的字段
- `AtomicMarkableReference`：原子更新带有标记位的引用类型

demo： 

```java
public class AtomicDemo {

    private static AtomicReference<User> reference = new AtomicReference<>();

    public static void main(String[] args) {
        User user1 = new User("a", 1);
        reference.set(user1);
        User user2 = new User("b",2);
        User user = reference.getAndSet(user2);
        System.out.println(user);
        System.out.println(reference.get());
    }

    static class User {
        private String userName;
        private int age;

        public User(String userName, int age) {
            this.userName = userName;
            this.age = age;
        }

        @Override
        public String toString() {
            return "User{" +
                    "userName='" + userName + '\'' +
                    ", age=" + age +
                    '}';
        }
    }
}

//输出结果：
User{userName='a', age=1}
User{userName='b', age=2}
```

## 4. 原子更新字段类型

工具类： 

- `AtomicIntegeFieldUpdater`：原子更新整型字段类
- `AtomicLongFieldUpdater`：原子更新长整型字段类
- `AtomicStampedReference`：原子更新引用类型，该更新方式带有版本号从而解决 CAS 的 ABA 问题

想使用原子更新字段需要两步操作：

- 原子更新字段类都是抽象类，通过静态方法 `newUpdater` 来创建一个更新器，且需设置想要更新的类和属性
- 更新类的属性必须使用 `public volatile` 进行修饰

demo： 

```java
public class AtomicDemo {

    private static AtomicIntegerFieldUpdater updater 
        = AtomicIntegerFieldUpdater.newUpdater(User.class,"age");
    public static void main(String[] args) {
        User user = new User("a", 1);
        int oldValue = updater.getAndAdd(user, 5);
        System.out.println(oldValue);
        System.out.println(updater.get(user));
    }

    static class User {
        private String userName;
        public volatile int age;

        public User(String userName, int age) {
            this.userName = userName;
            this.age = age;
        }
    }
} 

//输出结果：
1
6
```





# 八、并发工具

## 1. 倒计时器CountDownLatch





## 2. 循环栅栏CyclicBarrier





## 3. 资源访问控制Semaphore





## 4. 数据交换Exchanger





# 九、并发实践：生产者与消费者问题

## 1. 使用 Object 的 wait/notifyAll 实现

### 1. wait/notify 消息通知的潜在问题

- 





### 2. wait/notifyAll 实现生产者-消费者





## 2. 使用 Lock Condition 的 await/signalAll 实现





## 3. 使用 BlockingQueue 实现







