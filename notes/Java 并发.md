### [JUC 推荐阅读](https://www.cnblogs.com/pony1223/category/1241236.html)  

### [Java并发编程总结](http://oldblog.csdn.net/column/details/java-concurrent-prog.html) 



# 一、线程状态转换

<img src="../pics//ace830df-9919-48ca-91b5-60b193f593d2.png"/>

## 1. 新建（New）

创建后尚未启动

>  当一个Thread类或其子类的对象被声明并创建时，新生的线程对象处于新建状态

## 2. 就绪

> 处于新建状态的线程被start()后，将进入线程队列等待CPU时间片，此时它已具备了运行的条件

## 3. 可运行（Runnable）

可能正在运行，也可能正在等待 CPU 时间片

包含了操作系统线程状态中的 Running 和 Ready

>  当就绪的线程被调度并获得处理器资源时,便进入运行状态， run()方法定义了线程的操作和功能

## 4. 阻塞（Blocking）

等待获取一个排它锁，如果其线程释放了锁就会结束此状态

> 在某种特殊情况下，被人为挂起或执行输入输出操作时，让出 CPU 并临时中止自己的执行，进入阻塞状态

## 5. 无限期等待（Waiting）

等待其它线程显式地唤醒，否则不会被分配 CPU 时间片

| 进入方法                              | 退出方法                                 |
| --------------------------------- | ------------------------------------ |
| 没有设置 Timeout 参数的 Object.wait() 方法 | Object.notify() / Object.notifyAll() |
| 没有设置 Timeout 参数的 Thread.join() 方法 | 被调用的线程执行完毕                           |
| LockSupport.park() 方法             | -                                    |

## 6. 限期等待（Timed Waiting）

无需等待其它线程显式地唤醒，在一定时间之后会被系统自动唤醒

调用 Thread.sleep() 方法使线程进入限期等待状态时，常常用“使一个线程睡眠”进行描述

调用 Object.wait() 方法使线程进入限期等待或者无限期等待时，常常用“挂起一个线程”进行描述

睡眠和挂起是用来描述行为，而阻塞和等待用来描述状态

阻塞和等待的区别在于，阻塞是被动的，它是在等待获取一个排它锁。而等待是主动的，通过调用 Thread.sleep() 和 Object.wait() 等方法进入

| 进入方法                             | 退出方法                                     |
| -------------------------------- | ---------------------------------------- |
| Thread.sleep() 方法                | 时间结束                                     |
| 设置了 Timeout 参数的 Object.wait() 方法 | 时间结束 / Object.notify() / Object.notifyAll() |
| 设置了 Timeout 参数的 Thread.join() 方法 | 时间结束 / 被调用的线程执行完毕                        |
| LockSupport.parkNanos() 方法       | -                                        |
| LockSupport.parkUntil() 方法       | -                                        |

## 7. 死亡（Terminated）

可以是线程结束任务之后自己结束，或者产生了异常而结束。

> 线程完成了它的全部工作或线程被提前强制性地中止   



![](../pics/thread.png)

# 二、创建并使用线程

有四种使用线程的方法：

- 实现 Runnable 接口；
- 实现 Callable 接口；
- 继承 Thread 类。
- 线程池

实现 Runnable 和 Callable 接口的类只能当做一个可以在线程中运行的任务，不是真正意义上的线程，因此最后还需要通过 Thread 来调用。可以说任务是通过线程驱动从而执行的

## 1. 实现 Runnable 接口

需要实现 run() 方法。

通过 Thread 调用 start() 方法来启动线程。

```java
public class MyRunnable implements Runnable {
    public void run() {
        // ...
    }
}
```

```java
public static void main(String[] args) {
    MyRunnable instance = new MyRunnable();
    Thread thread = new Thread(instance);
    thread.start();
}
```

## 2. 实现 Callable 接口

- 实现 Callable 接口， 相较于实现 Runnable 接口的方式，方法可以有返回值，并且可以抛出异常
- 执行 Callable 方式，返回值通过 FutureTask 进行封装（FutureTask 是  Future 接口的实现类）

```java
public class TestCallable {
	public static void main(String[] args) {
		ThreadDemo td = new ThreadDemo();
		//1.执行 Callable 方式，需要 FutureTask 实现类的支持，用于接收运算结果。
		FutureTask<Integer> result = new FutureTask<>(td);
		new Thread(result).start();
		//2.接收线程运算后的结果
		try {
			Integer sum = result.get();  //FutureTask 可用于 闭锁
			System.out.println(sum);
			System.out.println("------------------------------------");
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}
}

class ThreadDemo implements Callable<Integer>{
	@Override
	public Integer call() throws Exception {
		int sum = 0;
		for (int i = 0; i <= 100000; i++) {
			sum += i;
		}
		return sum;
	}
}
```

## 3. 继承 Thread 类

同样也是需要实现 run() 方法，因为 Thread 类也实现了 Runable 接口。

当调用 start() 方法启动一个线程时，虚拟机会将该线程放入就绪队列中等待被调度，当一个线程被调度时会执行该线程的 run() 方法。

```java
public class MyThread extends Thread {
    public void run() {
        // ...
    }
}
```

```java
public static void main(String[] args) {
    MyThread mt = new MyThread();
    mt.start();
}
```

##4. 线程池(Executor为根接口)

>  线程池可以解决两个不同问题：
>
>  - 减少了每个任务调用的开销，通常可以在执行大量异步任务时提供增强的性能，并且可以提供绑定和管理资源（包括执行任务集时使用的线程）的方法
>  - 每个 ThreadPoolExecutor 还维护着一些基本的统计数据，如完成的任务数

1. **线程池**：提供一个线程队列，保存所有等待状态的线程，避免创建与销毁额外开销，提高响应速度

2. 线程池的体系结构：

   `java.util.concurrent.Executor`： 负责线程的使用与调度的根接口

   ExecutorService 子接口： 线程池的主要接口
   * ThreadPoolExecutor 线程池的实现类
    * ScheduledExecutorService 子接口：负责线程的调度
       * ScheduledThreadPoolExecutor ：继承 ThreadPoolExecutor， 实现 ScheduledExecutorService

3. 工具类 : `Executors `

   - `ExecutorService newFixedThreadPool()` : 创建固定大小的线程池
   - `ExecutorService newCachedThreadPool()`: 缓存线程池，线程池的数量不固定，可以根据需求自动的更改数量
   - `ExecutorService newSingleThreadExecutor()` : 创建单个线程池，线程池中只有一个线程
   - `ScheduledExecutorService newScheduledThreadPool()` : 创建固定大小的线程，可以延迟或定时的执行任务

```java
public class TestThreadPool {
	public static void main(String[] args) throws Exception {
		//1. 创建线程池
		ExecutorService pool = Executors.newFixedThreadPool(5);
		List<Future<Integer>> list = new ArrayList<>();
		for (int i = 0; i < 10; i++) {
			//2. 为线程池中的线程分配任务
			Future<Integer> future = pool.submit(new Callable<Integer>(){
				@Override
				public Integer call() throws Exception {
					int sum = 0;
					for (int i = 0; i <= 100; i++) {
						sum += i;
					}
					return sum;
				}
			});
			list.add(future);
		}
        //3. 关闭线程池
		pool.shutdown();
		for (Future<Integer> future : list) {
			System.out.println(future.get());
		}
	}
}
```

## 5. 实现接口 VS 继承 Thread

**实现接口会更好一些**，因为：

- **Java 不支持多重继承**，因此继承了 Thread 类就无法继承其它类，但是可以实现多个接口；
- 类可能只要求可执行就行，**继承整个 Thread 类开销过大**

# 三、基础线程机制

## 1. Executor

**Executor 管理多个异步任务的执行**，无需程序员显式地管理线程的生命周期

> 异步： 指多个任务的执行互不干扰，不需要进行同步操作

主要有三种 Executor：

- `CachedThreadPool`：一个任务创建一个线程(无界线程池，可以进行自动线程回收)
- `FixedThreadPool`：所有任务只能使用固定大小的线程(固定大小线程池)
- `SingleThreadExecutor`：相当于大小为 1 的 FixedThreadPool(单个后台线程)

```java
public static void main(String[] args) {
    ExecutorService executorService = Executors.newCachedThreadPool();
    for (int i = 0; i < 5; i++) {
        executorService.execute(new MyRunnable());
    }
    executorService.shutdown();
}
```

**线程调度：** 

```java
public static void main(String[] args) throws Exception {
    ScheduledExecutorService pool = Executors.newScheduledThreadPool(5);
    for (int i = 0; i < 5; i++) {
        Future<Integer> result = pool.schedule(new Callable<Integer>(){
            @Override
            public Integer call() throws Exception {
                int num = new Random().nextInt(100);//生成随机数
                System.out.println(Thread.currentThread().getName() 
                                   + " : " + num);
                return num;
            }
        }, 1, TimeUnit.SECONDS);
        System.out.println(result.get());
    }
    pool.shutdown();
}
```

## 2. Daemon

- 定义： **守护线程是程序运行时在后台提供服务的线程**

- 当所有非守护线程结束时，程序也就终止，同时会杀死所有守护线程

- **main() 属于非守护线程**

- **守护线程用来服务用户线程**，通过**在start()方法前调用thread.setDaemon(true)把用户线程变成守护线程**

```java
public static void main(String[] args) {
    Thread thread = new Thread(new MyRunnable());
    thread.setDaemon(true);
}
```

## 3. sleep()

`Thread.sleep(millisec)` 方法会休眠当前正在执行的线程，millisec 单位为毫秒

```java
public void run() {
    try {
        Thread.sleep(3000);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
}
```

## 4. yield()

`Thread.yield()`： 

- 暂停当前正在执行的线程，把执行机会让给优先级相同或更高的线程
- 若队列中没有同优先级的线程，忽略此方法

```java
public void run() {
    Thread.yield();
}
```

## 5. stop

>  强制线程生命期结束

# 四、中断

一个线程执行完毕之后会自动结束，如果在运行过程中发生异常也会提前结束

## 1. InterruptedException

`interrupt()`： 用于中断线程，若线程处于阻塞、限期等待或无限期等待状态，则会抛出 InterruptedException，从而提前结束该线程，但不能中断 I/O 阻塞和 synchronized 锁阻塞

```java
//线程调用了 Thread.sleep() 方法，此时调用  Thread.interrupt() 会抛出 InterruptedException
//从而提前结束线程，不执行之后的语句
public static void main(String[] args) throws InterruptedException {
    Thread thread1 = new MyThread1();
    thread1.start();
    thread1.interrupt();
    System.out.println("Main run");
}

public class InterruptExample {
    private static class MyThread1 extends Thread {
        @Override
        public void run() {
            try {
                Thread.sleep(2000);
                System.out.println("Thread run");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

//结果打印
Main run
java.lang.InterruptedException: sleep interrupted
    at java.lang.Thread.sleep(Native Method)
    at InterruptExample.lambda$main$0(InterruptExample.java:5)
    at InterruptExample$$Lambda$1/713338599.run(Unknown Source)
    at java.lang.Thread.run(Thread.java:745)
```

## 2. interrupted()

调用 interrupt() 方法会设置线程的中断标记，此时在循环体中使用 interrupted() 方法来判断线程是否处于中断状态，从而提前结束线程

```java
public static void main(String[] args) throws InterruptedException {
    Thread thread2 = new MyThread2();
    thread2.start();
    thread2.interrupt();
}

public class InterruptExample {
    private static class MyThread2 extends Thread {
        @Override
        public void run() {
            while (!interrupted()) {
                // ..
            }
            System.out.println("Thread end");
        }
    }
}

//结果
Thread end
```

## 3. Executor 的中断操作

- `shutdown()`： 会等待线程都执行完毕之后再关闭
- `shutdownNow()`： 相当于调用每个线程的 interrupt() 方法

```java
public static void main(String[] args) {
    ExecutorService executorService = Executors.newCachedThreadPool();
    executorService.execute(() -> {
        try {
            Thread.sleep(2000);
            System.out.println("Thread run");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    });
    executorService.shutdownNow();
    System.out.println("Main run");
}

//结果
Main run
java.lang.InterruptedException: sleep interrupted
   at java.lang.Thread.sleep(Native Method)
   at ExecutorInterruptExample.lambda$main$0(ExecutorInterruptExample.java:9)
   at ExecutorInterruptExample$$Lambda$1/1160460865.run(Unknown Source)
   at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
   at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
   at java.lang.Thread.run(Thread.java:745)
```

**中断 Executor 的一个线程**： 

- 使用 submit() 方法提交一个线程，得到一个 Future<?> 返回对象
- 调用该对象的 cancel(true) 方法来中断该线程

```java
Future<?> future = executorService.submit(() -> {
    // ..
});
future.cancel(true);
```

# 五、互斥同步

**java 锁机制**： 控制多个线程对共享资源的互斥访问

- JVM 实现的 synchronized
- JDK 实现的 ReentrantLock

## 1. synchronized

**1. 同步一个代码块** 

```java
public class SynchronizedExample {
    public void func1() {
        synchronized (this) {
            for (int i = 0; i < 10; i++) {
                System.out.print(i + " ");
            }
        }
    }
}
```

- 调用的是同一个对象的同步代码块，因此这两个线程会进行同步，当一个线程进入同步语句块时，另一个线程就必须等待

  ```java
  public static void main(String[] args) {
      SynchronizedExample e1 = new SynchronizedExample();
      ExecutorService executorService = Executors.newCachedThreadPool();
      executorService.execute(() -> e1.func1());
      executorService.execute(() -> e1.func1());
  }
  
  //结果
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
  ```

- 两个线程调用不同对象的同步代码块，因此这两个线程不会同步，交叉执行

  ```java
  public static void main(String[] args) {
      SynchronizedExample e1 = new SynchronizedExample();
      SynchronizedExample e2 = new SynchronizedExample();
      ExecutorService executorService = Executors.newCachedThreadPool();
      executorService.execute(() -> e1.func1());
      executorService.execute(() -> e2.func1());
  }
  
  //结果
  0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9
  ```


**2. 同步一个方法** 

```java
public synchronized void func () {
    // ...
}
```

- 与同步代码块一样，**作用于同一个对象**

**3. 同步一个类** 

```java
public void func() {
    synchronized (SynchronizedExample.class) {
        // ...
    }
}
```

- **作用于整个类**，即两个线程调用同一个类的不同对象上的同步语句，也会进行同步

  ```java
  public class SynchronizedExample {
      public void func2() {
          synchronized (SynchronizedExample.class) {
              for (int i = 0; i < 10; i++) {
                  System.out.print(i + " ");
              }
          }
      }
  }
  ```

  ```java
  public static void main(String[] args) {
      SynchronizedExample e1 = new SynchronizedExample();
      SynchronizedExample e2 = new SynchronizedExample();
      ExecutorService executorService = Executors.newCachedThreadPool();
      executorService.execute(() -> e1.func2());
      executorService.execute(() -> e2.func2());
  }
  
  //结果
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
  ```

**4. 同步一个静态方法** 

```java
public synchronized static void fun() {
    // ...
}
```

- **作用于整个类**
- **非静态方法的锁默认为 this，静态方法的锁对应 Class 实例** 
- **一个静态同步方法获取 Class 实例锁后，其他静态同步方法必须等待该方法释放锁才能获取锁**

## 2. ReentrantLock（同步锁）

- ReentrantLock 是 java.util.concurrent（J.U.C）包中的锁

- ReentrantLock 实现 Lock 接口，并提供了与 synchronized 相同的互斥性和内存可见性

- 相较于synchronized 提供了更高的处理锁的灵活性

```java
public class LockExample {

    private Lock lock = new ReentrantLock();

    public void func() {
        lock.lock();
        try {
            for (int i = 0; i < 10; i++) {
                System.out.print(i + " ");
            }
        } finally {
            lock.unlock(); // 确保释放锁，从而避免发生死锁。
        }
    }
}

public static void main(String[] args) {
    LockExample lockExample = new LockExample();
    ExecutorService executorService = Executors.newCachedThreadPool();
    executorService.execute(() -> lockExample.func());
    executorService.execute(() -> lockExample.func());
}

//结果
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
```


## 3. 比较

**1. 锁的实现** 

synchronized 是 JVM 实现的，而 ReentrantLock 是 JDK 实现的

**2. 性能** 

新版本 Java 对 synchronized 进行了很多优化，例如自旋锁等，synchronized 与 ReentrantLock 大致相同

**3. 等待可中断** 

当持有锁的线程长期不释放锁的时候，正在等待的线程可以选择放弃等待，改为处理其他事情

**ReentrantLock 可中断，而 synchronized 不行**

**4. 公平锁** 

公平锁是指多个线程在等待同一个锁时，必须按照申请锁的时间顺序来依次获得锁

**synchronized 锁非公平，ReentrantLock 默认非公平，但也可以公平**

**5. 锁绑定多个条件** 

**一个 ReentrantLock 可以同时绑定多个 Condition 对象**

## 4. 使用选择

- 优先使用 synchronized
-  synchronized 是 JVM 实现的锁机制，JVM 原生地支持它，而 ReentrantLock 不是所有的 JDK 版本都支持
- **使用 synchronized 不用担心没有释放锁而导致死锁问题**，因为 JVM 会确保锁的释放

# 六、线程之间的协作

当多个线程可以一起工作去解决某个问题时，如果某些部分必须在其它部分之前完成，那么就需要对线程进行协调

## 1. join()

**在线程中调用另一个线程的 join() 方法，会将当前线程挂起，直到目标线程结束**

- 低优先级的线程也可以获得执行 

```java
//因为在 b 线程中调用了 a 线程的 join() 方法，b 线程会等待 a 线程结束才继续执行
//因此能够保证 a 线程的输出先于 b 线程的输出
public class JoinExample {

    private class A extends Thread {
        @Override
        public void run() {
            System.out.println("A");
        }
    }

    private class B extends Thread {
        private A a;
        B(A a) {
            this.a = a;
        }
        @Override
        public void run() {
            try {
                a.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("B");
        }
    }

    public void test() {
        A a = new A();
        B b = new B(a);
        b.start();
        a.start();
    }
    
    public static void main(String[] args) {
        JoinExample example = new JoinExample();
        example.test();
    }
}

//结果
A
B
```

## 2. wait() notify() notifyAll()

**notify 丢失(虚假唤醒)**： 

- 线程 A 与 B 被同一个 Object.wait() 挂起，但等待条件不同
- 假设线程 B 的条件被满足，执行一个 notify 操作
- JVM 从 Object.wait() 的多个线程（A/B）中挑选一个唤醒，不幸的选择了 A
- 但 A 的条件不满足，于是 A 继续挂起，B 仍然在等待被唤醒

**虚假唤醒的解决方式**： 

- 使用 `notifyall()`，避免使用

  > notifyall() 会唤醒所有线程，但只有一个线程能够得到锁，因此会带来大量的上下文切换和大量的竞争锁请求

- wait() 最好放在 while 循环中，以避免“虚假唤醒”的情形

  ```java
  synchronized(obj){
     while(<condition does not hold>){
       obj.wait();
     }
  }
  ```

**wait() 和 sleep() 的区别** 

- **wait() 是 Object 的方法，而 sleep() 是 Thread 的静态方法**
- **wait() 会释放锁，sleep() 不会**

## 3. Condition 的 await() signal() signalAll()

**Condition 类**： 位于 `java.util.concurrent`，可以使用 Lock 来获取 Condition 对象

- `await()` 方法使线程等待，但可以指定等待的条件，因此更加灵活
- `signal() 或 signalAll()` 方法唤醒等待的线程

```java
public class AwaitSignalExample {

    private Lock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();

    public void before() {
        lock.lock();
        try {
            System.out.println("before");
            condition.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public void after() {
        lock.lock();
        try {
            condition.await();
            System.out.println("after");
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            lock.unlock();
        }
    }
    
    public static void main(String[] args) {
        ExecutorService executorService = Executors.newCachedThreadPool();
        AwaitSignalExample example = new AwaitSignalExample();
        executorService.execute(() -> example.after());
        executorService.execute(() -> example.before());
    }
}

//结果
before
after
```

# 七、等待唤醒机制

###1. synchronized 实现

```java
//生产者消费者案例
public class TestProductorAndConsumer {
	public static void main(String[] args) {
		Clerk clerk = new Clerk();
		
		Productor pro = new Productor(clerk);
		Consumer cus = new Consumer(clerk);
		
		new Thread(pro, "生产者 A").start();
		new Thread(cus, "消费者 B").start();
		
		new Thread(pro, "生产者 C").start();
		new Thread(cus, "消费者 D").start();
	}
}

//店员
class Clerk{
	private int product = 0;
	//进货
	public synchronized void get(){
		while(product >= 1){//为了避免虚假唤醒问题，应该在循环中使用 wait()
			System.out.println("产品已满！");
			try {
				this.wait();
			} catch (InterruptedException e) {}
		}
		System.out.println(Thread.currentThread().getName() + " : " + ++product);
		this.notifyAll();
	}
	//卖货
	public synchronized void sale(){//product = 0; 循环次数：0
		while(product <= 0){
			System.out.println("缺货！");
			try {
				this.wait();
			} catch (InterruptedException e) {}
		}
		System.out.println(Thread.currentThread().getName() + " : " + --product);
		this.notifyAll();
	}
}

//生产者
class Productor implements Runnable{
	private Clerk clerk;

	public Productor(Clerk clerk) {
		this.clerk = clerk;
	}
	@Override
	public void run() {
		for (int i = 0; i < 20; i++) {
			try {
				Thread.sleep(200);
			} catch (InterruptedException e) {
			}
			clerk.get();
		}
	}
}

//消费者
class Consumer implements Runnable{
	private Clerk clerk;

	public Consumer(Clerk clerk) {
		this.clerk = clerk;
	}
	@Override
	public void run() {
		for (int i = 0; i < 20; i++) {
			clerk.sale();
		}
	}
}
```

###2. ReentrantLock 实现(Condition)

```java
//生产者消费者案例
public class TestProductorAndConsumerForLock {

	public static void main(String[] args) {
		Clerk clerk = new Clerk();

		Productor pro = new Productor(clerk);
		Consumer con = new Consumer(clerk);

		new Thread(pro, "生产者 A").start();
		new Thread(con, "消费者 B").start();

//		 new Thread(pro, "生产者 C").start();
//		 new Thread(con, "消费者 D").start();
	}
}

class Clerk {
	private int product = 0;

	private Lock lock = new ReentrantLock();
	private Condition condition = lock.newCondition();

	// 进货
	public void get() {
		lock.lock();
		try {
			if (product >= 1) { // 为了避免虚假唤醒，应该总是使用在循环中。
				System.out.println("产品已满！");
				try {
					condition.await();
				} catch (InterruptedException e) {}
			}
			System.out.println(Thread.currentThread().getName() + " : " + ++product);
			condition.signalAll();
		} finally {
			lock.unlock();
		}
	}

	// 卖货
	public void sale() {
		lock.lock();
		try {
			if (product <= 0) {
				System.out.println("缺货！");
				try {
					condition.await();
				} catch (InterruptedException e) {}
			}
			System.out.println(Thread.currentThread().getName() + " : " + --product);
			condition.signalAll();
		} finally {
			lock.unlock();
		}
	}
}

// 生产者
class Productor implements Runnable {

	private Clerk clerk;

	public Productor(Clerk clerk) {
		this.clerk = clerk;
	}
	@Override
	public void run() {
		for (int i = 0; i < 20; i++) {
			try {
				Thread.sleep(200);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			clerk.get();
		}
	}
}

// 消费者
class Consumer implements Runnable {
	private Clerk clerk;

	public Consumer(Clerk clerk) {
		this.clerk = clerk;
	}
	@Override
	public void run() {
		for (int i = 0; i < 20; i++) {
			clerk.sale();
		}
	}
}
```

###3. 线程按序交替

```java
/*
 * 编写一个程序： 开启 3 个线程，ID 分别为 A、B、C，
 * 每个线程将自己的 ID 在屏幕上打印 10 遍，要求输出的结果必须按顺序显示
 *	如：ABCABCABC…… 依次递归
 */
public class TestABCAlternate {
	public static void main(String[] args) {
		AlternateDemo ad = new AlternateDemo();
        
		new Thread(new Runnable() {
			@Override
			public void run() {
				for (int i = 1; i <= 20; i++) {
					ad.loopA(i);
				}
			}
		}, "A").start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				for (int i = 1; i <= 20; i++) {
					ad.loopB(i);
				}
			}
		}, "B").start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				for (int i = 1; i <= 20; i++) {
					ad.loopC(i);
					System.out.println("-----------------------------------");
				}
			}
		}, "C").start();
	}
}

class AlternateDemo{
	
	private int number = 1; //当前正在执行线程的标记
	
	private Lock lock = new ReentrantLock();
	private Condition condition1 = lock.newCondition();
	private Condition condition2 = lock.newCondition();
	private Condition condition3 = lock.newCondition();
	
	//param totalLoop : 循环第几轮
	public void loopA(int totalLoop){
		lock.lock();
		try {
			//1. 判断
			if(number != 1){
				condition1.await();
			}
			//2. 打印
			for (int i = 1; i <= 1; i++) {
				System.out.println(Thread.currentThread().getName() 
                                   + "\t" + i + "\t" + totalLoop);
			}
			//3. 唤醒
			number = 2;
			condition2.signal();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			lock.unlock();
		}
	}
	
	public void loopB(int totalLoop){
		lock.lock();
		
		try {
			//1. 判断
			if(number != 2){
				condition2.await();
			}
			//2. 打印
			for (int i = 1; i <= 1; i++) {
				System.out.println(Thread.currentThread().getName() 
                                   + "\t" + i + "\t" + totalLoop);
			}
			//3. 唤醒
			number = 3;
			condition3.signal();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			lock.unlock();
		}
	}
	
	public void loopC(int totalLoop){
		lock.lock();
		
		try {
			//1. 判断
			if(number != 3){
				condition3.await();
			}
			//2. 打印
			for (int i = 1; i <= 1; i++) {
				System.out.println(Thread.currentThread().getName() 
                                   + "\t" + i + "\t" + totalLoop);
			}
			//3. 唤醒
			number = 1;
			condition1.signal();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			lock.unlock();
		}
	}
}
```

### 4. ReadWriteLock 读写锁

- ReadWriteLock 维护一对读写锁
  - `writeLock()`： 用于写入操作，写入锁是独占的
  - `readLock()`： 用于只读操作，读取锁可以由多个 reader 线程同时保持

```java
/*
 * ReadWriteLock : 读写锁
 * 
 * 写写/读写 需要“互斥”
 * 读读 不需要互斥
 */
public class TestReadWriteLock {
	public static void main(String[] args) {
		ReadWriteLockDemo rw = new ReadWriteLockDemo();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				rw.set((int)(Math.random() * 101));
			}
		}, "Write:").start();
		
		for (int i = 0; i < 100; i++) {
			new Thread(new Runnable() {
				@Override
				public void run() {
					rw.get();
				}
			}).start();
		}
	}
}

class ReadWriteLockDemo{
	private int number = 0;
	private ReadWriteLock lock = new ReentrantReadWriteLock();
	
	//读
	public void get(){
		lock.readLock().lock(); //上锁
		try{
			System.out.println(Thread.currentThread().getName() + " : " + number);
		}finally{
			lock.readLock().unlock(); //释放锁
		}
	}
	
	//写
	public void set(int number){
		lock.writeLock().lock();
		try{
			System.out.println(Thread.currentThread().getName());
			this.number = number;
		}finally{
			lock.writeLock().unlock();
		}
	}
}
```

### 5. 线程八锁

>  线程八锁的关键：
>
>  - 非静态方法的锁默认为  this,  静态方法的锁为对应的 Class 实例
>  - 某一个时刻内，只能有一个线程持有锁

**题目：** 判断打印的 "one" or "two" ?

1. 两个普通同步方法，两个线程，标准打印

```java
//one  two
public class TestThread8Monitor {
	public static void main(String[] args) {
		Number number = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getTwo();
			}
		}).start();
	}
}

class Number{
	public synchronized void getOne(){
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
}
```

2. 新增 Thread.sleep() 给 getOne() 

```java
//one  two
public class TestThread8Monitor {
	public static void main(String[] args) {
		Number number = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getTwo();
			}
		}).start();
	}
}

class Number{
	public synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {}
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
}
```

3. 新增普通方法 getThree()

```java
//three  one   two
public class TestThread8Monitor {
	public static void main(String[] args) {
		Number number = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getTwo();
			}
		}).start();
      
      new Thread(new Runnable() {
			@Override
			public void run() {
				number.getThree();
			}
		}).start();
	}
}

class Number{
	public synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {}
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
    //没有 synchronized
	public void getThree(){
		System.out.println("three");
	}
}
```

4. 两个普通同步方法，两个 Number 对象

```java
//two  one
public class TestThread8Monitor {
	public static void main(String[] args) {
        //两个对象
		Number number = new Number();
		Number number2 = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number2.getTwo();
			}
		}).start();
	}
}

class Number{
	public synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {}
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
}
```

5. 修改 getOne() 为静态同步方法

```java
//two one ？？
//非静态方法的锁默认为 this，静态方法的锁对应 Class 实例
public class TestThread8Monitor {
	public static void main(String[] args) {
		Number number = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getTwo();
			}
		}).start();
	}
}

class Number{
    //改为静态方法： 作用于整个类
	public static synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {}
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
}
```

6. 修改两个方法均为静态同步方法，一个 Number 对象

```java
//one  two
public class TestThread8Monitor {
	public static void main(String[] args) {
		Number number = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getTwo();
			}
		}).start();
	}
}

class Number{
	public static synchronized void getOne(){//均为静态同步方法
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {
		}
		System.out.println("one");
	}
	public static synchronized void getTwo(){//均为静态同步方法
		System.out.println("two");
	}
}
```

7. 一个静态同步方法，一个非静态同步方法，两个 Number 对象

```java
//two  one
public class TestThread8Monitor {
	public static void main(String[] args) {
        //两个对象
		Number number = new Number();
		Number number2 = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();//先睡眠 3s
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number2.getTwo();
			}
		}).start();
	}
}

class Number{
    //静态同步方法
	public static synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {
		}
		System.out.println("one");
	}
	public synchronized void getTwo(){
		System.out.println("two");
	}
}
```

8. 两个静态同步方法，两个 Number 对象

```java
//one  two
//一个静态同步方法获取 Class 实例锁后，其他静态同步方法必须等待该方法释放锁才能获取锁
public class TestThread8Monitor {
	public static void main(String[] args) {
        //两个对象
		Number number = new Number();
		Number number2 = new Number();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number.getOne();
			} 
		}).start();
		
		new Thread(new Runnable() {
			@Override
			public void run() {
				number2.getTwo();
			}
		}).start();
	}
}

class Number{
    //均为静态同步方法
	public static synchronized void getOne(){
		try {
			Thread.sleep(3000);
		} catch (InterruptedException e) {
		}
		System.out.println("one");
	}
	//均为静态同步方法
	public static synchronized void getTwo(){
		System.out.println("two");
	}
}
```

# 八、JUC

## 1.JUC 简介 

`JUC(java.util.concurrent)`： 

- 增加了并发编程常用的实用工具类，用于定义类似线程的自定义子系统，包括线程池、异步 IO 和轻量级任务框架，提供可调的、灵活的线程池
- 提供了设计用于多线程上下文的 Collection 实现

## 2. volatile 关键字与内存可见性

**volatile 关键字** ：当多个线程进行操作共享数据时，可以保证内存中的数据可见

- **保证内存可见性，但不保证原子性**

**内存可见性：** 确保当一个线程修改了对象状态后，其他线程能够看到发生的状态变化，避免可见性错误

- **可见性错误**： 指当读操作与写操作在不同的线程中执行时，无法确保执行读操作的线程能适时地看到其他线程写入的值

- **解决方式**： 
  - 通过同步机制保证对象被安全发布
  - 使用 volatile 变量

## 3. 原子变量与CAS算法

####1. **CAS 算法：**  

- **定义**： CAS (Compare-And-Swap) 是一种硬件对并发的支持，针对多处理器而设计的一种特殊指令
- **作用**： 用于管理对共享数据的并发访问，是一种无锁的非阻塞算法的实现
- **组成**： 内存值 V，旧值 A，新值 B
- **ABA 问题**： 设置版本号来解决该问题

####2. **模拟 CAS 算法:** 

```java
public class TestCompareAndSwap {
	public static void main(String[] args) {
		final CompareAndSwap cas = new CompareAndSwap();
		for (int i = 0; i < 10; i++) {
			new Thread(new Runnable() {
				@Override
				public void run() {
					int expectedValue = cas.get();
					boolean b = cas.compareAndSet(expectedValue, 
                                                  (int)(Math.random() * 101));
					System.out.println(b);
				}
			}).start();
		}
	}
}

class CompareAndSwap{
	private int value;
	//获取内存值
	public synchronized int get(){
		return value;
	}
	
	//比较
	public synchronized int compareAndSwap(int expectedValue, int newValue){
		int oldValue = value;
		
		if(oldValue == expectedValue){
			this.value = newValue;
		}
		return oldValue;
	}
	
	//设置
	public synchronized boolean compareAndSet(int expectedValue, int newValue){
		return expectedValue == compareAndSwap(expectedValue, newValue);
	}
}
```

# 九、J.U.C - AQS

>  java.util.concurrent（J.U.C）大大提高了并发性能，**AQS 被认为是 J.U.C 的核心**

## 1. CountdownLatch(闭锁)

- CountDownLatch： 允许一个或多个线程一直等待，即用来控制一个线程等待多个线程
- 闭锁可以延迟线程的进度直到其到达终止状态，闭锁可以用来确保某些活动直到其他活动都完成才继续执行：
  - 确保某个计算在其需要的所有资源都被初始化之后才继续执行
  - 确保某个服务在其依赖的所有其他服务都已经启动之后才启动
  - 等待直到某个操作所有参与者都准备就绪再继续执行

维护了一个计数器 cnt，每次调用 countDown() 方法会让计数器的值减 1，减到 0 的时候，那些因为调用 await() 方法而在等待的线程就会被唤醒

<img src="../pics//CountdownLatch.png" width=""/>

```java
public class CountdownLatchExample {
    public static void main(String[] args) throws InterruptedException {
        final int totalThread = 10;
        CountDownLatch countDownLatch = new CountDownLatch(totalThread);
        ExecutorService executorService = Executors.newCachedThreadPool();
        for (int i = 0; i < totalThread; i++) {
            executorService.execute(() -> {
                System.out.print("run..");
                countDownLatch.countDown();
            });
        }
        countDownLatch.await();
        System.out.println("end");
        executorService.shutdown();
    }
}
```

```html
run..run..run..run..run..run..run..run..run..run..end
```

## 2. CyclicBarrier 

- `CyclicBarrier(循环屏障)`： 用来控制多个线程互相等待，当多个线程都到达时，这些线程才会继续执行

- **与 CountdownLatch 的区别**： CyclicBarrier 的计数器通过调用 reset() 方法可以循环使用

```java
/**
 * 构造函数： 
 * parties： 指示计数器的初始值
 * barrierAction： 在所有线程都到达屏障时会执行一次
 */
public CyclicBarrier(int parties, Runnable barrierAction) {
    if (parties <= 0) throw new IllegalArgumentException();
    this.parties = parties;
    this.count = parties;
    this.barrierCommand = barrierAction;
}

public CyclicBarrier(int parties) {
    this(parties, null);
}
```

<img src="../pics//CyclicBarrier.png" width=""/>

```java
public class CyclicBarrierExample {
    public static void main(String[] args) {
        final int totalThread = 10;
        CyclicBarrier cyclicBarrier = new CyclicBarrier(totalThread);
        ExecutorService executorService = Executors.newCachedThreadPool();
        for (int i = 0; i < totalThread; i++) {
            executorService.execute(() -> {
                System.out.print("before..");
                try {
                    cyclicBarrier.await();
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                }
                System.out.print("after..");
            });
        }
        executorService.shutdown();
    }
}
```

```html
before..before..before..before..before..before..before..before..before..before..after..after..after..after..after..after..after..after..after..after..
```

## 3. Semaphore

Semaphore 类似于操作系统中的信号量，可以控制对互斥资源的访问线程数

<img src="../pics//Semaphore.png"/>

```java
//模拟了对某个服务的并发请求，每次只能有 3 个客户端同时访问，请求总数为 10
public class SemaphoreExample {
    public static void main(String[] args) {
        final int clientCount = 3;
        final int totalRequestCount = 10;
        Semaphore semaphore = new Semaphore(clientCount);
        ExecutorService executorService = Executors.newCachedThreadPool();
        for (int i = 0; i < totalRequestCount; i++) {
            executorService.execute(()->{
                try {
                    semaphore.acquire();
                    System.out.print(semaphore.availablePermits() + " ");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    semaphore.release();
                }
            });
        }
        executorService.shutdown();
    }
}
```

```html
2 1 2 2 2 2 2 1 2 2
```

# 十、J.U.C - 其它组件

## 1. FutureTask

```java
public class FutureTask<V> implements RunnableFuture<V>

public interface RunnableFuture<V> extends Runnable, Future<V>
```

- FutureTask 可用于异步获取执行结果或取消执行任务的场景
- 当一个任务需要执行很长时间，则可以用 FutureTask 封装该任务，主线程在完成自己的任务后再去获取结果

```java
public class FutureTaskExample {
    public static void main(String[] args) 
        throws ExecutionException, InterruptedException {
        
        FutureTask<Integer> futureTask 
            = new FutureTask<Integer>(new Callable<Integer>() {
            @Override
            public Integer call() throws Exception {
                int result = 0;
                for (int i = 0; i < 100; i++) {
                    Thread.sleep(10);
                    result += i;
                }
                return result;
            }
        });

        Thread computeThread = new Thread(futureTask);
        computeThread.start();

        Thread otherThread = new Thread(() -> {
            System.out.println("other task is running...");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        otherThread.start();
        System.out.println(futureTask.get());
    }
}

//结果
other task is running...
4950
```

## 2. BlockingQueue

java.util.concurrent.BlockingQueue 接口有以下阻塞队列的实现：

-  **FIFO 队列** ：LinkedBlockingQueue、ArrayBlockingQueue（固定长度）
-  **优先级队列** ：PriorityBlockingQueue

提供了阻塞的 take() 和 put() 方法：

- 如果队列为空 take() 将阻塞，直到队列中有内容
- 如果队列为满 put() 将阻塞，直到队列有空闲位置

**使用 BlockingQueue 实现生产者消费者问题** 

```java
public class ProducerConsumer {

    private static BlockingQueue<String> queue = new ArrayBlockingQueue<>(5);

    private static class Producer extends Thread {
        @Override
        public void run() {
            try {
                queue.put("product");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.print("produce..");
        }
    }

    private static class Consumer extends Thread {
        @Override
        public void run() {
            try {
                String product = queue.take();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.print("consume..");
        }
    }
    
    public static void main(String[] args) {
        for (int i = 0; i < 2; i++) {
            Producer producer = new Producer();
            producer.start();
        }
        for (int i = 0; i < 5; i++) {
            Consumer consumer = new Consumer();
            consumer.start();
        }
        for (int i = 0; i < 3; i++) {
            Producer producer = new Producer();
            producer.start();
        }
    }
}

//结果
produce..produce..consume..consume..produce..consume..produce..consume..produce..consume..
```

## 3. ForkJoin 分支/合并框架(工作窃取)

- 主要用于并行计算，类似 MapReduce，即把**大的计算任务拆分成多个小任务并行计算**

  ```java
  public class ForkJoinExample extends RecursiveTask<Integer> {
      private final int threshold = 5;
      private int first;
      private int last;
  
      public ForkJoinExample(int first, int last) {
          this.first = first;
          this.last = last;
      }
  
      @Override
      protected Integer compute() {
          int result = 0;
          if (last - first <= threshold) {
              // 任务足够小则直接计算
              for (int i = first; i <= last; i++) {
                  result += i;
              }
          } else {
              // 拆分成小任务
              int middle = first + (last - first) / 2;
              ForkJoinExample leftTask = new ForkJoinExample(first, middle);
              ForkJoinExample rightTask = new ForkJoinExample(middle + 1, last);
              leftTask.fork();
              rightTask.fork();
              result = leftTask.join() + rightTask.join();
          }
          return result;
      }
      
      public static void main(String[] args) 
          	throws ExecutionException, InterruptedException {
          ForkJoinExample example = new ForkJoinExample(1, 10000);
          //ForkJoinPool 是一个特殊的线程池，线程数量取决于 CPU 核数
          ForkJoinPool forkJoinPool = new ForkJoinPool();
          //使用 ForkJoinPool 来启动
          Future result = forkJoinPool.submit(example);
          System.out.println(result.get());
      }
  }
  ```

- `ForkJoinPool`： 实现了工作窃取算法来提高 CPU 的利用率

  - 每个线程维护一个双端队列，用来存储需要执行的任务

  - 工作窃取算法允许空闲的线程从其它线程的双端队列中窃取一个任务来执行

  - 窃取的任务必须是最晚的任务，避免和队列所属线程发生竞争

    > 但若队列中只有一个任务时还是会发生竞争

  ```java
  public class ForkJoinPool extends AbstractExecutorService
  ```

  例如下图： Thread2 从 Thread1 的队列中拿出最晚的 Task1 任务，Thread1 会拿出 Task2 来执行

  <img src="../pics//15b45dc6-27aa-4519-9194-f4acfa2b077f.jpg" width=""/>

- **Fork/Join 框架与线程池的区别:** 

  - **采用工作窃取模式**：执行新任务时，可以将其拆分成更小的任务执行，并将小任务加到线程队列中，然后再随机从一个线程的队列中偷一个任务并把它放在自己的队列中

  - 对于线程池，若一个线程正在执行的任务由于某些原因无法继续运行，则该线程处于等待状态

    对于 fork/join 框架，若某个子问题由于等待另外一个子问题的完成而无法继续运行，则处理该子问题的线程会主动寻找其他尚未运行的子问题来执行

    > 减少了线程的等待时间，提高了性能

  ```java
  public class TestForkJoinPool {
  	public static void main(String[] args) {
  		Instant start = Instant.now();
  		ForkJoinPool pool = new ForkJoinPool();
  		ForkJoinTask<Long> task = new ForkJoinSumCalculate(0L, 50000000000L);
  		Long sum = pool.invoke(task);
  		System.out.println(sum);
  		Instant end = Instant.now();
  		System.out.println("耗费时间为：" + 
                           Duration.between(start, end).toMillis());//166-1996-10590
  	}
  	
  	@Test
  	public void test1(){
  		Instant start = Instant.now();
  		long sum = 0L;
  		
  		for (long i = 0L; i <= 50000000000L; i++) {
  			sum += i;
  		}
  		System.out.println(sum);
  		Instant end = Instant.now();
  		System.out.println("耗费时间为：" + 
                            Duration.between(start, end).toMillis());//35-3142-15704
  	}
  	
  	//java8 新特性
  	@Test
  	public void test2(){
  		Instant start = Instant.now();
  		Long sum = LongStream.rangeClosed(0L, 50000000000L)
              				 .parallel().reduce(0L, Long::sum);
  		System.out.println(sum);
  		Instant end = Instant.now();
  		System.out.println("耗费时间为：" + 
                             Duration.between(start, end).toMillis());//1536-8118
  	}
  }
  
  class ForkJoinSumCalculate extends RecursiveTask<Long>{
  	private static final long serialVersionUID = -259195479995561737L;
  	
  	private long start;
  	private long end;
  	private static final long THURSHOLD = 10000L;  //临界值
  	
  	public ForkJoinSumCalculate(long start, long end) {
  		this.start = start;
  		this.end = end;
  	}
  
  	@Override
  	protected Long compute() {
  		long length = end - start;
  		if(length <= THURSHOLD){
  			long sum = 0L;
  			for (long i = start; i <= end; i++) {
  				sum += i;
  			}
  			return sum;
  		}else{
  			long middle = (start + end) / 2;
  			ForkJoinSumCalculate left = new ForkJoinSumCalculate(start, middle); 
  			left.fork(); //进行拆分，同时压入线程队列(自动递归拆分)
  			ForkJoinSumCalculate right = new ForkJoinSumCalculate(middle+1, end);
  			right.fork();
  			return left.join() + right.join();
  		}
  	}
  }
  ```

# 十一、线程不安全示例

```java
public class ThreadUnsafeExample {

    private int cnt = 0;

    public void add() {
        cnt++;
    }

    public int get() {
        return cnt;
    }
    
    public static void main(String[] args) throws InterruptedException {
        final int threadSize = 1000;
        ThreadUnsafeExample example = new ThreadUnsafeExample();
        final CountDownLatch countDownLatch = new CountDownLatch(threadSize);
        ExecutorService executorService = Executors.newCachedThreadPool();
        for (int i = 0; i < threadSize; i++) {
            executorService.execute(() -> {
                example.add();
                countDownLatch.countDown();
            });
        }
        countDownLatch.await();
        executorService.shutdown();
        System.out.println(example.get());
    }
}

//结果
997
```

# 十二、Java 内存模型

## 1. 主内存与工作内存

- **高速缓存作用**： 解决寄存器与内存读写速度的差异

- **高速缓存的问题**： 缓存一致性

  <img src="../pics//68778c1b-15ab-4826-99c0-3b4fd38cb9e9.png"/>

- 所有变量都存储在主内存中，每个线程的工作内存存储在高速缓存或者寄存器中

- 线程只能直接操作工作内存中的变量，不同线程间的变量值传递需要通过主内存完成

  <img src="../pics//47358f87-bc4c-496f-9a90-8d696de94cee.png"/>

## 2. 内存间交互操作

<img src="../pics//536c6dfd-305a-4b95-b12c-28ca5e8aa043.png"/>

Java 内存模型定义的完成主内存和工作内存的交互操作： 

- `read`：把一个变量的值从主内存传输到工作内存中
- `load`：在 read 之后执行，把 read 得到的值放入工作内存的变量副本中
- `use`：把工作内存中一个变量的值传递给执行引擎
- `assign`：把一个从执行引擎接收到的值赋给工作内存的变量
- `store`：把工作内存的一个变量的值传送到主内存中
- `write`：在 store 之后执行，把 store 得到的值放入主内存的变量中
- `lock`：作用于主内存的变量
- `unlock`

## 3. 内存模型三大特性

### 1. 原子性

- 下图演示两个线程同时对 int 进行操作: 

  <img src="../pics//ef8eab00-1d5e-4d99-a7c2-d6d68ea7fe92.png"/>

- AtomicInteger 保证多个线程修改的原子性： 

  <img src="../pics//952afa9a-458b-44ce-bba9-463e60162945.png"/>

- **使用 AtomicInteger 实现线程安全**

  ```java
  public class AtomicExample {
      private AtomicInteger cnt = new AtomicInteger();
  
      public void add() {
          cnt.incrementAndGet();
      }
  
      public int get() {
          return cnt.get();
      }
      
      public static void main(String[] args) throws InterruptedException {
          final int threadSize = 1000;
          AtomicExample example = new AtomicExample(); // 只修改这条语句
          final CountDownLatch countDownLatch = new CountDownLatch(threadSize);
          ExecutorService executorService = Executors.newCachedThreadPool();
          for (int i = 0; i < threadSize; i++) {
              executorService.execute(() -> {
                  example.add();
                  countDownLatch.countDown();
              });
          }
          countDownLatch.await();
          executorService.shutdown();
          System.out.println(example.get());
      }
  }
  
  //结果
  1000
  ```

- **使用 synchronized 互斥锁保证原子性**

  - 对应的内存间交互操作为：lock 和 unlock
  - 在虚拟机实现上对应的字节码指令为 monitorenter 和 monitorexit

  ```java
  public class AtomicSynchronizedExample {
      private int cnt = 0;
  
      public synchronized void add() {
          cnt++;
      }
  
      public synchronized int get() {
          return cnt;
      }
      
      public static void main(String[] args) throws InterruptedException {
          final int threadSize = 1000;
          AtomicSynchronizedExample example = new AtomicSynchronizedExample();
          final CountDownLatch countDownLatch = new CountDownLatch(threadSize);
          ExecutorService executorService = Executors.newCachedThreadPool();
          for (int i = 0; i < threadSize; i++) {
              executorService.execute(() -> {
                  example.add();
                  countDownLatch.countDown();
              });
          }
          countDownLatch.await();
          executorService.shutdown();
          System.out.println(example.get());
      }
  }
  
  //结果
  1000
  ```

### 2. 可见性

- **可见性**： 指当一个线程修改了共享变量的值，其它线程能够立即得知这个修改
- **原理**：通过在变量修改后将新值同步回主内存，在变量读取前从主内存刷新变量值来实现
- **实现方式**： 
  - `volatile` 关键字
  - `synchronized`： 对变量执行 unlock 操作前，必须把变量值同步回主内存
  - `final`： final 修饰的变量在初始化后且没有发生 this 逃逸，则其它线程就能看见 final 字段的值

### 3. 有序性

- **有序性**：在本线程内观察，所有操作都是有序的；但在其他线程内观察，所有操作都是无序的

  > - 无序是因为发生了指令重排序
  > - 重排序不影响单线程程序的执行，却会影响多线程并发执行的正确性

- **保证有序性**： 

  - `volatile`： 通过添加内存屏障的方式来禁止指令重排

  - `synchronized`： 保证每个时刻只有一个线程执行同步代码，相当于是让线程顺序执行同步代码

## 先行发生原则

### 1. 单一线程原则

- 定义： **在一个线程内，程序前面的操作先行发生于后面的操作** 

  <img src="../pics//single-thread-rule.png"/>

### 2. 管程锁定规则

- **定义**： **一个 unlock 操作先行发生于后面对同一个锁的 lock 操作**

  <img src="../pics//monitor-lock-rule.png"/>

### 3. volatile 变量规则

- 定义： **对一个 volatile 变量的写操作先行发生于后面对这个变量的读操作**

  <img src="../pics//volatile-variable-rule.png"/>

### 4. 线程启动规则

- 定义： **Thread 对象的 start() 方法调用先行发生于此线程的每一个动作**

  <img src="../pics//thread-start-rule.png"/>

### 5. 线程加入规则

- 定义： **Thread 对象的结束先行发生于 join() 方法返回**

  <img src="../pics//thread-join-rule.png"/>

### 6. 线程中断规则

- 定义： **对线程 interrupt() 方法的调用先行发生于被中断线程的代码检测到中断事件的发生**

### 7. 对象终结规则

- 定义： **一个对象的初始化完成先行发生于它的 finalize() 方法的开始**

### 8. 传递性

- 定义： **如果操作 A 先行发生于操作 B，操作 B 先行发生于操作 C，那么操作 A 先行发生于操作 C**

# 十三、线程安全

## 1. 线程安全定义

- **一个类或方法可以被多个线程安全调用**

## 2. 线程安全分类

### 1. 不可变

- 不可变：一定是线程安全的

- 不可变的类型：
  - `final` 关键字修饰的基本数据类型
  - `String`
  - **枚举类型**
  - **Number 部分子类**
    - Long 和 Double 等数值包装类型，BigInteger 和 BigDecimal 等大数据类型
    - 但同为 Number 的原子类 AtomicInteger 和 AtomicLong 则是可变的

  对于集合类型，可以使用 `Collections.unmodifiableXXX()` 方法来获取不可变集合

  ```java
  public class ImmutableExample {
      public static void main(String[] args) {
          Map<String, Integer> map = new HashMap<>();
          //先对原始的集合进行拷贝
          Map<String, Integer> unmodifiableMap = Collections.unmodifiableMap(map);
          //需要对集合进行修改的方法都直接抛出异常
          unmodifiableMap.put("a", 1);
      }
  }
  ```

### 2. 绝对线程安全

- 不管运行时环境如何，调用者都不需要任何额外的同步措施

### 3. 相对线程安全

- 相对线程安全需要保证对这个对象单独的操作是线程安全的，在调用时不需要做额外的保障措施

- 但对于一些特定顺序的连续调用，就需要在调用端使用额外的同步手段来保证调用的正确性

  > java 相对线程安全类： Vector、HashTable、Collections 的 synchronizedCollection() 方法包装的集合

  ```java
  public class VectorUnsafeExample {
      private static Vector<Integer> vector = new Vector<>();
  
      public static void main(String[] args) {
          while (true) {
              for (int i = 0; i < 100; i++) {
                  vector.add(i);
              }
              ExecutorService executorService = Executors.newCachedThreadPool();
              executorService.execute(() -> {
                  for (int i = 0; i < vector.size(); i++) {
                      vector.remove(i);
                  }
              });
              executorService.execute(() -> {
                  for (int i = 0; i < vector.size(); i++) {
                      vector.get(i);//访问已被删除的元素会报错
                  }
              });
              executorService.shutdown();
          }
      }
  }
  
  //保证上面的代码能正确执行，就需要对删除元素和获取元素的代码进行同步
  executorService.execute(() -> {
      synchronized (vector) {
          for (int i = 0; i < vector.size(); i++) {
              vector.remove(i);
          }
      }
  });
  executorService.execute(() -> {
      synchronized (vector) {
          for (int i = 0; i < vector.size(); i++) {
              vector.get(i);
          }
      }
  });
  ```

### 4. 线程兼容

- 线程兼容： 指对象本身并不是线程安全的，但可以**通过在调用端正确地使用同步手段来保证对象在并发环境中可以安全地使用**

### 5. 线程对立

- 线程对立： 指无论调用端是否采取了同步措施，都**无法在多线程环境中并发使用的代码**

## 3. 线程安全的实现方法

### 1. 互斥同步

- `synchronized 和 ReentrantLock`

### 2. 非阻塞同步

- **互斥同步(阻塞同步)**： 具有线程阻塞和唤醒所带来的性能问题

- 非阻塞同步的方式： 

  - `CAS`： 先进行操作，若没有线程争用共享数据，则操作就成功，否则采取补偿措施(不断地重试，直到成功为止)

  - `AtomicInteger`： 原子类 AtomicInteger 的方法调用了 Unsafe 类的 CAS 操作

    ```java
    public final int incrementAndGet() {
        return unsafe.getAndAddInt(this, valueOffset, 1) + 1;
    }
    //getAndAddInt 源码
    //var1 指示对象内存地址，var2 指示该字段相对对象内存地址的偏移，var4 指示操作需要加的数值
    public final int getAndAddInt(Object var1, long var2, int var4) {
        int var5;
        do {
            var5 = this.getIntVolatile(var1, var2);
        } while(!this.compareAndSwapInt(var1, var2, var5, var5 + var4));
    
        return var5;
    }
    ```

  - `ABA` 问题： 添加版本号来解决

### 3. 无同步方案

- 定义： **如果方法不涉及共享数据，无须任何同步措施去保证正确性**

- 实现的方式： 

  - **栈封闭**： 多个线程访问同一个方法的局部变量时，不会出现线程安全问题，因为局部变量存储在虚拟机栈中，属于线程私有

    ```java
    public class StackClosedExample {
        public void add100() {
            int cnt = 0;
            for (int i = 0; i < 100; i++) {
                cnt++;
            }
            System.out.println(cnt);
        }
        
        public static void main(String[] args) {
            StackClosedExample example = new StackClosedExample();
            ExecutorService executorService = Executors.newCachedThreadPool();
            executorService.execute(() -> example.add100());
            executorService.execute(() -> example.add100());
            executorService.shutdown();
        }
    }
    
    //结果
    100
    100
    ```

  - **线程本地存储**： 若共享数据的代码能保证在同一个线程中执行，则可以把共享数据的可见范围限制在同一个线程之内，无须同步也能保证线程之间不出现数据争用

    - **典型应用**： 经典 Web 交互模型中的“一个请求对应一个服务器线程”的处理方式，该处理方式使得很多 Web 服务端应用都可以使用线程本地存储来解决线程安全问题

    - 使用 `java.lang.ThreadLocal` 类来实现线程本地存储功能

      ```java
      public class ThreadLocalExample {
          public static void main(String[] args) {
              ThreadLocal threadLocal = new ThreadLocal();
              Thread thread1 = new Thread(() -> {
                  threadLocal.set(1);
                  try {
                      Thread.sleep(1000);
                  } catch (InterruptedException e) {
                      e.printStackTrace();
                  }
                  System.out.println(threadLocal.get());
                  threadLocal.remove();
              });
              Thread thread2 = new Thread(() -> {
                  threadLocal.set(2);
                  threadLocal.remove();
              });
              thread1.start();
              thread2.start();
          }
      }
      
      //结果
      1
      ```

      `ThreadLocal` 的 `get() 与 set()` 方法： 

      ```java
      public void set(T value) {
          Thread t = Thread.currentThread();
          ThreadLocalMap map = getMap(t);
          if (map != null)
              map.set(this, value);
          else
              createMap(t, value);
      }
      
      public T get() {
          Thread t = Thread.currentThread();
          ThreadLocalMap map = getMap(t);
          if (map != null) {
              ThreadLocalMap.Entry e = map.getEntry(this);
              if (e != null) {
                  @SuppressWarnings("unchecked")
                  T result = (T)e.value;
                  return result;
              }
          }
          return setInitialValue();
      }
      ```

  - **可重入代码**： 可以在代码执行的任何时刻中断它，转而去执行另外一段代码(包括递归调用它本身)，而在控制权返回后，原来的程序不会出现任何错误

    > 可重入代码的共同特征：不依赖存储在堆上的数据和公用的系统资源、用到的状态量都由参数中传入、不调用非可重入的方法等

# 十四、锁优化

## 1. 自旋锁

- **自旋锁**： 让一个线程在请求一个共享数据的锁时忙循环(自旋)一段时间，若在这段时间内获得锁，就可以避免进入阻塞状态
- **适用场景**： 由于进行忙循环操作占用 CPU 时间，只适用于共享数据的锁定状态很短的场景

## 2. 锁消除

- **锁消除**： 指对于被检测出不可能存在竞争的共享数据的锁进行消除
- **实现**： 通过逃逸分析来支持，若堆上的共享数据不可能逃逸，则可以当成私有数据

## 3. 锁粗化

- **锁粗化**： 若虚拟机探测到一串零碎的操作都对同一个对象加锁，将会把加锁的范围扩展(粗化)到整个操作序列的外部

## 4. 轻量级锁

- 偏向锁和轻量级锁可以让锁拥有四个状态：**无锁状态、偏向锁状态、轻量级锁状态、重量级锁状态**

- **轻量级锁**： 相对于传统的重量级锁而言，使用 CAS 操作来避免重量级锁使用互斥量的开销
  - 先采用 CAS 操作进行同步
  - 如果 CAS 失败再改用互斥量进行同步，即膨胀为重量级锁

## 5. 偏向锁

- **偏向锁**： 先让获取偏向锁的线程不进行同步操作，当有其他线程去尝试获取这个锁对象时，偏向锁状态恢复到未锁定状态或者轻量级锁状态

# 十五、多线程开发良好的实践

- 给线程起个有意义的名字，这样可以方便找 Bug

- 缩小同步范围，从而减少锁争用

- 多用同步工具少用 wait() 和 notify()，如： CountDownLatch, CyclicBarrier, Semaphore 和 Exchanger

- 使用 BlockingQueue 实现生产者消费者问题

- 多用并发集合少用同步集合，例如应该使用 ConcurrentHashMap 而不是 Hashtable

- 使用本地变量和不可变类来保证线程安全

- 使用线程池而不是直接创建线程

# 参考资料

- BruceEckel. Java 编程思想: 第 4 版 [M]. 机械工业出版社, 2007.
- 周志明. 深入理解 Java 虚拟机 [M]. 机械工业出版社, 2011.
- [Threads and Locks](https://docs.oracle.com/javase/specs/jvms/se6/html/Threads.doc.html)
- [线程通信](http://ifeve.com/thread-signaling/#missed_signal)
- [Java 线程面试题 Top 50](http://www.importnew.com/12773.html)
- [BlockingQueue](http://tutorials.jenkov.com/java-util-concurrent/blockingqueue.html)
- [thread state java](https://stackoverflow.com/questions/11265289/thread-state-java)
- [CSC 456 Spring 2012/ch7 MN](http://wiki.expertiza.ncsu.edu/index.php/CSC_456_Spring_2012/ch7_MN)
- [Java - Understanding Happens-before relationship](https://www.logicbig.com/tutorials/core-java-tutorial/java-multi-threading/happens-before.html)
- [6장 Thread Synchronization](https://www.slideshare.net/novathinker/6-thread-synchronization)
- [How is Java's ThreadLocal implemented under the hood?](https://stackoverflow.com/questions/1202444/how-is-javas-threadlocal-implemented-under-the-hood/15653015)
- [Concurrent](https://sites.google.com/site/webdevelopart/21-compile/06-java/javase/concurrent?tmpl=%2Fsystem%2Fapp%2Ftemplates%2Fprint%2F&showPrintDialog=1)
- [JAVA FORK JOIN EXAMPLE](http://www.javacreed.com/java-fork-join-example/ "Java Fork Join Example")
- [聊聊并发（八）——Fork/Join 框架介绍](http://ifeve.com/talk-concurrency-forkjoin/)
- [Eliminating SynchronizationRelated Atomic Operations with Biased Locking and Bulk Rebiasing](http://www.oracle.com/technetwork/java/javase/tech/biasedlocking-oopsla2006-preso-150106.pdf)
