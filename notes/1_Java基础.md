# 一、数据类型

## 1. 包装类型 

八个基本类型：

- boolean/1
- byte/8
- char/16
- short/16
- int/32
- float/32
- long/64
- double/64

基本类型都有对应的包装类型，基本类型与其对应的包装类型之间的赋值使用自动装箱与拆箱完成。

```java
Integer i = 1;
int j = i;
/**
*    上面代码编译成class后
*/
Integer i = Integer.valueOf(1); // 装箱
int j = i.intValue(); // 拆箱
```

## 2. 缓存池

new Integer(123) 与 Integer.valueOf(123) 的区别在于：

- new Integer(123) 每次都会新建一个对象
- Integer.valueOf(123) 会使用缓存池中的对象，多次调用会取得同一个对象的引用

```java
Integer x = new Integer(123);
Integer y = new Integer(123);
System.out.println(x == y);   // false <==>  System.out.println(x.equals(y)); //true
Integer z = Integer.valueOf(123);
Integer k = Integer.valueOf(123);
System.out.println(z == k);   // true
```

注：equals 与 == 的区别：

- **等号**  比较两个纯字符串时，比较基本类型，如果值相同，则返回 true；比较引用时，如果引用指向同一内存中的同一对象，则返回 true
- **equals**  只比较两个对象的内容是否相等，相等则返回true

`valueOf()`： 先判断值是否在缓存池中，如果在，则直接返回缓存池的内容

```java
//IntegerCache 是 Integer 的内部类
//下面是 valueOf 的源码
public static Integer valueOf(int i) {
    if (i >= IntegerCache.low && i <= IntegerCache.high)
        return IntegerCache.cache[i + (-IntegerCache.low)];
    return new Integer(i);
}
```

在 Java 8 中，**Integer 缓存池的大小默认为 -128\~127** 

```java
//-128-127原因：
//1. 这个范围的整数值是使用最广泛的
//2. 而且在程序中第一次使用 Integer 的时候也需要一定的额外时间来初始化这个缓存
static final int low = -128;
static final int high;//high可以配置，默认为 127
static final Integer cache[];

//静态代码块，Integer类加载时就缓存
static {
    // high value may be configured by property
    int h = 127;
    //读取 JVM 参数配置，灵活提高性能，vm 参数：-XX:AutoBoxCacheMax=<size>
    String integerCacheHighPropValue =
        sun.misc.VM.getSavedProperty("java.lang.Integer.IntegerCache.high");
    if (integerCacheHighPropValue != null) {
        try {
            int i = parseInt(integerCacheHighPropValue);
            i = Math.max(i, 127);
            // Maximum array size is Integer.MAX_VALUE
            h = Math.min(i, Integer.MAX_VALUE - (-low) -1);//防止越界
        } catch( NumberFormatException nfe) {
            // If the property cannot be parsed into an int, ignore it.
        }
    }
    high = h;

    cache = new Integer[(high - low) + 1];//创建缓存数组
    int j = low;
    for(int k = 0; k < cache.length; k++)
        cache[k] = new Integer(j++);

    // range [-128, 127] must be interned (JLS7 5.1.7)
    assert IntegerCache.high >= 127;//保证[-128, 127]在缓存范围内
}
```

**注：** 断言讲解：[java 断言 assert](https://blog.csdn.net/yin__ren/article/details/82759338)

编译器会在自动装箱过程调用 valueOf() 方法，因此多个 Integer 实例使用自动装箱来创建并且值相同，那么就会引用相同的对象。

```java
Integer m = 123;
Integer n = 123;
System.out.println(m == n); // true
```

**注：** [JAVA 基本类型的封装类及对应常量池](https://blog.csdn.net/TaoTaoFu/article/details/74943337) 

1. java中基本类型的包装类的大部分都实现了常量池技术，这些类是Byte,Short,Integer,Long,Character,Boolean,另外两种浮点数类型的包装类则没有实现
2. Byte,Short,Integer,Long,Character这5种整型的包装类也只是在对应值小于等于127时才可使用对象池，也即对象不负责创建和管理大于127的这些类的对象。

基本类型对应的缓冲池如下：

- boolean values true and false
- all byte values
- short values between -128 and 127
- int values between -128 and 127
- char in the range \u0000 to \u007F

在使用这些基本类型对应的包装类型时，就可以直接使用缓冲池中的对象

# 二、String

## 1. 概览

- **String 被声明为 final**，因此它不可被继承

- **使用 `final char` 数组存储数据** ，且 String 内部没有改变 value 数组的方法，因此可以保证 String 不可变

```java
public final class String
    implements java.io.Serializable, Comparable<String>, CharSequence {
    /** The value is used for character storage. */
    private final char value[];
```

## 2. 不可变的好处

**1. 可以缓存 hash 值** 

- 因为 String 的 hash 值经常被使用，**不可变的特性可以使得 hash 值也不可变**，因此只需要进行一次计算

**2. String Pool 的需要** 

- 若一个 String 对象已经被创建，则会从 String Pool 中取得引用，只有 String 不可变，才能使用 String Pool

<img src="../pics//f76067a5-7d5f-4135-9549-8199c77d8f1c.jpg" width=""/>

**3. 安全性** 

- String 经常作为参数，**String 不可变性可以保证参数不可变**

**4. 线程安全** 

- String 不可变性天生具备线程安全，可以在多个线程中安全地使用

[Program Creek : Why String is immutable in Java?](https://www.programcreek.com/2013/04/why-string-is-immutable-in-java/)

## 3. String, StringBuffer, StringBuilder

**1. 可变性** 

- String 不可变
- StringBuffer 和 StringBuilder 可变

**2. 线程安全** 

- String 不可变，因此是线程安全的
- StringBuilder 不是线程安全的
- StringBuffer 是线程安全的，内部使用 synchronized 进行同步

[StackOverflow : String, StringBuffer, and StringBuilder](https://stackoverflow.com/questions/2971315/string-stringbuffer-and-stringbuilder)

## 4. String Pool

- **字符串常量池(String Poll)**： 保存着所有字符串字面量，这些字面量在编译时期确定
- 可以使用 String 的 intern() 方法在运行过程中将字符串添加到 String Poll 中
- **intern() 方法**： 
  - 将 String 常量池从 Perm 区移动到了 Java Heap 区
  - `String#intern` 方法时，**如果存在堆中的对象，会直接保存对象的引用**，而不会重新创建对象

```java
//示例一
String s = new String("1"); //创建的 String 对象放入Heap中
s.intern(); //因为是对象，所以只返回对象的引用(Heap 中)
String s2 = "1"; //放入常量池中
System.out.println(s == s2);//false

String s3 = new String("1") + new String("1"); //Heap 中的 String 对象相加而得到的字符串
s3.intern(); //将 s3 字符串放入常量池中
String s4 = "11"; //从常量池返回结果
System.out.println(s3 == s4);//true

//示例二
String s1 = new String("aaa");
String s2 = new String("aaa");
System.out.println(s1 == s2);           // false
String s3 = s1.intern();
String s4 = s1.intern();
System.out.println(s3 == s4);           // true
String s5 = "a";
String s6 = "aa";
String s7 = s5 + s6;
System.out.println(s1 == s7);			//false
String s5 = "bbb";
String s6 = "bbb";
System.out.println(s6 == s5);  // true

//示例三
String str1 = "a";
String str2 = "b";
String str3 = "ab";
String str4 = str1 + str2;
String str5 = new String("ab");

System.out.println(str5.equals(str3)); //true
System.out.println(str5 == str3); //false 
//str5调用intern时，会检查字符串池中是否含有该字符串，由于已有该字符串，所以会得到相同的引用
System.out.println(str5.intern() == str3); //true
System.out.println(str5.intern() == str4); //false

//示例四
String a = new String("ab");
String b = new String("ab");
String c = "ab";
String d = "a" + "b";
String e = "b";
String f = "a" + e;

//创建的 String 对象放入Heap中，所以常量池中不含
System.out.println(b.intern() == a); //false
System.out.println(b.intern() == c); //true
//参与运算的字符串都为静态字符串时，会添加到常量池中
System.out.println(b.intern() == d); //true
//变量参与运算，不会放入常量池中
System.out.println(b.intern() == f); //false
//调用 intern 方法，将创建的 String 对象放入常量池中
System.out.println(b.intern() == a.intern()); //true
```

推荐阅读： 

- [String中intern的方法](https://www.cnblogs.com/wanlipeng/archive/2010/10/21/1857513.html) 
- [深入解析 String#intern](https://tech.meituan.com/in_depth_understanding_string_intern.html) 

# 三、运算

## 1. 参数传递

- **Java 的参数是以值传递的形式传入方法中**，而不是引用传递

- 形参是基本数据类型：将实参的值传递给形参的基本数据类型的变量
- 形参是引用数据类型：将实参的引用（对应的堆空间的对象实体的首地址值）传递给形参的引用类型变量

```java
public class Dog {
    String name;

    Dog(String name) {
        this.name = name;
    }

    String getName() {
        return this.name;
    }

    void setName(String name) {
        this.name = name;
    }

    String getObjectAddress() {
        return super.toString();
    }
}
```

```java
public class PassByValueExample {
    public static void main(String[] args) {
        Dog dog = new Dog("A");
        System.out.println(dog.getObjectAddress()); // Dog@4554617c
        func(dog);
        System.out.println(dog.getObjectAddress()); // Dog@4554617c
        System.out.println(dog.getName());          // A
    }

    private static void func(Dog dog) { //该 dog 是一个指针类型
        System.out.println(dog.getObjectAddress()); // Dog@4554617c
        dog = new Dog("B");
        System.out.println(dog.getObjectAddress()); // Dog@74a14482
        System.out.println(dog.getName());          // B
    }
}
```

但是如果在方法中改变对象的字段值会改变原对象该字段值，因为改变的是同一个地址指向的内容

```java
class PassByValueExample {
    public static void main(String[] args) {
        Dog dog = new Dog("A");
        func(dog);
        System.out.println(dog.getName());          // B
    }

    private static void func(Dog dog) {
        dog.setName("B");
    }
}
```

[StackOverflow: Is Java “pass-by-reference” or “pass-by-value”?](https://stackoverflow.com/questions/40480/is-java-pass-by-reference-or-pass-by-value)

## 2. float 与 double

1.1 字面量属于 double 类型，不能直接将 1.1 直接赋值给 float 变量，因为这是向下转型。Java 不能隐式执行向下转型，因为这会使得精度降低。

```java
// float f = 1.1;
```

1.1f 字面量才是 float 类型。

```java
float f = 1.1f;
```

## 3. 隐式类型转换

**转换规则**： 

- 数值型数据的转换：`byte→short→int→long→float→double`
- 字符型转换为整型：`char→int`

因为字面量 1 是 int 类型，它比 short 类型精度要高，因此不能隐式地将 int 类型下转型为 short 类型。

```java
short s1 = 1;
// s1 = s1 + 1;
```

但是使用 += 运算符可以执行隐式类型转换。

```java
s1 += 1;
```

上面的语句相当于将 s1 + 1 的计算结果进行了向下转型：

```java
s1 = (short) (s1 + 1);
```

[StackOverflow : Why don't Java's +=, -=, *=, /= compound assignment operators require casting?](https://stackoverflow.com/questions/8710619/why-dont-javas-compound-assignment-operators-require-casting)

## 4. switch

- switch 支持 String，但不支持 long

```java
String str = "world";
switch (str) {
    case "hello":
        System.out.println("hello");
        break;
    case "world":
        System.out.println("world");
        break;
    default:
        break;
}
//反编译后的结果
String str = "world";
String s;
switch((s = str).hashCode())
{
    default:
        break;
    case 99162322:
        if(s.equals("hello"))
            System.out.println("hello");
        break;
    case 113318802:
        if(s.equals("world"))
            System.out.println("world");
        break;
}
```

[StackOverflow : Why can't your switch statement data type be long, Java?](https://stackoverflow.com/questions/2676210/why-cant-your-switch-statement-data-type-be-long-java)

# 四、继承

## 1. 访问权限

|           | 同一个类 | 同一个包 | 不同包的子类 | 不同包的非子类 |
| :-------: | :------: | :------: | :----------: | :------------: |
|  private  |    √     |          |              |                |
|  default  |    √     |    √     |              |                |
| protected |    √     |    √     |      √       |                |
|  public   |    √     |    √     |      √       |       √        |

- Java 的三个访问权限修饰符：`private、protected、public`，如果**不加访问修饰符，表示包级可见** 

  > protected 用于修饰成员，表示在继承体系中成员对于子类可见，但对于类没有意义

- 对类或类中的成员（字段以及方法）加上访问修饰符：
  - 类可见表示其它类可以用这个类创建实例对象
  - 成员可见表示其它类可以用这个类的实例对象访问到该成员

- 信息隐藏或封装： 隐藏所有的实现细节，模块之间只通过 API 进行通信

- [设计模式的六大原则](http://wiki.jikexueyuan.com/project/java-design-pattern-principle/)：

  - **单一职责原则**： 一个类只负责一项职责，防止职责扩散

  - **里氏替换原则**： 类 B 继承类 A 时，除添加新的方法完成新增功能外，尽量不要重写父类 A 的方法，也尽量不要重载父类 A 的方法

  - **依赖倒置原则**： 高层模块不依赖低层模块，二者都应该依赖其抽象；抽象不应该依赖细节，细节应该依赖抽象

    > 类 B 和类 C 各自实现接口 I，类 A 通过接口 I 间接与类B或者类C发生联系，则会降低修改类A的几率

  - **接口隔离原则**： 将臃肿的接口拆分为独立的几个接口，类A和类C分别与他们需要的接口建立依赖关系
  - **迪米特法则**： 尽量降低类与类之间的耦合
  - **开闭原则**： 当软件需要变化时，尽量通过扩展软件实体的行为来实现变化，而不是通过修改已有的代码来实现变化

## 2. 抽象类与接口

**1. 抽象类** 

抽象类和抽象方法都使用 abstract 关键字进行声明。抽象类一般会包含抽象方法，抽象方法一定位于抽象类中。

抽象类和普通类最大的区别是，抽象类不能被实例化，需要继承抽象类才能实例化其子类。

```java
public abstract class AbstractClassExample {

    protected int x;
    private int y;

    public abstract void func1();

    public void func2() {
        System.out.println("func2");
    }
}
```

```java
public class AbstractExtendClassExample extends AbstractClassExample {
    @Override
    public void func1() {
        System.out.println("func1");
    }
}
```

```java
// AbstractClassExample ac1 = new AbstractClassExample(); // 'AbstractClassExample' is abstract; cannot be instantiated
AbstractClassExample ac2 = new AbstractExtendClassExample();
ac2.func1();
```

**2. 接口** 

接口是抽象类的延伸，在 Java 8 之前，它可以看成是一个完全抽象的类，也就是说它不能有任何的方法实现。

从 Java 8 开始，接口也可以拥有默认的方法实现，这是因为不支持默认方法的接口的维护成本太高了。在 Java 8 之前，如果一个接口想要添加新的方法，那么要修改所有实现了该接口的类。

接口的成员（字段 + 方法）默认都是 public 的，并且不允许定义为 private 或者 protected。

接口的字段默认都是 static 和 final 的。

```java
public interface InterfaceExample {

    void func1();

    default void func2(){
        System.out.println("func2");
    }

    int x = 123;
    // int y;               // Variable 'y' might not have been initialized
    public int z = 0;       // Modifier 'public' is redundant for interface fields
    // private int k = 0;   // Modifier 'private' not allowed here
    // protected int l = 0; // Modifier 'protected' not allowed here
    // private void fun3(); // Modifier 'private' not allowed here
}
```

```java
public class InterfaceImplementExample implements InterfaceExample {
    @Override
    public void func1() {
        System.out.println("func1");
    }
}
```

```java
// InterfaceExample ie1 = new InterfaceExample(); // 'InterfaceExample' is abstract; cannot be instantiated
InterfaceExample ie2 = new InterfaceImplementExample();
ie2.func1();
System.out.println(InterfaceExample.x);
```

**3. 比较** 

- 从设计层面上看，抽象类提供了一种 IS-A 关系，那么就必须满足里式替换原则，即子类对象必须能够替换掉所有父类对象。而接口更像是一种 LIKE-A 关系，它只是提供一种方法实现契约，并不要求接口和实现接口的类具有 IS-A 关系。
- 从使用上来看，一个类可以实现多个接口，但是不能继承多个抽象类。
- 接口的字段只能是 static 和 final 类型的，而抽象类的字段没有这种限制。
- 接口的成员只能是 public 的，而抽象类的成员可以有多种访问权限。

**4. 使用选择** 

使用接口：

- 需要让不相关的类都实现一个方法，例如不相关的类都可以实现 Compareable 接口中的 compareTo() 方法；
- 需要使用多重继承。

使用抽象类：

- 需要在几个相关的类中共享代码。
- 需要能控制继承来的成员的访问权限，而不是都为 public。
- 需要继承非静态和非常量字段。

在很多情况下，接口优先于抽象类。因为接口没有抽象类严格的类层次结构要求，可以灵活地为一个类添加行为。并且从 Java 8 开始，接口也可以有默认的方法实现，使得修改接口的成本也变的很低。

- [深入理解 abstract class 和 interface](https://www.ibm.com/developerworks/cn/java/l-javainterface-abstract/)
- [When to Use Abstract Class and Interface](https://dzone.com/articles/when-to-use-abstract-class-and-intreface)

## 3. super

作用： 

- **访问父类的构造函数**
- **访问父类的成员**

```java
public class SuperExample {
    protected int x;
    protected int y;

    public SuperExample(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void func() {
        System.out.println("SuperExample.func()");
    }
}
```

```java
public class SuperExtendExample extends SuperExample {
    private int z;

    public SuperExtendExample(int x, int y, int z) {
        super(x, y);
        this.z = z;
    }

    @Override
    public void func() {
        super.func();
        System.out.println("SuperExtendExample.func()");
    }
}
```

```java
SuperExample e = new SuperExtendExample(1, 2, 3);
e.func();
```

```html
SuperExample.func()
SuperExtendExample.func()
```

[Using the Keyword super](https://docs.oracle.com/javase/tutorial/java/IandI/super.html)

## 4. this

1. 使用在类中，可以用来修饰属性、方法、构造器
2. 表示当前对象或者是当前正在创建的对象
3. 当形参与成员变量重名时，如果在方法内部需要使用成员变量，必须添加this来表明该变量时类成员
4. 在任意方法内，如果使用当前类的成员变量或成员方法可以在其前面添加this，增强程序的阅读性
5. 在构造器中使用“this(形参列表)”显式的调用本类中重载的其它的构造器
   1. **要求“this(形参列表)”要声明在构造器的首行**！
   2. 类中若存在n个构造器，那么最多有n-1构造器中使用了this

```java
public class TestPerson {
	public static void main(String[] args) {
		Person p1 = new Person();
		System.out.println(p1.getName() + ":" + p1.getAge());
		
		Person p2 = new Person("BB",23);
		int temp = p2.compare(p1);
		System.out.println(temp);
	}
}
class Person{	
	private String name;
	private int age;
	
	public Person(){
		this.name = "AA";
		this.age = 1;
	}	
	public Person(String name){
		this();
		this.name = name;
	}
	public Person(String name,int age){
		this(name);
		this.age = age;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getAge() {
		return age;
	}
	public void setAge(int age) {
		this.age = age;
	}
	public void eat(){
		System.out.println("eating");
	}
	public void sleep(){
		System.out.println("sleeping");
		this.eat();
	}
	//比较当前对象与形参的对象的age谁大。
	public int compare(Person p){
		if(this.age > p.age)
			return 1;
		else if(this.age < p.age)
			return -1;
		else
			return 0;
	}	
}
```

## 5. 重写与重载

**1. 重写（Override）** 

- 定义： 指子类实现了一个与父类在方法声明上完全相同的一个方法

- 为满足里式替换原则，重写的限制：
  - **子类方法的访问权限必须大于等于父类方法**
  - **子类方法的返回类型必须是父类方法返回类型或为其子类型**

  使用 @Override 注解，可以让编译器帮忙检查是否满足上面的两个限制条件

**2. 重载（Overload）** 

- 定义： 指一个方法与已经存在的方法名称上相同，但是参数类型、个数、顺序至少有一个不同

  > 注意： **返回值不同，其它都相同不算是重载**

## 6. 其他关系模型

- **[继承、实现、依赖、关联、聚合、组合的联系与区别](https://www.cnblogs.com/jiqing9006/p/5915023.html)**

# 五、Object 通用方法

## 1. 概览 

```java
public final native Class<?> getClass()
public native int hashCode()
public boolean equals(Object obj)
protected native Object clone() throws CloneNotSupportedException
public String toString()
public final native void notify()
public final native void notifyAll()
public final native void wait(long timeout) throws InterruptedException
public final void wait(long timeout, int nanos) throws InterruptedException
public final void wait() throws InterruptedException
protected void finalize() throws Throwable {}
```

## 2. equals()

- 推荐阅读： **[equals()与hashCode()方法详解](https://www.cnblogs.com/Qian123/p/5703507.html)** 

```java
//Object 类中的实现
public boolean equals(Object obj) {  
    return (this == obj);  
}  
```

**1. 等价关系** 

- **自反性**： 对于任意不为 `null` 的引用值x，`x.equals(x)` 一定是 `true`

  ```java
  x.equals(x); // true
  ```

- **对称性**： 对于任意不为`null`的引用值`x`和`y`，当且仅当`x.equals(y)`是`true`时，`y.equals(x)`也是`true`

  ```java
  x.equals(y) == y.equals(x); // true
  ```

- **传递性**： 对于任意不为`null`的引用值`x`、`y`和`z`，如果`x.equals(y)`是`true`，同时`y.equals(z)`是`true`，那么`x.equals(z)`一定是`true`

  ```java
  if (x.equals(y) && y.equals(z))
      x.equals(z); // true
  ```

- **一致性**： 对于任意不为`null`的引用值`x`和`y`，如果用于equals比较的对象信息没有被修改的话，多次调用时`x.equals(y)`要么一致地返回`true`要么一致地返回`false`

- **与 null 的比较**： 对于任意不为`null`的引用值`x`，`x.equals(null)`返回`false` 

  ```java
  x.equals(null); // false
  ```

**2. 等价与相等** 

- 对于基本类型，== 判断两个值是否相等，基本类型没有 equals() 方法
- 对于引用类型，== 判断两个变量是否引用同一个对象，而 equals() 判断引用的对象是否等价

```java
Integer x = new Integer(1);
Integer y = new Integer(1);
System.out.println(x.equals(y)); // true
System.out.println(x == y);      // false
```

**3. 实现** 

```java
public class EqualExample {

    private int x;
    private int y;
    private int z;

    public EqualExample(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public boolean equals(Object o) {
        //检查是否为同一个对象的引用，如果是直接返回 true
        if (this == o) return true;
        //检查是否是同一个类型，如果不是，直接返回 false
        if (o == null || getClass() != o.getClass()) return false;
		//将 Object 对象进行转型
        EqualExample that = (EqualExample) o; 
		//判断每个关键域是否相等
        if (x != that.x) return false;
        if (y != that.y) return false;
        return z == that.z;
    }
}
```

- 注意： **当equals()方法被override时，hashCode()也要被override**

## 3. hashCode()

```java
//Object 类中的 hashCode
public native int hashCode();
```

- hashCode() 返回散列值，而 equals() 是用来判断两个对象是否等价
- 等价的两个对象散列值一定相同，但是散列值相同的两个对象不一定等价
- 在覆盖 equals() 方法时应当总是覆盖 hashCode() 方法，保证等价的两个对象散列值也相等
- 理想的散列函数应当具有均匀性，即不相等的对象应当均匀分布到所有可能的散列值上
- equals相等两个对象，则hashcode一定要相等。但是hashcode相等的两个对象不一定equals相等

## 4. toString()

默认返回 ToStringExample@4554617c 这种形式，其中 @ 后面的数值为散列码的无符号十六进制表示

```java
public class ToStringExample {
    private int number;

    public ToStringExample(int number) {
        this.number = number;
    }
}
```

```java
ToStringExample example = new ToStringExample(123);
System.out.println(example.toString());
```

```html
ToStringExample@4554617c
```

## 5. clone() 

### 1. 简介

- clone() 是 Object 的 protected 方法，即一个类不显式去重写 clone()，其它类就不能直接去调用该类实例的 clone() 方法

  ```java
  //重写 clone()
  public class CloneExample {
      private int a;
      private int b;
  
      @Override
      protected CloneExample clone() throws CloneNotSupportedException {
          return (CloneExample)super.clone();
      }
  }
  ```

- 如果一个类没有实现 Cloneable 接口又调用了 clone() 方法，就会抛出 CloneNotSupportedException

### 2. 浅拷贝

- 定义： **拷贝对象和原始对象的引用类型引用同一个对象**

```java
public class ShallowCloneExample implements Cloneable {
    private int[] arr;

    public ShallowCloneExample() {
        arr = new int[10];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = i;
        }
    }
    public void set(int index, int value) {
        arr[index] = value;
    }
    public int get(int index) {
        return arr[index];
    }
    @Override
    protected ShallowCloneExample clone() throws CloneNotSupportedException {
        return (ShallowCloneExample) super.clone();
    }
}

//测试
ShallowCloneExample e1 = new ShallowCloneExample();
ShallowCloneExample e2 = null;
try {
    e2 = e1.clone();
} catch (CloneNotSupportedException e) {
    e.printStackTrace();
}
e1.set(2, 222);
System.out.println(e2.get(2)); // 222
```

### 3. 深拷贝

- 定义： **拷贝对象和原始对象的引用类型引用不同对象**

```java
public class DeepCloneExample implements Cloneable {
    private int[] arr;

    public DeepCloneExample() {
        arr = new int[10];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = i;
        }
    }
    public void set(int index, int value) {
        arr[index] = value;
    }
    public int get(int index) {
        return arr[index];
    }

    @Override
    protected DeepCloneExample clone() throws CloneNotSupportedException {
        DeepCloneExample result = (DeepCloneExample) super.clone();
        result.arr = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result.arr[i] = arr[i];
        }
        return result;
    }
}

//测试
DeepCloneExample e1 = new DeepCloneExample();
DeepCloneExample e2 = null;
try {
    e2 = e1.clone();
} catch (CloneNotSupportedException e) {
    e.printStackTrace();
}
e1.set(2, 222);
System.out.println(e2.get(2)); // 2
```

### 4. clone() 的替代方案

- 缺点： **使用 clone() 方法来拷贝一个对象即复杂又有风险，它会抛出异常，并且还需要类型转换**
- 替代： **可以使用拷贝构造函数或者拷贝工厂来拷贝一个对象** 

```java
public class CloneConstructorExample {
    private int[] arr;

    public CloneConstructorExample() {
        arr = new int[10];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = i;
        }
    }
    public CloneConstructorExample(CloneConstructorExample original) {
        arr = new int[original.arr.length];
        for (int i = 0; i < original.arr.length; i++) {
            arr[i] = original.arr[i];
        }
    }
    public void set(int index, int value) {
        arr[index] = value;
    }
    public int get(int index) {
        return arr[index];
    }
}

//测试
CloneConstructorExample e1 = new CloneConstructorExample();
CloneConstructorExample e2 = new CloneConstructorExample(e1);
e1.set(2, 222);
System.out.println(e2.get(2)); // 2
```

# 六、关键字

## 1. final

- 用来修饰数据，包括成员变量和局部变量，该变量只能被赋值一次且它的值无法被改变

  > 对于成员变量来讲，我们必须在声明时或者构造方法中对它赋值

- 用来修饰方法参数，表示在变量的生存期中它的值不能被改变

- 修饰方法，表示该方法无法被重写

- 修饰类，表示该类无法被继承

## 2. static

### 1. 静态变量

- **静态变量**：又称为类变量，即该变量属于类的，类所有的实例都共享静态变量，可以直接通过类名来访问
- **实例变量**：每创建一个实例就会产生一个实例变量，与该实例同生共死

### 2. 静态方法

- 静态方法在类加载时已存在，不依赖于任何实例
- 静态方法必须有实现，即不能是抽象方法

- **静态方法只能访问所属类的静态字段和静态方法**，方法中不能有 this 和 super 关键字

### 3. 静态语句块

- 静态语句块在类初始化时运行一次

```java
static{
    //...
}
```

### 4. 静态内部类

- 非静态内部类依赖于外部类的实例，而静态内部类不需要
- 静态内部类不能访问外部类的非静态的变量和方法

```java
public class OuterClass {
    class InnerClass { //非静态内部类
    }
    static class StaticInnerClass { //静态内部类
    }

    public static void main(String[] args) {
        OuterClass outerClass = new OuterClass();
        InnerClass innerClass = outerClass.new InnerClass();//依赖于外部类的实例
        StaticInnerClass staticInnerClass = new StaticInnerClass();//不依赖
    }
}
```

### 5. 静态导包

- 在使用静态变量和方法时不用再指明 ClassName，从而简化代码，但可读性大大降低
- 静态导包后，当调用类的静态方法时，不需要加上类名

```java
import static com.xxx.ClassName.* //静态导包格式

//测试
public class StaticDemo {
    public static void sayHi() {
        System.out.println("Hi");
    }
    
    public static void sayBye() {
        System.out.println("Bye");
    }
}
//调用
import static com.jas.test.StaticDemo.*; //静态导入

public class StaticDemoDriven {
    public static void main(String[] args) {
        //直接调用，不用加类名
        sayHi(); 
        sayBye();
    }
}
```

### 6. 初始化顺序

- 静态变量和静态语句块优先于实例变量和普通语句块，静态变量和静态语句块的初始化顺序取决于它们在代码中的顺序
- 存在继承的情况下，**初始化顺序**：
  - 父类（静态变量、静态语句块）
  - 子类（静态变量、静态语句块）
  - 父类（实例变量、普通语句块）
  - 父类（构造函数）
  - 子类（实例变量、普通语句块）
  - 子类（构造函数）


## 3. native 

> 实现 java 与其他语言的交互（如：C，C++）

[java中的native关键字](http://www.blogjava.net/shiliqiang/articles/287920.html)

[Java中Native关键字的作用](https://www.cnblogs.com/Qian123/p/5702574.html) 

## 4. transient 

- **[Java 序列化的高级认识](https://www.ibm.com/developerworks/cn/java/j-lo-serial/index.html?mhq=%E4%BB%80%E4%B9%88%E6%98%AF%E5%BA%8F%E5%88%97%E5%8C%96%E4%B8%8E%E5%8F%8D%E5%BA%8F%E5%88%97%E5%8C%96%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E5%BA%8F%E5%88%97%E5%8C%96)** 
- **[序列化与单例模式](http://www.cnblogs.com/ixenos/p/5831067.html)** 
- **[Java反序列化漏洞分析](https://www.cnblogs.com/he1m4n6a/p/10131566.html)** 



- 序列化： 把实体对象状态按照一定的格式写入到有序字节流

  反序列化： 从有序字节流重建对象，恢复对象状态

- 序列化的作用：

  - 永久性保存对象，保存对象的字节序列到本地文件或者数据库中
  - 通过序列化以字节流的形式使对象在网络中进行传递和接收
  - 通过序列化在进程间传递对象

- 序列化步骤：

  - 将对象实例相关的类元数据输出

    > 元数据： 数据的数据，即描述代码间关系，或者代码与其他资源之间内在联系的数据
    >
    > 四种类型的元数据： 类、枚举、接口、注解
    >
    > 框架的元数据： xml 配置文件

  - 递归地输出类的超类描述直到不再有超类

  - 类元数据完了以后，开始从最顶层的超类开始输出对象实例的实际数据值

  - 从上至下递归输出实例的数据

- `serialVersionUID`，类路径，功能代码决定了虚拟机是否允许反序列化

- 静态变量不属于对象，属于类，不能被序列化，即序列化不保存静态变量

  ```java
  public class Test implements Serializable {
      private static final long serialVersionUID = 1L;
      public static int staticVar = 5; //静态变量
      public static void main(String[] args) {
          try {
              //初始时staticVar为5
              ObjectOutputStream out = new ObjectOutputStream(
                      new FileOutputStream("result.obj"));
              out.writeObject(new Test());
              out.close();
              //序列化后修改为10
              Test.staticVar = 10;
   			//反序列化并读取
              ObjectInputStream oin = new ObjectInputStream(new FileInputStream(
                      "result.obj"));
              Test t = (Test) oin.readObject();
              oin.close();
              //再读取，通过t.staticVar打印新的值
              System.out.println(t.staticVar); //结果为 10
          } catch (Exception e) {
              e.printStackTrace();
          }
      }
  }
  ```

- 若子类实现 Serializable 接口而父类未实现时，对子类进行反序列化时，会默认调用父类的无参构造器作为默认父类对象，因此父类对象的值为无参构造器的值

  > - 父类的无参构造器若未明显指定，则变量值为默认声明的值，如： int 型为 0，String 型为 null
  > - 部分字段放在父类中的好处：当有另外一个 Child 类时，字段不会被序列化，不用重复写 transient

- Transient 关键字： 避免变量被序列化，在被反序列化后，transient 变量的值被设为初始值，如 int 型的是 0，对象型的是 null

- 序列化过程中，虚拟机会调用对象类中的 `writeObject` 与 `readObject` 方法进行序列化与反序列化

  > - 该方法可以用户自定义，允许用户控制序列化的过程，如： 动态改变序列化的值，对敏感字段加密
  >
  > - 实现 Externalizable 接口的类完全由自身来控制序列化的行为
  >
  >   实现 Serializable 接口的类采用默认的序列化方式

- 序列化对同一对象进行多次写入时，不会再次存储，而是再次存储一份引用

- 序列化会破坏单例模式(发射会破坏单例模式)： **反序列化会通过反射调用无参数的构造方法创建一个新的对象**

  ```java
  class Singleton implements Serializable{
      //静态内部类实现单例
  	private static class SingletonClassInstance {
  	    private static final Singleton instance = new Singleton();
  	}
  	public static Singleton getInstance() {
  	    return SingletonClassInstance.instance;
  	}
  	private Singleton() {}
  }
  //测试
  //Write Obj to file
  ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("tempFile"));
  oos.writeObject(Singleton.getSingleton());
  //Read Obj from file
  File file = new File("tempFile");
  ObjectInputStream ois =  new ObjectInputStream(new FileInputStream(file));
  Singleton newInstance = (Singleton) ois.readObject();
  //判断是否是同一个对象
  System.out.println(newInstance == Singleton.getSingleton()); //&& false &&
  ```

  - 解决方法一： **在Singleton类中定义readResolve**，并指定对象的生成策略

    反序列化过程： **readObject--->readObject0--->readOrdinaryObject--->checkResolve** 

    单例破坏反序列化原因： 若类可被实例化，则会通过反射的方式调用无参构造方法**新建一个对象**

    ```java
    //指定获取单例策略
    private Object readResolve() throws ObjectStreamException {  
        return SingletonClassInstance.instance;
    }
    ```

  - 解决方法二： 通过枚举实现单例

    ```java
    //每一个枚举类型极其定义的枚举变量在JVM中是唯一的
    enum SingletonEnum{
        INSTANCE;
        
        private Resource instance;
        SingletonEnum() {
            instance = new Resource();
        }
        public Resource getInstance() {
            return instance;
        }
    }
    ```

- Java 序列化漏洞： 通过 `apach common collections` 实现，[Java反序列化漏洞分析](https://www.cnblogs.com/he1m4n6a/p/10131566.html) 

  > **攻击目标**： 可序列化的类重写了 readObject() 方法且使用 Map 类型的变量进行了键值修改操作

  - 首先构造一个 Map 和一个能够执行代码的 ChainedTransformer

    > ChainedTransformer 用于挨个执行定义的命令

  - 生成一个 TransformedMap 实例，用于修改 Map 中的数据

    > 利用其 value 修改时触发 transform() 特性

  - 实例化 AnnotationInvocationHandler，并对其进行序列化，用于检测 readObject 方法

  - 当触发readObject()反序列化的时候，就能实现命令执行

  > POC执行流程： TransformedMap->AnnotationInvocationHandler.readObject()->setValue()-> 漏洞成功触发

## 5. abstract

abstract：抽象的，可以用来修饰类、方法

1. abstract修饰类：抽象类
   1. 不可被实例化
   2. 抽象类有构造器 (凡是类都有构造器)
   3. 抽象方法所在的类，一定是抽象类
   4. 抽象类中可以没有抽象方法


2. abstract修饰方法：抽象方法
   1. 格式：没有方法体，包括{}.如：public abstract void eat();
   2. 抽象方法只保留方法的功能，而具体的执行，交给继承抽象类的子类，由子类重写此抽象方法
   3. 若子类继承抽象类，并重写了所有的抽象方法，则此类是一个"实体类",即可以实例化
   4. 若子类继承抽象类，没有重写所有的抽象方法，意味着此类中仍有抽象方法，则此类必须声明为抽象的

## 6. interface

接口是与类并行的一个概念

1. 接口可以看做是一个特殊的抽象类。是常量与抽象方法的一个集合，不能包含变量、一般的方法
2. 接口是没有构造器的
3. 接口定义的就是一种功能，此功能可以被类所实现
4. 实现接口的类，必须要重写其中的所有的抽象方法，若没有重写所有的抽象方法，则此类仍为一个抽象类
5. 类可以实现多个接口
6. 接口与接口之间也是继承的关系，而且可以实现多继承
7. 接口与具体的实现类之间也存在多态性

## 7. 代码块

执行顺序：静态代码块>mian方法>构造代码块>构造方法
作用：用来初始化类的属性


1. **静态代码块**： 
   1. 使用 static 关键字声明的代码块
   2. 里面可以有输出语句
   3. 随着类的加载而加载，而且只被加载一次
   4. 多个静态代码块之间按照顺序结构执行
   5. 静态代码块的执行要早于非静态代码块的执行
   6. 静态的代码块中只能执行静态的结构(类属性，类方法)
   7. 静态代码块不能存在于任何方法体内
   8. 静态代码块不能直接访问静态实例变量和实例方法，需要通过类的实例对象来访问
2. **非静态代码块：** 普通代码块，构造代码块，同步代码块

   1. 可以对类的属性(静态的 & 非静态的)进行初始化操作，也可以调用本类声明的方法(静态的 & 非静态的)
   2. 里面可以有输出语句
   3. 一个类中可以有多个非静态的代码块，多个代码块之间按照顺序结构执行
   4. 每创建一个类的对象，非静态代码块就加载一次
   5. 非静态代码块的执行要早于构造器

# 七、反射

**[深入解析 Java 反射（1）- 基础](http://www.sczyh30.com/posts/Java/java-reflection-1/)** 

- Java Reflection     
  - 反射机制允许程序在执行期借助于Reflection API取得任何类的内部信息，并能直接操作任意对象的内部属性及方法
- Java反射机制提供的功能
  - 在运行时判断任意一个对象所属的类
  - 在运行时构造任意一个类的对象
  - 在运行时判断任意一个类所具有的成员变量和方法
  - 在运行时调用任意一个对象的成员变量和方法
  - 生成动态代理

Class 和 java.lang.reflect 一起对反射提供了支持，java.lang.reflect 类库主要包含了以下三个类：

- **java.lang.Class：** 代表一个类,是反射的源头
- **Field** ：可以使用 get() 和 set() 方法读取和修改 Field 对象关联的字段（代表类的成员变量）
- **Method** ：可以使用 invoke() 方法调用与 Method 对象关联的方法（代表类的方法）
- **Constructor** ：可以用 Constructor 创建新的对象（代表类的构造方法）

##1.  理解Class类并实例化Class类对象

在Object类中定义了以下的方法，此方法将被所有子类继承：

- public final Class getClass()

对于每个类而言，JRE 都为其保留一个不变的 Class 类型的对象。一个 Class 对象包含了特定某个类的有关信息： 

- Class 本身也是一个类
- Class 对象只能由系统建立对象
- 一个类在 JVM 中只会有一个 Class 实例 
- 一个 Class 对象对应的是一个加载到JVM中的一个.class文件
- 每个类的实例都会记得自己是由哪个 Class 实例所生成
- 通过 Class 可以完整地得到一个类中的完整结构 

```java
@Test
public void test3(){
    Person p = new Person();
    Class clazz = p.getClass();//通过运行时类的对象，调用其getClass()，返回其运行时类。
    System.out.println(clazz);
}
	
```

![](../pics/reflect.png)

### 1. 实例化Class类对象(四种方法) 

1. 前提：若已知具体的类，**通过类的class属性获取**，该方法最为安全可靠，程序性能最高       

  实例：Class clazz = String.class

2. 前提：已知某个类的实例，**调用该实例的 getClass() 方法获取Class对象**       

   实例：Class clazz = person.getClass()

3. 前提：已知一个类的全类名，且该类在类路径下，可**通过Class类的静态方法forName()获取**，可能抛出ClassNotFoundException       

   实例：Class clazz = Class.forName(“java.lang.String”)

4. 其他方式(不做要求)

   ClassLoader cl = this.getClass().getClassLoader()

   Class clazz4 = cl.loadClass(“类的全类名”)

```java
@Test
public void test4() throws ClassNotFoundException{
    //1.调用运行时类本身的.class属性
    Class clazz1 = Person.class;
    System.out.println(clazz1.getName());

    Class clazz2 = String.class;
    System.out.println(clazz2.getName());

    //2.通过运行时类的对象获取
    Person p = new Person();
    Class clazz3 = p.getClass();
    System.out.println(clazz3.getName());

    //3.通过Class的静态方法获取.通过此方式，体会一下，反射的动态性。
    String className = "com.java.Person";//要加载类的路径
    Class clazz4 = Class.forName(className);
    System.out.println(clazz4.getName());

    //4.（了解）通过类的加载器
    ClassLoader classLoader = this.getClass().getClassLoader();
    Class clazz5 = classLoader.loadClass(className);
    System.out.println(clazz5.getName());

    System.out.println(clazz1 == clazz3);//true
    System.out.println(clazz1 == clazz4);//true
    System.out.println(clazz1 == clazz5);//true
}
```

**Class.forName 和 ClassLoader 区别**： Class.forName 会运行静态代码块，静态方法

**[在Java的反射中，Class.forName和ClassLoader的区别](https://www.cnblogs.com/jimoer/p/9185662.html)**

- `Class.forName`： 对类进行了初始化

  > JDBC 使用该方式加载数据库连接驱动

- `ClassLoader`： 没有对类进行初始化，只是把类加载到了虚拟机中

  > SpringIoc 的实现

### 2. 类的加载过程

当程序主动使用某个类时，如果该类还未被加载到内存中，则系统会通过如下三个步骤来对该类进行初始化：

1. 类的加载：将类的class文件读入内存，并为之创建一个 java.lang.Class 对象。此过程由类加载器完成
2. 类的连接：将类的二进制数据合并到JRE中
3. 类的初始化：JVM 负责对类进行初始化

![](../pics/reflect2.png)

### 3. ClassLoader

类加载器是用来把类(class)装载进内存的

JVM 规范定义了两种类型的类加载器：启动类加载器(bootstrap)和用户自定义加载器(user-defined class loader)

 JVM在运行时会产生3个类加载器组成的初始化加载器层次结构

- 引导类加载器：用C++编写的，是JVM自带的类加载器，负责Java平台核心库，用来加载核心类库
- 扩展类加载器：负责jre/lib/ext目录下的jar包或 –D java.ext.dirs 指定目录下的jar包装入工作库
- 系统类加载器：负责java –classpath 或 –D java.class.path所指的目录下的类与jar包装入工作 

![](../pics/reflect3.png)

关于类的加载器：ClassLoader(引导类加载器与核心类库无法加载)

```java
@Test
public void test5() throws Exception{
    ClassLoader loader1 = ClassLoader.getSystemClassLoader();
    System.out.println(loader1);

    ClassLoader loader2 = loader1.getParent();
    System.out.println(loader2);

    ClassLoader loader3 = loader2.getParent();
    System.out.println(loader3);//null

    Class clazz1 = Person.class;
    ClassLoader loader4 = clazz1.getClassLoader();
    System.out.println(loader4);

    String className = "java.lang.String";
    Class clazz2 = Class.forName(className);
    ClassLoader loader5 = clazz2.getClassLoader();
    System.out.println(loader5);//null

    //掌握如下
    //法一：
    ClassLoader loader = this.getClass().getClassLoader();
    InputStream is = loader.getResourceAsStream("com\\java\\jdbc.properties");//文件在一个包中
    //法二：
//		FileInputStream is = new FileInputStream(new File("jdbc1.properties"));//文件在当前工程下

    Properties pros = new Properties();
    pros.load(is);
    String name = pros.getProperty("user");
    System.out.println(name);

    String password = pros.getProperty("password");
    System.out.println(password);

}
```

##2. 运行时创建类对象并获取类的完整结构

### 1. 通过Class对象创建类的对象

1. 调用Class对象的 **newInstance()方法** 

   要求：

   1. 类必须有一个无参数的构造器 
   2. 类的构造器的访问权限需要足够

   ```java
   @Test
   public void test1() throws Exception{
       String className = "com.java.Person";
       Class clazz = Class.forName(className);
       //创建对应的运行时类的对象。使用newInstance()，实际上就是调用了运行时类的空参的构造器
       //要想能够创建成功：①要求对应的运行时类要有空参的构造器。②构造器的权限修饰符的权限要足够
       Object obj = clazz.newInstance();
       Person p = (Person)obj;
       System.out.println(p);
   }
   ```

2. 通过有参构造器创建：在操作的时候明确的调用类中的构造方法，并将参数传递进去之后，才可以实例化操作

   - 通过Class类的getDeclaredConstructor(Class … parameterTypes)取得本类的指定形参类型的构造器
   - 向构造器的形参中传递一个对象数组进去，里面包含了构造器中所需的各个参数
   - 在 Constructor  类中存在一个方法： public T newInstance(Object... initargs)

### 2. 通过反射调用类的完整结构

**使用反射可以取得：** 

1. 实现的全部接口

   - public Class<?>[] getInterfaces()：确定此对象所表示的类或接口实现的接口 

2. 所继承的父类

   - public Class<? Super T> getSuperclass()：返回表示此 Class 所表示的实体（类、接口、基本类型）的父类的 Class

3. 全部的构造器

   - public Constructor\<T>[] getConstructors()：返回此 Class 对象所表示的类的所有public构造方法


   - public Constructor\<T>[] getDeclaredConstructors()：返回此 Class 对象表示的类声明的所有构造方法

   **Constructor类中：** 

   - 取得修饰符: public int getModifiers()
   - 取得方法名称: public String getName()
   - 取得参数的类型：public Class<?>[] getParameterTypes()

4. 全部的方法

   - public Method[] getDeclaredMethods()：返回此Class对象所表示的类或接口的全部方法


   - public Method[] getMethods()：返回此Class对象所表示的类或接口的public的方法

   **Method类中：** 

   - public Class<?> getReturnType()：取得全部的返回值
   - public Class<?>[] getParameterTypes()：取得全部的参数
   - public int getModifiers()：取得修饰符
   - public Class<?>[] getExceptionTypes()：取得异常信息

5. 全部的Field

   - public Field[] getFields()：返回此Class对象所表示的类或接口的public的Field
   - public Field[] getDeclaredFields()：返回此Class对象所表示的类或接口的全部Field

   **Field方法中：** 

   - public int getModifiers()：以整数形式返回此Field的修饰符
   - public Class<?> getType()：得到Field的属性类型
   - public String getName()：返回Field的名称

6. Annotation相关

   - getAnnotation(Class\<T> annotationClass) 
   - getDeclaredAnnotations() 

7. 泛型相关

   - 获取父类泛型类型：Type getGenericSuperclass()
   - 泛型类型：ParameterizedType
   - 获取实际的泛型类型参数数组：getActualTypeArguments()

8. 类所在的包    Package getPackage() 

**Constructor 创建对应的运行时类的对象：** 

```java
@Test
public void test1() throws Exception{
    String className = "com.java.Person";
    Class clazz = Class.forName(className);
    //创建对应的运行时类的对象。使用newInstance()，实际上就是调用了运行时类的空参的构造器。
    //要想能够创建成功：①要求对应的运行时类要有空参的构造器。②构造器的权限修饰符的权限要足够。
    Object obj = clazz.newInstance();
    Person p = (Person)obj;
    System.out.println(p);
}
```

**Constructor 获取所有的构造器：** 

```java
@Test
public void test2() throws ClassNotFoundException{
    String className = "com.java.Person";
    Class clazz = Class.forName(className);
    Constructor[] cons = clazz.getDeclaredConstructors();
    for(Constructor c : cons){
        System.out.println(c);
    }
}
```

**Field 获取对应的运行时类的属性：** 

````java
@Test
public void test1(){
    Class clazz = Person.class;
    //1.getFields():只能获取到运行时类中及其父类中声明为public的属性
    Field[] fields = clazz.getFields();
    for(int i = 0;i < fields.length;i++){
        System.out.println(fields[i]);
    }
    System.out.println();
    //2.getDeclaredFields():获取运行时类本身声明的所有的属性
    Field[] fields1 = clazz.getDeclaredFields();
    for(Field f : fields1){
        System.out.println(f.getName());
    }
}
````

**Field 获取属性的各个部分的内容：** 权限修饰符，变量类型，变量名

```java
@Test
public void test2(){
    Class clazz = Person.class;
    Field[] fields1 = clazz.getDeclaredFields();
    for(Field f : fields1){
        //1.获取每个属性的权限修饰符
        int i = f.getModifiers();
        String str1 = Modifier.toString(i);
        System.out.print(str1 + " ");
        //2.获取属性的类型
        Class type = f.getType();
        System.out.print(type.getName() + " ");
        //3.获取属性名
        System.out.print(f.getName());

        System.out.println();
    }
}
```

**Method 获取运行时类的方法：** 

```java
@Test
public void test1(){
    Class clazz = Person.class;
    //1.getMethods():获取运行时类及其父类中所有的声明为public的方法
    Method[] m1 = clazz.getMethods();
    for(Method m : m1){
        System.out.println(m);
    }
    System.out.println();

    //2.getDeclaredMethods():获取运行时类本身声明的所有的方法
    Method[] m2 = clazz.getDeclaredMethods();
    for(Method m : m2){
        System.out.println(m);
    }
}	
```

**Method 注解 权限修饰符 返回值类型 方法名 形参列表 异常：** 

```java
@Test
public void test2(){
    Class clazz = Person.class;

    Method[] m2 = clazz.getDeclaredMethods();
    for(Method m : m2){
        //1.注解
        Annotation[] ann = m.getAnnotations();
        for(Annotation a : ann){
            System.out.println(a);
        }
        //2.权限修饰符
        String str = Modifier.toString(m.getModifiers());
        System.out.print(str + " ");
        //3.返回值类型
        Class returnType = m.getReturnType();
        System.out.print(returnType.getName() + " ");
        //4.方法名
        System.out.print(m.getName() + " ");
        //5.形参列表
        System.out.print("(");
        Class[] params = m.getParameterTypes();
        for(int i = 0;i < params.length;i++){
            System.out.print(params[i].getName() + " args-" + i + " ");
        }
        System.out.print(")");
        //6.异常类型
        Class[] exps = m.getExceptionTypes();
        if(exps.length != 0){
            System.out.print("throws ");
        }
        for(int i = 0;i < exps.length;i++){
            System.out.print(exps[i].getName() + " ");
        }
        System.out.println();
    }
}
```

**获取其他的属性：** 

```java
public class TestOthers {
  
  	//1.获取运行时类的父类
	@Test
	public void test1(){
		Class clazz = Person.class;
		Class superClass = clazz.getSuperclass();
		System.out.println(superClass);
	}
  
  	//2.获取带泛型的父类
	@Test
	public void test2(){
		Class clazz = Person.class;
		Type type1 = clazz.getGenericSuperclass();
		System.out.println(type1);
	}
  
  	//3.获取父类的泛型
	@Test
	public void test3(){
		Class clazz = Person.class;
		Type type1 = clazz.getGenericSuperclass();
		
		ParameterizedType param = (ParameterizedType)type1;
		Type[] ars = param.getActualTypeArguments();
		
		System.out.println(((Class)ars[0]).getName());
	}
  
  	//4.获取实现的接口
	@Test
	public void test4(){
		Class clazz = Person.class;
		Class[] interfaces = clazz.getInterfaces();
		for(Class i : interfaces){
			System.out.println(i);
		}
	}
  
    //5.获取所在的包
	@Test
	public void test5(){
		Class clazz = Person.class;
		Package pack = clazz.getPackage();
		System.out.println(pack);
	}
  
	//6.获取注解
	@Test
	public void test6(){
		Class clazz = Person.class;
		Annotation[] anns = clazz.getAnnotations();
		for(Annotation a : anns){
			System.out.println(a);
		}
	}	
}
```

>  其他相关类：

**Person 类：** 

```java
@MyAnnotation(value = "reflect")
public class Person extends Creature<String> implements Comparable,MyInterface{
	public String name;
	private int age;
	int id;
	//创建类时，尽量保留一个空参的构造器。
	public Person() {
		super();
	}
	public Person(String name) {
		super();
		this.name = name;
	}
	private Person(String name, int age) {
		super();
		this.name = name;
		this.age = age;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getAge() {
		return age;
	}
	public void setAge(int age) {
		this.age = age;
	}
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	@MyAnnotation(value = "abc123")
	public void show(){
		System.out.println("我是一个人！");
	}
	
	private Integer display(String nation,Integer i) throws Exception{
		System.out.println("我的国籍是：" + nation);
		return i;
	}
	@Override
	public String toString() {
		return "Person [name=" + name + ", age=" + age + "]";
	}
	@Override
	public int compareTo(Object o) {
		return 0;
	}
	
	public static void info(){
		System.out.println("中国人！");
	}
	
	class Bird{
	}
}
```

**Creature 类(Person 父类)：** 

```java
public class Creature<T>{
	public double weight;
	
	public void breath(){
		System.out.println("呼吸！");
	}
}
```

**MyInterface 接口(Person 继承)：** 

```java
public interface MyInterface extends Serializable{

}
```

**MyAnnotation 注解(Person 中使用)：** 

```java
@Target({TYPE, FIELD, METHOD, PARAMETER, CONSTRUCTOR, LOCAL_VARIABLE})
@Retention(RetentionPolicy.RUNTIME)
public @interface MyAnnotation {
	String value();
}
```

##3. 通过反射调用类的指定方法、指定属性

### 1. 调用指定方法

通过反射，调用类中的方法，通过Method类完成。步骤：

1. 通过Class类的getMethod(String name,Class…parameterTypes)方法取得一个Method对象，并设置此方法操作时所需要的参数类型

2. 之后使用Object invoke(Object obj, Object[] args)进行调用，并向方法中传递要设置的obj对象的参数信息

   **Object invoke(Object obj, Object …  args)：** 

   1. Object 对应原方法的返回值，若原方法无返回值，此时返回null
   2. 若原方法若为静态方法，此时形参Object obj可为null
   3. 若原方法形参列表为空，则Object[] args为null
   4. 若原方法声明为private,则需要在调用此invoke()方法前，显式调用方法对象的setAccessible(true)方法，将可访问private的方法

**调用运行时类中指定的方法：** 

```java
@Test
public void test3() throws Exception{
    Class clazz = Person.class;
    //getMethod(String methodName,Class ... params):获取运行时类中声明为public的指定的方法
    Method m1 = clazz.getMethod("show");
    Person p = (Person)clazz.newInstance();
    //调用指定的方法：Object invoke(Object obj,Object ... obj)
    Object returnVal = m1.invoke(p);//我是一个人
    System.out.println(returnVal);//null

    Method m2 = clazz.getMethod("toString");
    Object returnVal1 = m2.invoke(p);
    System.out.println(returnVal1);//Person [name=null, age=0]
    //对于运行时类中静态方法的调用
    Method m3 = clazz.getMethod("info");
    m3.invoke(Person.class);

    //getDeclaredMethod(String methodName,Class ... params):获取运行时类中声明了指定的方法
    Method m4 = clazz.getDeclaredMethod("display",String.class,Integer.class);
    m4.setAccessible(true);
    Object value = m4.invoke(p,"CHN",10);//我的国籍是：CHN
    System.out.println(value);//10
}
```

### 2. 调用指定属性

在反射机制中，可以直接通过 Field 类操作类中的属性，通过Field类提供的set()和get()方法就可以完成设置和取得属性内容的操作：

- public Field getField(String name)：返回此Class对象表示的类或接口的指定的public的Field
- public Field getDeclaredField(String name)：返回此Class对象表示的类或接口的指定的Field

**在Field中：** 

- public Object get(Object obj) 取得指定对象obj上此Field的属性内容
- public void set(Object obj,Object value) 设置指定对象obj上此Field的属性内容

**注：** 在类中属性都设置为private的前提下，在使用set()和get()方法时，首先要使用Field类中的setAccessible(true)方法将需要操作的属性设置为可以被外部访问

- public void setAccessible(true)访问私有属性时，让这个属性可见

```java
//有了反射，可以通过反射创建一个类的对象，并调用其中的结构
@Test
public void test2() throws Exception{
    Class clazz = Person.class;
    //1.创建clazz对应的运行时类Person类的对象
    Person p = (Person)clazz.newInstance();
    System.out.println(p);
    //2.通过反射调用运行时类的指定的属性
    //2.1
    Field f1 = clazz.getField("name");
    f1.set(p,"LiuDeHua");
    System.out.println(p);
    //2.2
    Field f2 = clazz.getDeclaredField("age");
    f2.setAccessible(true);
    f2.set(p, 20);
    System.out.println(p);
    //3.通过反射调用运行时类的指定的方法
    Method m1 = clazz.getMethod("show");
    m1.invoke(p);

    Method m2 = clazz.getMethod("display",String.class);
    m2.invoke(p,"CHN");
}
```

**调用运行时类中指定的属性：** 

```java
@Test
public void test3() throws Exception{
    Class clazz = Person.class;
    //1.获取指定的属性
    //getField(String fieldName):获取运行时类中声明为public的指定属性名为fieldName的属性
    Field name = clazz.getField("name");
    //2.创建运行时类的对象 
    Person p = (Person)clazz.newInstance();
    System.out.println(p);
    //3.将运行时类的指定的属性赋值
    name.set(p,"Jerry");
    System.out.println(p);
    System.out.println("%"+name.get(p));

    System.out.println();
    //getDeclaredField(String fieldName):获取运行时类中指定的名为fieldName的属性
    Field age = clazz.getDeclaredField("age");
    //由于属性权限修饰符的限制，为了保证可以给属性赋值，需要在操作前使得此属性可被操作。
    age.setAccessible(true);
    age.set(p,10);
    System.out.println(p);
}
```

**调用指定的构造器,创建运行时类的对象：** 

```java
@Test
public void test3() throws Exception{
    String className = "com.atguigu.java.Person";
    Class clazz = Class.forName(className);

    Constructor cons = clazz.getDeclaredConstructor(String.class,int.class);
    cons.setAccessible(true);
    Person p = (Person)cons.newInstance("罗伟",20);
    System.out.println(p);
}
```

##4. 动态代理

###1. 动态代理简介

- 动态代理：指客户通过代理类来调用其它对象的方法，且是在程序运行时根据需要动态创建目标类的代理对象
- 动态代理使用场合：
  - 调试
  - 远程方法调用
- 代理设计模式的原理：使用一个代理将对象包装起来, 然后用该代理对象取代原始对象. 任何对原始对象的调用都要通过代理. 代理对象决定是否以及何时将方法调用转到原始对象上
- Proxy ：专门完成代理的操作类，是所有动态代理类的父类，通过此类为一个或多个接口动态地生成实现类
- 提供用于创建动态代理类和动态代理对象的静态方法
  - `static Class<?> getProxyClass(ClassLoader loader, Class<?>... interfaces)`  创建一个动态代理类所对应的Class对象
  - `static Object newProxyInstance(ClassLoader loader, Class<?>[] interfaces, InvocationHandler h)`  直接创建一个动态代理对象

### 2. 代理实现方式

#### 1. 静态代理模式

- **实现**： 定义接口或父类，被代理对象与代理对象实现相同接口或继承相同父类
- **优点**： 在不修改目标对象的功能前提下，对目标功能扩展
- **缺点**： 代理对象与目标对象要实现相同接口，因此会有很多代理类，维护麻烦

```java
//接口
interface ClothFactory{
	void productCloth();
}
//被代理类(目标对象)
class NikeClothFactory implements ClothFactory{
	@Override
	public void productCloth() {
		System.out.println("Nike工厂生产一批衣服");
	}	
}
//代理类
class ProxyFactory implements ClothFactory{
	ClothFactory cf;
	//创建代理类的对象时，实际传入一个被代理类的对象
	public ProxyFactory(ClothFactory cf){
		this.cf = cf;
	}
	@Override
	public void productCloth() {
		System.out.println("代理类开始执行，收代理费$1000");
		cf.productCloth();
	}
}
//测试
public class TestClothProduct {
	public static void main(String[] args) {
		NikeClothFactory nike = new NikeClothFactory();//创建被代理类的对象
		ProxyFactory proxy = new ProxyFactory(nike);//创建代理类的对象
		proxy.productCloth();
	}
}
```

#### 2. 动态代理

1. 创建一个实现接口 InvocationHandler 的类，它必须实现 invoke 方法，以完成代理的具体操作

2. 创建被代理的类以及接口

3. 通过Proxy的静态方法`newProxyInstance(ClassLoader loader,Class[] interfaces,InvocationHandler h)` 创建一个Subject 接口代理

   > - `ClassLoader loader`： 指定当前目标对象使用的类加载器
   > - `Class<?>[] interfaces`： 目标对象实现的接口的类型，使用泛型方式确认类型
   >
   > - `InvocationHandler h`： 事件处理，执行目标对象的方法时，会触发事件处理器的方法，会把当前执行目标对象的方法作为参数传入

4. 通过 Subject 代理调用 RealSubject 实现类的方法

```java
//InvocationHandler 的实现类
public class PersonInvocation implements InvocationHandler{
	Object target;
	public PersonInvocation(Object target){
		super();
		this.target = target;
	}
	@Override
	public Object invoke(Object proxy, Method method, Object[] arg2)
        	throws Throwable {
		System.out.println("修改个人信息前记录日志");
		method.invoke(target);
		System.out.println("修改个人信息后记录日志");
		return null;
	}
}
//代理类
public class PersonProxy {
	private Object target;
	private InvocationHandler ph;
	public PersonProxy(Object target,InvocationHandler ph){
		this.target = target;
		this.ph = ph;
	}
	public Object getPersonProxy(){
		Object p = Proxy.newProxyInstance(
            target.getClass().getClassLoader(),target.getClass().getInterfaces(),ph);
		return p;
	}
}
//测试
public class App {
    public static void main(String[] args) {
        // 目标对象
        IUserDao target = new UserDao();
        // 【原始的类型 class cn.itcast.b_dynamic.UserDao】
        System.out.println(target.getClass());
        // 给目标对象，创建代理对象
        IUserDao proxy = (IUserDao) new ProxyFactory(target).getProxyInstance();
        // class $Proxy0   内存中动态生成的代理对象
        System.out.println(proxy.getClass());
        // 执行方法   【代理对象】
        proxy.save();
    }
}
```

#### 3. Cglib代理

- **Cglib 代理**：也叫子类代理，在内存中构建一个子类对象从而实现对目标对象功能的扩展

- **优点**： 
  - 可以代理没有实现接口的类
  - 可以在运行期扩展 java 类，实现 java 接口，广泛用于 AOP 框架，如： Spring AOP

- **实现原理**： 底层是通过使用一个小而快的字节码处理框架 ASM 来转换字节码并生成新的类
- **注意点**： 
  - 目标对象的方法若为 `final/static`，就不会被拦截，即不会执行目标对象额外的业务方法
  - 不能对声明为 final 的方法进行代理，因为CGLib原理是动态生成被代理类的子类
- 所需 jar 包： `cglib.jar 与 asm.jar`，spring 中为 `spring-core.jar`

```java
//目标对象类
public class PersonDao {
	public void update() {
		System.out.println("修改个人信息");
	}
}

//Cglib代理工厂
public class ProxyFactory implements MethodInterceptor{
	private Object target;
	public ProxyFactory(Object target){
		this.target = target;
	}
	//给目标对象创建一个代理对象
	public Object getProxyInstance(){
		//工具类
		Enhancer en = new Enhancer();
		//设置父类
		en.setSuperclass(target.getClass());
		//设置回调函数
		en.setCallback(this);
		//创建子类代理对象
		return en.create();
	}
	
	@Override
	public Object intercept(Object obj, Method method, Object[] arg2,
                            MethodProxy proxy) throws Throwable {
		System.out.println("开始事务...");
		Object returnValue = method.invoke(target, arg2);
		System.out.println("提交事务...");
		return returnValue;
	}
}
//测试
public class App {
	public static void main(String[] args){
		PersonDao target = new PersonDao();
		ProxyFactory proxy = new ProxyFactory(target);
		PersonDao personPproxy = (PersonDao)proxy.getProxyInstance();
		personPproxy.update();
	}
}
```

#### 4. 动态代理与Cglib 代理的比较

- **原理区别**： 
  - java 动态代理利用反射机制生成一个实现代理接口的匿名类，调用具体方法前调用 InvokeHandler来处理
  - cglib 动态代理利用 asm 开源包，对代理对象类的 class 文件加载，通过修改其字节码生成子类来处理

- **字节码生成的区别**： 
  - java 动态代理只能对实现了接口的类生成代理
  - CGLIB 是针对类实现代理，主要对指定的类生成一个子类，覆盖其中的方法

- **使用场景区别**： 

  - 如果目标对象实现了接口，默认情况下会采用 JDK 的动态代理实现AOP 
  - 如果目标对象实现了接口，可以强制使用 CGLIB 实现AOP 

  - 如果目标对象没有实现接口，必须采用 CGLIB 库，spring会自动在JDK动态代理和CGLIB之间转换

    > 强制使用 Cglib： spring配置文件加入`<aop:aspectj-autoproxy proxy-target-class="true"/>`

- **性能区别**： 
  - CGLib 创建的动态代理对象性能比 JDK 创建的动态代理对象的性能高
  - CGLib 在创建代理对象时，所花费的时间却比 JDK 多
  - 对于单例的对象，因无需频繁创建对象，用 CGLib 合适；其他场景使用 java 动态代理合适

### 3. 动态代理与AOP

- 使用 Proxy 生成动态代理时，通常都是为指定的目标对象生成动态代理
- AOP代理： 可代替目标对象，AOP代理包含了目标对象的全部方法

![](../pics/reflect4.png)

```java
interface Human {
	void info();
	void fly();
}

// 被代理类
class SuperMan implements Human {
	public void info() {
		System.out.println("我是超人！我怕谁！");
	}

	public void fly() {
		System.out.println("I believe I can fly!");
	}
}

class HumanUtil {
	public void method1() {
		System.out.println("=======方法一=======");
	}
	public void method2() {
		System.out.println("=======方法二=======");
	}
}

class MyInvocationHandler implements InvocationHandler {
	Object obj;// 被代理类对象的声明

	public void setObject(Object obj) {
		this.obj = obj;
	}
	@Override
	public Object invoke(Object proxy, Method method, Object[] args)
			throws Throwable {
		HumanUtil h = new HumanUtil();
		h.method1();
		Object returnVal = method.invoke(obj, args);
		h.method2();
		return returnVal;
	}
}

class MyProxy {
	// 动态的创建一个代理类的对象
	public static Object getProxyInstance(Object obj) {
		MyInvocationHandler handler = new MyInvocationHandler();
		handler.setObject(obj);

		return Proxy.newProxyInstance(obj.getClass().getClassLoader(), obj
				.getClass().getInterfaces(), handler);
	}
}

public class TestAOP {
	public static void main(String[] args) {
		SuperMan man = new SuperMan();//创建一个被代理类的对象
		Object obj = MyProxy.getProxyInstance(man);//返回一个代理类的对象
		Human hu = (Human)obj;
		hu.info();//通过代理类的对象调用重写的抽象方法
		System.out.println();
		hu.fly();
		
		//*********
		NikeClothFactory nike = new NikeClothFactory();
		Object obj1 = MyProxy.getProxyInstance(nike);
		ClothFactory cloth = (ClothFactory)obj1;
		cloth.productCloth();
	}
}
```

# 八、异常

- **[Java 入门之异常处理](https://www.tianmaying.com/tutorial/Java-Exception)** 
- **[Java 异常的面试问题及答案 -Part 1](http://www.importnew.com/7383.html)** 

Throwable 可以用来表示任何可以作为异常抛出的类，分为两种： **Error**  和 **Exception**。其中 Error 用来表示 JVM 无法处理的错误，Exception 分为两种：

-  **受检异常** ：需要用 try...catch... 语句捕获并进行处理，并且可以从异常中恢复；
-  **非受检异常** ：是程序运行时错误，例如除 0 会引发 Arithmetic Exception，此时程序崩溃并且无法恢复。

![img](file:///D:/architect_learn/CS-Notes/pics/PPjwP.png?lastModify=1537428140) <img src="../pics//PPjwP.png" width="600"/> 

异常处理：

1. 捕获异常

   1. try：执行可能产生异常的代码
   2. catch：捕获异常
   3. finally：无论是否发生异常，代码总被执行

2. 抛出异常

   throw：异常的生成阶段，手动抛出异常对象

3. 声明异常

   throws：异常的处理方式，声明方法可能要抛出的各种异常类

# 九、泛型

- **[Java 泛型详解](http://www.importnew.com/24029.html)**
- **[10 道 Java 泛型面试题](https://cloud.tencent.com/developer/article/1033693)** 

**java 泛型面试题**： 

- **Java中的泛型是**： 防止在集合中存储对象并在使用前进行类型转换
- **泛型的好处**： 提供编译期的类型安全，确保只把正确类型的对象放入集合中，避免在运行时出现ClassCastException
- **泛型如何工作**： 泛型是通过类型擦除来实现，编译器在编译时擦除了所有类型相关的信息，所以在运行时不存在任何类型相关的信息
- **类型擦除**： 通过类型参数合并，将泛型类型实例关联到同一份字节码上；关键在于从泛型类型中清除类型参数的相关信息，并且在必要的时候添加类型检查和类型转换的方法

- **泛型中的限定通配符和非限定通配符**： 泛型类型必须用限定内的类型来进行初始化，否则会导致编译错误

  - **限定通配符**： 对类型进行了限制
    - `<? extends T>`： 通过确保类型必须是 T 的子类来设定类型的上界
    - `<? super T>`： 通过确保类型必须是 T 的父类来设定类型的下界

  - **非限定通配符 `<?>`**： <?>可以用任意类型来替代

- **`List<? extends T>和List <? super T>`的区别**： 
  - `List<? extends T>` 可以接受任何继承自 T 类型的List
  - `List<? super T>` 可以接受任何 T 的父类构成的List

- **编写一个能接受泛型参数并返回泛型类型的泛型方法**： 

  ```java
   public V put(K key, V value) {
        return cache.put(key, value);
  }
  ```

- **使用泛型编写带有参数的类**： 



- **编写一段泛型程序来实现 LRU 缓存**： 

  **提示**： 

  - LinkedHashMap 可以用来实现固定大小的LRU缓存，当LRU缓存已满时，会把最老的键值对移出缓存
  - LinkedHashMap 提供了一个称为 removeEldestEntry() 的方法，该方法会被put()和putAll()调用来删除最老的键值对

- **不能把 `List<String>` 传递给一个接受 `List<Object>` 参数的方法**：

  因为 `List<Object>` 可以存储任何类型的对象，而 `List<String>` 只能存储String

  ```java
  List<Object> objectList;
  List<String> stringList;
  objectList = stringList;  //compilation error incompatible types
  ```

- **Array 不支持泛型**：建议使用 List 来代替 Array，因为 List 可以提供编译期的类型安全保证，Array 不能

- **如果把泛型和原始类型混合起来使用， Java 5的 javac 编译器会产生类型未检查警告**： 

  如代码： ` List<String> rawList = new ArrayList()` 

​      

# 十、注解 

- **[JAVA自定义注解、元注解介绍及自定义注解使用场景](https://blog.csdn.net/bluuusea/article/details/79996572)** 
- **[注解 Annotation 实现原理与自定义注解例子](https://www.cnblogs.com/acm-bingzi/p/javaAnnotation.html)** 

1. JDK提供的常用的三个注解

   @Override: 限定重写父类方法, 该注释只能用于方法
   @Deprecated: 用于表示某个程序元素(类, 方法等)已过时
   @SuppressWarnings: 抑制编译器警告

2. 如何自定义注解

   以SuppressWarnings为例进行创建即可

3. 元注解：可以对已有的注解进行解释说明

   @Retention
   @Target
   @Documented
   @Inherited

---

Java 注解是附加在代码中的一些元信息，用于一些工具在编译、运行时进行解析和使用，起到说明、配置的功能。注解不会也不能影响代码的实际逻辑，仅仅起到辅助性的作用

# 十一、枚举类

## 1. 自定义枚举类

```java
public class TestSeason {
	public static void main(String[] args) {
		Season spring = Season.SPRING;
		System.out.println(spring);
		spring.show();
		System.out.println(spring.getSeasonName());
	}
}
//枚举类
class Season{
	//1.提供类的属性，声明为private final 
	private final String seasonName;
	private final String seasonDesc;
	//2.声明为final的属性，在构造器中初始化。
	private Season(String seasonName,String seasonDesc){
		this.seasonName = seasonName;
		this.seasonDesc = seasonDesc;
	}
	//3.通过公共的方法来调用属性
	public String getSeasonName() {
		return seasonName;
	}
	public String getSeasonDesc() {
		return seasonDesc;
	}
	//4.创建枚举类的对象:将类的对象声明public static final
	public static final Season SPRING = new Season("spring", "春暖花开");
	public static final Season SUMMER = new Season("summer", "夏日炎炎");
	public static final Season AUTUMN = new Season("autumn", "秋高气爽");
	public static final Season WINTER = new Season("winter", "白雪皑皑");
	@Override
	public String toString() {
		return "Season [seasonName=" + seasonName + ", seasonDesc="
				+ seasonDesc + "]";
	}
	public void show(){
		System.out.println("这是一个季节");
	}
}
```

## 2.  enum 关键字用于定义枚举类

```java
public class TestSeason1 {
	public static void main(String[] args) {
		Season1 spring = Season1.SPRING;
		System.out.println(spring);
		spring.show();
		System.out.println(spring.getSeasonName());
		
		System.out.println();
		//1.values()
		Season1[] seasons = Season1.values();
		for(int i = 0;i < seasons.length;i++){
			System.out.println(seasons[i]);
		}
		//2.valueOf(String name):要求传入的形参name是枚举类对象的名字。
		//否则，报java.lang.IllegalArgumentException异常
		String str = "WINTER";
		Season1 sea = Season1.valueOf(str);
		System.out.println(sea);
		System.out.println();
		
		Thread.State[] states = Thread.State.values();
		for(int i = 0;i < states.length;i++){
			System.out.println(states[i]);
		}
		sea.show();
	}
}
interface Info{
	void show();
}
//枚举类
enum Season1 implements Info{
	SPRING("spring", "春暖花开"){
		public void show(){
			System.out.println("春天在哪里？");
		}
	},
	SUMMER("summer", "夏日炎炎"){
		public void show(){
			System.out.println("生如夏花");
		}
	},
	AUTUMN("autumn", "秋高气爽"){
		public void show(){
			System.out.println("秋天是用来分手的季节");
		}
	},
	WINTER("winter", "白雪皑皑"){
		public void show(){
			System.out.println("冬天里的一把火");
		}
	};
	
	private final String seasonName;
	private final String seasonDesc;
	
	private Season1(String seasonName,String seasonDesc){
		this.seasonName = seasonName;
		this.seasonDesc = seasonDesc;
	}
	public String getSeasonName() {
		return seasonName;
	}
	public String getSeasonDesc() {
		return seasonDesc;
	}
	
	@Override
	public String toString() {
		return "Season [seasonName=" + seasonName + ", seasonDesc="
				+ seasonDesc + "]";
	}
}
```

##3. 枚举类详解

1. 枚举类的属性:
   - 枚举类对象的属性不应允许被改动, 所以应该使用 private final 修饰
     - 枚举类的使用 private final 修饰的属性应该在构造器中为其赋值
     - 若枚举类显式的定义了带参数的构造器, 则在列出枚举值时也必须对应的传入参数
   - 必须在枚举类的第一行声明枚举类对象。
   - 枚举类和普通类的区别：
     - 使用 enum 定义的枚举类默认继承了 java.lang.Enum 类
     - 枚举类的构造器只能使用 private 访问控制符
     - 枚举类的所有实例必须在枚举类中显式列出(, 分隔    ; 结尾). 列出的实例系统会自动添加 public static final 修饰
   - JDK 1.5 中可以在 switch 表达式中使用Enum定义的枚举类的对象作为表达式, case 子句可以直接使用枚举值的名字, 无需添加枚举类作为限定
2. 枚举类的主要方法：
   - values()方法：返回枚举类型的对象数组。该方法可以很方便地遍历所有的枚举值。
   - valueOf(String str)：可以把一个字符串转为对应的枚举类对象。要求字符串必须是枚举类对象的“名字”。如不是，会有运行时异常。
3. 实现接口的枚举类:
   - 若需要每个枚举值在调用实现的接口方法呈现出不同的行为方式, 则可以让每个枚举值分别来实现该方法

# 十二、网络编程

## 1. 网络编程概述

![](../pics/internet2.png)

![](../pics/internet.png)



## 2. InetAddress类

```java
public class TestInetAddress {
	public static void main(String[] args) throws Exception {
		//创建一个InetAddress对象：getByName()
		InetAddress inet = InetAddress.getByName("www.baidu.com");
		//inet = InetAddress.getByName("111.13.100.92");
		System.out.println(inet);
		//两个方法
		System.out.println(inet.getHostName()); //获取 IP 地址对应的域名
		System.out.println(inet.getHostAddress()); //获取 IP 地址
		//获取本机的IP:getLocalHost()
		InetAddress inet1 = InetAddress.getLocalHost();
		System.out.println(inet1);
		System.out.println(inet1.getHostName());
		System.out.println(inet1.getHostAddress());
	}
}
```

## 3. TCP网络通信(传输控制协议)

###1. TCP 简介

- 使用TCP协议前，须先建立TCP连接，形成传输数据通道
- 传输前，采用“三次握手”方式，是可靠的
- TCP协议进行通信的两个应用进程：客户端、服务端
- 在连接中可进行大数据量的传输
- 传输完毕，需释放已建立的连接，效率低

**IP(Internet Protocol)协议** 是网络层的主要协议，支持网间互连的数据通信

> TCP/IP协议模型从更实用的角度出发，形成了高效的四层体系结构，即物理链路层、IP层、传输层和应用层

### 2. Socket

1. 客户端Socket的工作过程： 
   - 创建 Socket：根据指定服务端的 IP 地址或端口号构造 Socket 类对象。若服务器端响应，则建立客户端到服务器的通信线路。若连接失败，会出现异常
   - 打开连接到 Socket 的输入/出流： 使用 getInputStream()方法获得输入流，使用 getOutputStream()方法获得输出流，进行数据传输
   - 按照一定的协议对 Socket  进行读/写操作：通过输入流读取服务器放入线路的信息（但不能读取自己放入线路的信息），通过输出流将信息写入线程
   - 关闭 Socket：断开客户端到服务器的连接，释放线路 
2. 服务器程序的工作过程：
   - 调用 ServerSocket(int port) ：创建一个服务器端套接字，并绑定到指定端口上。用于监听客户端的请求
   - 调用 accept()：监听连接请求，如果客户端请求连接，则接受连接，返回通信套接字对象
   - 调用 该Socket类对象的 getOutputStream() 和 getInputStream ()：获取输出流和输入流，开始网络数据的发送和接收
   - 关闭ServerSocket和Socket对象：客户端访问结束，关闭通信套接字。

### 3. Socket实现 TCP 实例

**1. TCP编程例一：客户端给服务端发送信息。服务端输出此信息到控制台上：** 

```java
public class TestTCP1 {
	// 客户端
	@Test
	public void client() {
		Socket socket = null;
		OutputStream os = null;
		try {
			// 1.创建一个Socket的对象，通过构造器指明服务端的IP地址，以及其接收程序的端口号
			socket = new Socket(InetAddress.getByName("127.0.0.1"), 9090);
			// 2.getOutputStream()：发送数据，方法返回OutputStream的对象
			os = socket.getOutputStream();
			// 3.具体的输出过程
			os.write("我是客户端，请多关照".getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			// 4.关闭相应的流和Socket对象
			if (os != null) {
				try {
					os.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			if (socket != null) {
				try {
					socket.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	// 服务端
	@Test
	public void server() {
		ServerSocket ss = null;
		Socket s = null;
		InputStream is = null;
		try {
			// 1.创建一个ServerSocket的对象，通过构造器指明自身的端口号
			ss = new ServerSocket(9090);
			// 2.调用其accept()方法，返回一个Socket的对象
			s = ss.accept();
			// 3.调用Socket对象的getInputStream()获取一个从客户端发送过来的输入流
			is = s.getInputStream();
			// 4.对获取的输入流进行的操作
			byte[] b = new byte[20];
			int len;
			while ((len = is.read(b)) != -1) {
				String str = new String(b, 0, len);
				System.out.print(str);
			}
			System.out.println("收到来自于" + s.getInetAddress().getHostAddress() + "的连接");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			// 5.关闭相应的流以及Socket、ServerSocket的对象
			if (is != null) {
				try {
					is.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (s != null) {
				try {
					s.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (ss != null) {
				try {
					ss.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
```

**2. TCP编程例二：客户端给服务端发送信息，服务端将信息打印到控制台上，同时发送“已收到信息”给客户端** 

```java
public class TestTCP2 {
	//客户端
	@Test
	public void client(){
		Socket socket = null;
		OutputStream os = null;
		InputStream is = null;
		try {
			socket = new Socket(InetAddress.getByName("127.0.0.1"),8989);
			os = socket.getOutputStream();
			os.write("我是客户端".getBytes());
			//shutdownOutput():执行此方法，显式的告诉服务端发送完毕！
			socket.shutdownOutput();
			is = socket.getInputStream();
			byte[] b = new byte[20];
			int len;
			while((len = is.read(b)) != -1){
				String str = new String(b,0,len);
				System.out.print(str);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(is != null){
				try {
					is.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if(os != null){
				try {
					os.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if(socket != null){
				try {
					socket.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
  
	//服务端
	@Test
	public void server(){
		ServerSocket ss = null;
		Socket s = null;
		InputStream is = null;
		OutputStream os = null;
		try {
			ss = new ServerSocket(8989);
			s = ss.accept();
			is = s.getInputStream();
			byte[] b = new byte[20];
			int len;
			while((len = is.read(b)) != -1){
				String str = new String(b,0,len);
				System.out.print(str);
			}
			os = s.getOutputStream();
			os.write("我已收到你的情意".getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(os != null){
				try {
					os.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if(is != null){
				try {
					is.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if(s != null){
				try {
					s.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if(ss != null){
				try {
					ss.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
```

**3. TCP编程例三：从客户端发送文件给服务端，服务端保存到本地并返回“发送成功”给客户端，关闭相应的连接** 

```java
public class TestTCP3 {
	@Test
	public void client()throws Exception{
		//1.创建Socket的对象
		Socket socket = new Socket(InetAddress.getByName("127.0.0.1"), 9898);
		//2.从本地获取一个文件发送给服务端
		OutputStream os = socket.getOutputStream();
		FileInputStream fis = new FileInputStream(new File("1.jpg"));
		byte[] b = new byte[1024];
		int len;
		while((len = fis.read(b)) != -1){
			os.write(b,0,len);
		}
		socket.shutdownOutput();
		//3.接收来自于服务端的信息
		InputStream is = socket.getInputStream();
		byte[] b1 = new byte[1024];
		int len1;
		while((len1 = is.read(b1)) != -1){
			String str = new String(b1,0,len1);
			System.out.print(str);
		}
		//4.关闭相应的流和Socket对象
		is.close();
		os.close();
		fis.close();
		socket.close();
	}
	@Test
	public void server() throws Exception{
		//1.创建一个ServerSocket的对象
		ServerSocket ss = new ServerSocket(9898);
		//2.调用其accept()方法，返回一个Socket的对象
		Socket s = ss.accept();
		//3.将从客户端发送来的信息保存到本地
		InputStream is = s.getInputStream();
		FileOutputStream fos = new FileOutputStream(new File("3.jpg"));
		byte[] b = new byte[1024];
		int len;
		while((len = is.read(b)) != -1){
			fos.write(b, 0, len);
		}
		System.out.println("收到来自于" + s.getInetAddress().getHostAddress() + "的文件");
		//4.发送"接收成功"的信息反馈给客户端
		OutputStream os = s.getOutputStream();
		os.write("你发送的图片我已接收成功！".getBytes());
		//5.关闭相应的流和Socket及ServerSocket的对象
		os.close();
		fos.close();
		is.close();
		s.close();
		ss.close();
	}
}
```

## 5. UDP网络通信(用户数据报协议)

###1. UDP 简介

- 将数据、源、目的封装成数据包，不需要建立连接
- 每个数据报的大小限制在64K内
- 因无需连接，故是不可靠的
- 发送数据结束时无需释放资源，速度快

### 2. 编程实例

```java
public class TestUDP {
	// 发送端
	@Test
	public void send() {
		DatagramSocket ds = null;
		try {
			ds = new DatagramSocket();
			byte[] b = "你好，我是要发送的数据".getBytes();
			//创建一个数据报：每一个数据报不能大于64k，都记录着数据信息，发送端的IP、端口号,
            //以及要发送到的接收端的IP、端口号
			DatagramPacket pack = new DatagramPacket(b, 0, b.length,
					InetAddress.getByName("127.0.0.1"), 9090);
			ds.send(pack);
		}catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(ds != null){
				ds.close();
			}
		}
	}

	// 接收端
	@Test
	public void rceive() {
		DatagramSocket ds = null;
		try {
			ds = new DatagramSocket(9090);
			byte[] b = new byte[1024];
			DatagramPacket pack = new DatagramPacket(b, 0, b.length);
			ds.receive(pack);
			String str = new String(pack.getData(), 0, pack.getLength());
			System.out.println(str);
		}catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(ds != null){
				ds.close();
			}
		}
	}
}
```

## 6. URL编程

###1. URL 简介

- URL(Uniform Resource Locator)：统一资源定位符，它表示 Internet 上某一资源的地址。通过 URL 我们可以访问 Internet 上的各种网络资源，比如最常见的 www，ftp 站点。浏览器通过解析给定的 URL 可以在网络上查找相应的文件或其他资源。  

- URL的基本结构由5部分组成：<传输协议>://<主机名>:<端口号>/<文件名>例如: `http://192.168.1.100:8080/helloworld/index.jsp`

- URL 类的方法：

  ```java
  public String getProtocol(  )     获取该URL的协议名
  public String getHost(  )           获取该URL的主机名
  public String getPort(  )            获取该URL的端口号
  public String getPath(  )           获取该URL的文件路径
  public String getFile(  )             获取该URL的文件名
  public String getRef(  )             获取该URL在文件中的相对位置
  public String getQuery(   )        获取该URL的查询名
  ```

### 2. 针对HTTP协议的URLConnection类

- URL的方法 openStream()：能从网络上读取数据
- 若希望输出数据，例如向服务器端的 CGI （公共网关接口-Common Gateway Interface-的简称，是用户浏览器和服务器端的应用程序进行连接的接口）程序发送一些数据，则必须先与URL建立连接，然后才能对其进行读写，此时需要使用 URLConnection 。
- URLConnection：表示到URL所引用的远程对象的连接。当与一个URL建立连接时，首先要在一个 URL 对象上通过方法 openConnection() 生成对应的 URLConnection 对象。如果连接过程失败，将产生IOException. 
  - URL netchinaren = new URL ("http://www.baidu.com/index.shtml"); 
  - URLConnectonn u = netchinaren.openConnection( ); 
- 通过URLConnection对象获取的输入流和输出流，即可以与现有的CGI程序进行交互

```java
public class TestURL {
	public static void main(String[] args) throws Exception {
		//1.创建一个URL的对象
		URL url = new URL("http://127.0.0.1:8080/examples/HelloWorld.txt?a=b");//File file = new File("文件的路径");

		//如何将服务端的资源读取进来:openStream()
		InputStream is = url.openStream();
		byte[] b = new byte[20];
		int len;
		while((len = is.read(b)) != -1){
			String str = new String(b,0,len);
			System.out.print(str);
		}
		is.close();
		//如果既有数据的输入，又有数据的输出，则考虑使用URLConnection
		URLConnection urlConn = url.openConnection();
		InputStream is1 = urlConn.getInputStream();
		FileOutputStream fos = new FileOutputStream(new File("abc.txt"));
		byte[] b1 = new byte[20];
		int len1;
		while((len1 = is1.read(b1)) != -1){
			fos.write(b1, 0, len1);
		}
		fos.close();
		is1.close();
	}
}
```

# 十三、特性

## Java 各版本的新特性

**New highlights in Java SE 8** 

1. Lambda Expressions
2. Pipelines and Streams
3. Date and Time API
4. Default Methods
5. Type Annotations
6. Nashhorn JavaScript Engine
7. Concurrent Accumulators
8. Parallel operations
9. PermGen Error Removed

**New highlights in Java SE 7** 

1. Strings in Switch Statement
2. Type Inference for Generic Instance Creation
3. Multiple Exception Handling
4. Support for Dynamic Languages
5. Try with Resources
6. Java nio Package
7. Binary Literals, Underscore in literals
8. Diamond Syntax

- [Difference between Java 1.8 and Java 1.7?](http://www.selfgrowth.com/articles/difference-between-java-18-and-java-17)
- [Java 8 特性](http://www.importnew.com/19345.html)

## Java 与 C++ 的区别

- Java 是纯粹的面向对象语言，所有的对象都继承自 java.lang.Object，C++ 为了兼容 C 即支持面向对象也支持面向过程。
- Java 通过虚拟机从而实现跨平台特性，但是 C++ 依赖于特定的平台。
- Java 没有指针，它的引用可以理解为安全指针，而 C++ 具有和 C 一样的指针。
- Java 支持自动垃圾回收，而 C++ 需要手动回收。
- Java 不支持多重继承，只能通过实现多个接口来达到相同目的，而 C++ 支持多重继承。
- Java 不支持操作符重载，虽然可以对两个 String 对象执行加法运算，但是这是语言内置支持的操作，不属于操作符重载，而 C++ 可以。
- Java 的 goto 是保留字，但是不可用，C++ 可以使用 goto。
- Java 不支持条件编译，C++ 通过 #ifdef #ifndef 等预处理命令从而实现条件编译。

[What are the main differences between Java and C++?](http://cs-fundamentals.com/tech-interview/java/differences-between-java-and-cpp.php)

## JRE or JDK

- JRE is the JVM program, Java application need to run on JRE.
- JDK is a superset of JRE, JRE + tools for developing java programs. e.g, it provides the compiler "javac"

# 十四. Java8

## 1. Lambda 表达式

### 1. Lambda 表达式

- Lambda 是一个**匿名函数**，可以将 Lambda 分为两部分：

  - 左侧：指定了 Lambda 表达式需要的所有参数
  - 右侧：指定了 Lambda 体，即 Lambda 表达式要执行的功能

- Lambda 表达式需要**函数式接口**支持

  - **函数式接口**：只有**一个抽象方法的接口**，可以使用注解 `@FunctionalInterface` 检查是否是函数式接口

    ```java
    //声明函数式接口
    @FunctionalInterface
    public interface MyFun {
    	public Integer getValue(Integer num);
    }
    
    //对一个数进行运算
    @Test
    public void test(){
        Integer num = operation(100, (x) -> x * x);
        System.out.println(num);
    
        System.out.println(operation(200, (y) -> y + 200));
    }
    
    public Integer operation(Integer num, MyFun mf){
        return mf.getValue(num);
    }
    ```

- **语法格式**：

  - **语法格式一**：无参数，无返回值：`() -> System.out.println("Hello Lambda!");`

    ```java
    @Test
    public void test(){
        int num = 0;
    
        Runnable r = new Runnable() {
            @Override
            public void run() {
                System.out.println("Hello World!" + num);
            }
        };
        r.run();
    
        System.out.println("-------------------------------");
    
     	Runnable r1 = () -> System.out.println("Hello Lambda!");
        r1.run();
    }
    ```

  - **语法格式二**：有一个参数，并且无返回值：`(x) -> System.out.println(x)`

    ```java
    @Test
    public void test(){
        Consumer<String> con = (x) -> System.out.println(x);
        con.accept("hello");
    }
    ```

  - **语法格式三**：若只有一个参数，小括号可以省略不写：`x -> System.out.println(x)`

    ```java
    @Test
    public void test2(){
        Consumer<String> con = x -> System.out.println(x);
        con.accept("hello");
    }
    ```

  - **语法格式四**：有两个以上的参数，有返回值，并且 Lambda 体中有多条语句

    ```java
    Comparator<Integer> com = (x, y) -> {
    	System.out.println("函数式接口");
    	return Integer.compare(x, y);
     };
    ```

    - **语法格式五**：若 Lambda 体中只有一条语句， return 和 大括号都可以省略不写：

      ```java
      Comparator<Integer> com = (x, y) -> Integer.compare(x, y);
      ```

    - **语法格式六**：Lambda 表达式的参数列表的数据类型可以省略不写，因为JVM编译器通过上下文推断出，数据类型，即“类型推断”：`(Integer x, Integer y) -> Integer.compare(x, y);`

### 2. 类型推断

- **类型推断**： Lambda 表达式中无需指定类型，由编译器推断出来

### 3. 新特性体验

**需求：**获取公司中年龄小于 35 的员工信息

**对象封装类：** 

```java
public class Employee {

	private int id;
	private String name;
	private int age;
	private double salary;

	public Employee() {
	}

	public Employee(String name) {
		this.name = name;
	}

	public Employee(String name, int age) {
		this.name = name;
		this.age = age;
	}

	public Employee(int id, String name, int age, double salary) {
		this.id = id;
		this.name = name;
		this.age = age;
		this.salary = salary;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getAge() {
		return age;
	}

	public void setAge(int age) {
		this.age = age;
	}

	public double getSalary() {
		return salary;
	}

	public void setSalary(double salary) {
		this.salary = salary;
	}

	public String show() {
		return "测试方法引用！";
	}

	@Override
	public String toString() {
		return "Employee [id=" + id + ", name=" + name + ", age=" + age + ", salary=" + salary + "]";
	}
}
```

**存储的数据：** 

```java
List<Employee> emps = Arrays.asList(
        new Employee(101, "张三", 18, 9999.99),
        new Employee(102, "李四", 59, 6666.66),
        new Employee(103, "王五", 28, 3333.33),
        new Employee(104, "赵六", 8, 7777.77),
        new Employee(105, "田七", 38, 5555.55)
);
```

**原始的方法：** 

```java
public List<Employee> filterEmployeeAge(List<Employee> emps){
    List<Employee> list = new ArrayList<>();
    for (Employee emp : emps) {
        if(emp.getAge() <= 35){
            list.add(emp);
        }
    }
    return list;
}

```

**优化方式一：** 策略设计模式（可以通过修改接口中的实现，来修改功能，提高代码共用行）

```java
public List<Employee> filterEmployee(List<Employee> emps, MyPredicate<Employee> mp){
    List<Employee> list = new ArrayList<>();
    for (Employee employee : emps) {
        if(mp.test(employee)){
            list.add(employee);
        }
    }
    return list;
}

//设计模式的接口
@FunctionalInterface
public interface MyPredicate<T> {
	public boolean test(T t);
}

//接口的实现
public class FilterEmployeeForAge implements MyPredicate<Employee>{
	@Override
	public boolean test(Employee t) {
		return t.getAge() <= 35;
	}
}

```

**优化方式二：** 匿名内部类

```java
@Test
public void test5(){
    List<Employee> list = filterEmployee(emps, new MyPredicate<Employee>() {
        @Override
        public boolean test(Employee t) {
            return t.getId() <= 103;
        }
    });
    for (Employee employee : list) {
        System.out.println(employee);
    }
}

```

**优化方式三：** Lambda 表达式

```java
@Test
public void test6(){
    List<Employee> list = filterEmployee(emps, (e) -> e.getAge() <= 35);
    list.forEach(System.out::println);
}

```

**优化方式四：** Stream API

```java
@Test
public void test7(){
    emps.stream()
        .filter((e) -> e.getAge() <= 35)
        .forEach(System.out::println);
}

```

## 2. 函数式接口

### 1. 什么是函数式接口

- **函数式接口**：只包含一个抽象方法的接口
- `@FunctionalInterface` 注解用于检查是否是一个函数式接口，同时也向 javadoc 声明为函数式接口

### 2. Java 内置四大核心函数式接口

1. **消费型接口：** `void Consumer<T>`，对类型为T的对象应用操作，包含方法：`void accept(T t)` 

   ```java
   @Test
   public void test(){
       happy(10000, (m) -> System.out.println("每次消费：" + m + "元"));
   } 
   
   public void happy(double money, Consumer<Double> con){
       con.accept(money);
   }
   
   ```

2. **供给型接口：** `T Supplier<T>`，返回类型为T的对象，包含方法：`T get()`

   ```java
   @Test
   public void test(){
       List<Integer> numList = getNumList(10, () -> (int)(Math.random() * 100));
       for (Integer num : numList) {
           System.out.println(num);
       }
   }
   
   //需求：产生指定个数的整数，并放入集合中
   public List<Integer> getNumList(int num, Supplier<Integer> sup){
       List<Integer> list = new ArrayList<>();
       for (int i = 0; i < num; i++) {
           Integer n = sup.get();
           list.add(n);
       }
       return list;
   }
   
   ```

3. **函数型接口：** `R Function<T, R>`，对类型为T的对象应用操作，并返回结果，结果是R类型的对象，包含方法：`R apply(T t)`

   ```java
   @Test
   public void test3(){
       String newStr = strHandler("\t\t\t hello   ", (str) -> str.trim());
       System.out.println(newStr);
   
       String subStr = strHandler("hello", (str) -> str.substring(2, 5));
       System.out.println(subStr);
   }
   
   //需求：用于处理字符串
   public String strHandler(String str, Function<String, String> fun){
       return fun.apply(str);
   }
   
   ```

4. **断定型接口：** `boolean Predicate<T>`，确定类型为 T 的对象是否满足某约束，并返回 boolean 值，包含方法`boolean test(T t)`

   ```java
   @Test
   public void test4(){
       List<String> list = Arrays.asList("Hello", "atguigu", "Lambda", "www", "ok");
       List<String> strList = filterStr(list, (s) -> s.length() > 3);
       for (String str : strList) {
           System.out.println(str);
       }
   }
   
   //需求：将满足条件的字符串，放入集合中
   public List<String> filterStr(List<String> list, Predicate<String> pre){
       List<String> strList = new ArrayList<>();
       for (String str : list) {
           if(pre.test(str)){
               strList.add(str);
           }
       }
       return strList;
   }
   
   ```

### 3. 其他接口

![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8.png)

## 3. 方法引用与构造器引用

### 1. 方法引用

- **方法引用：** 使用操作符 `::`  将方法名和对象或类的名字分隔开来

  ```java
  @Test
  public void test1(){
      PrintStream ps = System.out;
      Consumer<String> con = (str) -> ps.println(str);
      con.accept("Hello World！");
  
      System.out.println("--------------------------------");
  
      Consumer<String> con2 = ps::println;
      con2.accept("Hello Java8！");
  
      Consumer<String> con3 = System.out::println;
  }
  
  ```

- **三种主要使用情况**：

  - `对象::实例方法`

    ```java
    @Test
    public void test2(){
        Employee emp = new Employee(101, "张三", 18, 9999.99);
    
        Supplier<String> sup = () -> emp.getName();
        System.out.println(sup.get());
    
        System.out.println("----------------------------------");
    
        Supplier<String> sup2 = emp::getName;
        System.out.println(sup2.get());
    }
    
    ```

  - `类::静态方法`

    ```java
    @Test
    public void test3(){
        BiFunction<Double, Double, Double> fun = (x, y) -> Math.max(x, y);
        System.out.println(fun.apply(1.5, 22.2));
    
        System.out.println("--------------------------------------------------");
    
        BiFunction<Double, Double, Double> fun2 = Math::max;
        System.out.println(fun2.apply(1.2, 1.5));
    }
    
    @Test
    public void test4(){
        Comparator<Integer> com = (x, y) -> Integer.compare(x, y);
    
        System.out.println("-------------------------------------");
    
        Comparator<Integer> com2 = Integer::compare;
    }
    
    ```

  - `类::实例方法`

    ```java
    @Test
    public void test5(){
        BiPredicate<String, String> bp = (x, y) -> x.equals(y);
        System.out.println(bp.test("abcde", "abcde"));
    
        System.out.println("-----------------------------------------");
        BiPredicate<String, String> bp2 = String::equals;
        System.out.println(bp2.test("abc", "abc"));
    
        System.out.println("-----------------------------------------");
        Function<Employee, String> fun = (e) -> e.show();
        System.out.println(fun.apply(new Employee()));
    
        System.out.println("-----------------------------------------");
        Function<Employee, String> fun2 = Employee::show;
        System.out.println(fun2.apply(new Employee()));
    }
    
    ```

**注意：**

- 方法引用所引用的方法的参数列表与返回值类型需要与函数式接口中抽象方法的参数列表和返回值类型一致
- 若 Lambda 的参数列表的第一个参数是实例方法的调用者，第二个参数(或无参)是实例方法的参数时，格式： ClassName::MethodName

### 2. 构造器引用

> 格式：`ClassName::new` 

与函数式接口相结合，自动与函数式接口中方法兼容

可以把构造器引用赋值给定义的方法，与构造器参数列表要与接口中抽象方法的参数列表一致

```java
@Test
public void test6(){
    Supplier<Employee> sup = () -> new Employee();
    System.out.println(sup.get());

    System.out.println("------------------------------------");

    Supplier<Employee> sup2 = Employee::new;
    System.out.println(sup2.get());
}

@Test
public void test7(){
    Function<String, Employee> fun = Employee::new;
    BiFunction<String, Integer, Employee> fun2 = Employee::new;
}

```

### 3. 数组引用

> 格式：`type[] :: new`

```java
@Test
public void test8(){
    Function<Integer, String[]> fun = (args) -> new String[args];
    String[] strs = fun.apply(10);
    System.out.println(strs.length);

    System.out.println("--------------------------");

    Function<Integer, Employee[]> fun2 = Employee[] :: new;
    Employee[] emps = fun2.apply(20);
    System.out.println(emps.length);
}

```

## 4. Stream API

### 1. 什么是Stream

流(Stream)是数据渠道，用于操作数据源（集合、数组等）所生成的元素序列

> 集合讲的是数据，流讲的是计算

**注意：** 

- Stream 自己不会存储元素
- Stream 不会改变源对象，会返回一个持有结果的新 Stream
- Stream 操作是延迟执行的，因此会等到需要结果时才执行

### 2. Stream 的操作三个步骤

- **创建Stream**：一个数据源（如：集合、数组），获取一个流

- **中间操作**：一个中间操作链，对数据源的数据进行处理

- **终止操作(终端操作)**：一个终止操作，执行中间操作链，并产生结果

![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8_2.png)

### 3. 创建Stream

- **获取流**：

  - `default Stream<E> stream() `：返回一个顺序流

  - `default Stream<E> parallelStream()`：返回一个并行流

    ```java
    List<String> list = new ArrayList<>();
    Stream<String> stream = list.stream(); //获取一个顺序流
    Stream<String> parallelStream = list.parallelStream(); //获取一个并行流
    
    ```

- **创建流**： 

  - **由数组创建流**： `static <T> Stream<T> stream(T[] array)` 

    重载方法： 

    - public static IntStream stream(int[] array)
    - public static LongStream stream(long[] array)
    - public static DoubleStream stream(double[] array)

    ```java
    Integer[] nums = new Integer[10];
    Stream<Integer> stream1 = Arrays.stream(nums);
    
    ```

  - **由值创建流**：使用静态方法 `Stream.of()` 显示值创建一个流，可以接收任意数量的参数

    `public static<T> Stream<T> of(T... values) `

    ```java
    Stream<Integer> stream2 = Stream.of(1,2,3,4,5,6);
    
    ```

  - **由函数创建流**：创建无限流，使用静态方法 `Stream.iterate()` 和 `Stream.generate()`

    - 迭代：`public static<T> Stream<T> iterate(final T seed, final UnaryOperator<T> f)`
    - 生成：`public static<T> Stream<T> generate(Supplier<T> s)` 

    ```java
    //迭代
    Stream<Integer> stream3 = Stream.iterate(0, (x) -> x + 2).limit(10);
    stream3.forEach(System.out::println);
    
    //生成
    Stream<Double> stream4 = Stream.generate(Math::random).limit(2);
    stream4.forEach(System.out::println);
    
    ```

### 4. Stream 的中间操作

- **惰性求值：** 终止操作时一次性全部处理

- **筛选与切片**

  - `filter(Predicatep)` ：接收 Lambda ，从流中排除某些元素

    ```java
    @Test
    public void test2(){
        //所有的中间操作不会做任何的处理
        Stream<Employee> stream = emps.stream()
            .filter((e) -> {
                System.out.println("测试中间操作");
                return e.getAge() <= 35;
            });
        //只有当做终止操作时，所有的中间操作会一次性的全部执行，称为“惰性求值”
        stream.forEach(System.out::println);
    }
    
    ```

  - `distinct()` ： 筛选，通过流所生成元素的 hashCode() 和equals() 去除重复元素

    ```java
    @Test
    public void test6(){
        emps.stream()
            .distinct()
            .forEach(System.out::println);
    }
    
    ```

    要想 distinct() 起作用，必须实现 hashCode() 和equals()  方法

    ```java
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + age;
        result = prime * result + id;
        result = prime * result + ((name == null) ? 0 : name.hashCode());
        long temp;
        temp = Double.doubleToLongBits(salary);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Employee other = (Employee) obj;
        if (age != other.age)
            return false;
        if (id != other.id)
            return false;
        if (name == null) {
            if (other.name != null)
                return false;
        } else if (!name.equals(other.name))
            return false;
        if (Double.doubleToLongBits(salary) != Double.doubleToLongBits(other.salary))
            return false;
        return true;
    }
    
    ```

  - `limit(long maxSize)` ： 截断流，使其元素不超过给定数量

    ```java
    @Test
    public void test4(){
        emps.stream()
            .filter((e) -> {
                System.out.println("短路！"); // &&  ||
                return e.getSalary() >= 5000;
            }).limit(3)
            .forEach(System.out::println);
    }
    
    ```

  - `skip(long n)` ：跳过元素，返回一个扔掉了前 n 个元素的流，若流中元素不足，则返回一个空流

    ```java
    @Test
    public void test5(){
        emps.parallelStream()
            .filter((e) -> e.getSalary() >= 5000)
            .skip(2)
            .forEach(System.out::println);
    }
    
    ```

- **映射**： 

  - `map(Function f)` ：接收一个函数作为参数，该函数会被应用到每个元素上，并将其映射成新的元素
  - `mapToDouble(ToDoubleFunction f)` ： 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的DoubleStream
  - `mapToInt(ToIntFunction f)` ： 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的IntStream
  - `mapToLong(ToLongFunction f)` ： 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的LongStream
  - `flatMap(Function f)` ： 接收一个函数作为参数，将流中的每个值都换成另一个流，然后把所有流连接成一个流

  ```java
  @Test
  public void test1(){
      Stream<String> str = emps.stream()
          					 .map((e) -> e.getName());
  
      System.out.println("-------------------------------------------");
  
      List<String> strList = Arrays.asList("aaa", "bbb", "ccc", "ddd", "eee");
  
      Stream<String> stream = strList.stream()
             .map(String::toUpperCase);
      stream.forEach(System.out::println);
  
      Stream<Stream<Character>> stream2 = strList.stream()
             .map(TestStream::filterCharacter);
  
      stream2.forEach((sm) -> {
          sm.forEach(System.out::println);
      });
  
      System.out.println("---------------------------------------------");
  
      Stream<Character> stream3 = strList.stream()
             .flatMap(TestStream::filterCharacter);
  
      stream3.forEach(System.out::println);
  }
  
  public static Stream<Character> filterCharacter(String str){
      List<Character> list = new ArrayList<>();
      for (Character ch : str.toCharArray()) {
          list.add(ch);
      }
      return list.stream();
  }
  
  ```

- **排序**： 

  - `sorted()` ：产生一个新流，其中按自然顺序排序
  - `sorted(Comparator comp)` ： 产生一个新流，其中按比较器顺序排序

  ```java
  @Test
  public void test2(){
     emps.stream()
         .map(Employee::getName)
         .sorted()
         .forEach(System.out::println);
  
     System.out.println("------------------------------------");
  
     emps.stream()
         .sorted((x, y) -> {
             if(x.getAge() == y.getAge()){
                 return x.getName().compareTo(y.getName());
             }else{
                 return Integer.compare(x.getAge(), y.getAge());
             }
         }).forEach(System.out::println);
  }  
  
  ```

### 5. Stream 的终止操作

**使用的数据源：** 

```java
List<Employee> emps = Arrays.asList(
        new Employee(102, "李四", 59, 6666.66, Status.BUSY),
        new Employee(101, "张三", 18, 9999.99, Status.FREE),
        new Employee(103, "王五", 28, 3333.33, Status.VOCATION),
        new Employee(104, "赵六", 8, 7777.77, Status.BUSY),
        new Employee(104, "赵六", 8, 7777.77, Status.FREE),
        new Employee(104, "赵六", 8, 7777.77, Status.FREE),
        new Employee(105, "田七", 38, 5555.55, Status.BUSY)
);

```

1. **查找与匹配**

   - `allMatch(Predicate p)` ： 检查是否匹配所有元素

     ```java
     boolean bl = emps.stream()
                 	 .allMatch((e) -> e.getStatus().equals(Status.BUSY));
             System.out.println(bl);
     
     ```

   - `anyMatch(Predicate p)` ： 检查是否至少匹配一个元素

     ```java
     boolean bl1 = emps.stream()
                 	  .anyMatch((e) -> e.getStatus().equals(Status.BUSY));
             System.out.println(bl1);
     
     ```

   - `noneMatch(Predicatep)` ： 检查是否没有匹配所有元素

     ```java
     boolean bl2 = emps.stream()
                       .noneMatch((e) -> e.getStatus().equals(Status.BUSY));
             System.out.println(bl2);
     
     ```

   - `findFirst()` ： 返回第一个元素

     ```java
     Optional<Employee> op = emps.stream()
             .sorted((e1, e2) -> Double.compare(e1.getSalary(), e2.getSalary()))
             .findFirst();
         System.out.println(op.get());
     
     ```

   - `findAny()` ： 返回当前流中的任意元素

     ```java
     Optional<Employee> op2 = emps.parallelStream()
     			.filter((e) -> e.getStatus().equals(Status.FREE))
     			.findAny();
     		System.out.println(op2.get());
     
     ```

   - `count()` ： 返回流中元素总数

     ```java
     long count = emps.stream()
     			     .filter((e) -> e.getStatus().equals(Status.FREE))
     				 .count();
     		System.out.println(count);
     
     ```

   - `max(Comparatorc)` ： 返回流中最大值

     ```java
     Optional<Double> op = emps.stream()
     			.map(Employee::getSalary)
     			.max(Double::compare);
     		System.out.println(op.get());
     
     ```

   - `min(Comparatorc)` ： 返回流中最小值

     ```java
     Optional<Employee> op2 = emps.stream()
     			.min((e1, e2) -> Double.compare(e1.getSalary(), e2.getSalary())	
     //等价于
     Optional<Employee> op2 = emps.stream()
     							 .min(Double：：compare);
     		System.out.println(op2.get());
     
     ```

   - `forEach(Consumerc)` ： 内部迭代

     > **外部迭代：** 使用 Collection 接口需要用户去做迭代

2. **归约**

   - `reduce(T iden, BinaryOperator b)` ：可以将流中元素反复结合起来，得到一个值；返回 T
   - `reduce(BinaryOperator b)` ： 可以将流中元素反复结合起来，得到一个值；返回`Optional<T>` 

   > **备注：** map 和reduce 的连接通常称为map-reduce 模式，因Google 用它来进行网络搜索而出名

   ```java
   @Test
   public void test1(){
       List<Integer> list = Arrays.asList(1,2,3,4,5,6,7,8,9,10);
       Integer sum = list.stream()
           .reduce(0, (x, y) -> x + y);
       System.out.println(sum);
   
       System.out.println("----------------------------------------");
   
       Optional<Double> op = emps.stream()
           .map(Employee::getSalary)
           .reduce(Double::sum);
       System.out.println(op.get());
   }
   
   //需求：搜索名字中 “六” 出现的次数
   @Test
   public void test2(){
       Optional<Integer> sum = emps.stream()
           .map(Employee::getName)
           .flatMap(TestStream::filterCharacter)
           .map((ch) -> {
               if(ch.equals('六'))
                   return 1;
               else 
                   return 0;
           }).reduce(Integer::sum);
   
       System.out.println(sum.get());
   }
   
   ```

3. **收集**

   - `collect(Collector c)` ： 将流转换为其他形式，接收一个Collector接口的实现，用于汇总Stream元素

     > Collector 接口中方法的实现决定了如何对流执行收集操作(如收集到List、Set、Map)
     >
     > Collectors 实用类提供了很多静态方法，可以方便地创建常见收集器实例

   ![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8_4.png)

   ![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8_5.png)

   ```java
   @Test
   public void test3(){
       List<String> list = emps.stream()
           .map(Employee::getName)
           .collect(Collectors.toList());
       list.forEach(System.out::println);
   
       System.out.println("----------------------------------");
   
       Set<String> set = emps.stream()
           .map(Employee::getName)
           .collect(Collectors.toSet());
       set.forEach(System.out::println);
   
       System.out.println("----------------------------------");
   
       HashSet<String> hs = emps.stream()
           .map(Employee::getName)
           .collect(Collectors.toCollection(HashSet::new));
       hs.forEach(System.out::println);
   }
   
   @Test
   public void test4(){
       Optional<Double> max = emps.stream()
           .map(Employee::getSalary)
           .collect(Collectors.maxBy(Double::compare));
       System.out.println(max.get());
   
       Optional<Employee> op = emps.stream()
           .collect(Collectors.minBy((e1, e2) -> 
                           Double.compare(e1.getSalary(), e2.getSalary())));
       System.out.println(op.get());
   
       Double sum = emps.stream()
           .collect(Collectors.summingDouble(Employee::getSalary));
       System.out.println(sum);
   
       Double avg = emps.stream()
           .collect(Collectors.averagingDouble(Employee::getSalary));
       System.out.println(avg);
   
       Long count = emps.stream()
           .collect(Collectors.counting());
       System.out.println(count);
   
       System.out.println("--------------------------------------------");
   
       DoubleSummaryStatistics dss = emps.stream()
           .collect(Collectors.summarizingDouble(Employee::getSalary));
       System.out.println(dss.getMax());
   }
   
   //分组
   @Test
   public void test5(){
       Map<Status, List<Employee>> map = emps.stream()
           .collect(Collectors.groupingBy(Employee::getStatus));
       System.out.println(map);
   }
   
   //多级分组
   @Test
   public void test6(){
       Map<Status, Map<String, List<Employee>>> map = emps.stream()
           .collect(Collectors.groupingBy(Employee::getStatus, 
                                          Collectors.groupingBy((e) -> {
               if(e.getAge() >= 60)
                   return "老年";
               else if(e.getAge() >= 35)
                   return "中年";
               else
                   return "成年";
           })));
       System.out.println(map);
   }
   
   //分区
   @Test
   public void test7(){
       Map<Boolean, List<Employee>> map = emps.stream()
           .collect(Collectors.partitioningBy((e) -> e.getSalary() >= 5000));
       System.out.println(map);
   }
   
   @Test
   public void test8(){
       String str = emps.stream()
           .map(Employee::getName)
           .collect(Collectors.joining("," , "----", "----"));
       System.out.println(str);
   }
   
   @Test
   public void test9(){
       Optional<Double> sum = emps.stream()
           .map(Employee::getSalary)
           .collect(Collectors.reducing(Double::sum));
       System.out.println(sum.get());
   }
   
   ```

### 6. 并行流与串行流

- **并行流**：把一个内容分成多个数据块，并用不同的线程分别处理每个数据块的流

- Stream API 可以声明性地通过 `parallel() 与sequential()` 在并行流与顺序流之间进行切换

- **Fork/Join 框架**：就是在必要的情况下，将一个大任务，进行拆分(fork)成若干个小任务（拆到不可再拆时），再将一个个的小任务运算的结果进行join 汇总.

  ![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8_3.png)



- **Fork/Join 框架与传统线程池的区别：** 

  **工作窃取模式：** 当执行新的任务时，可以拆分成更小的任务执行，并将小任务加到线程队列中，然后再从一个随机线程的队列中偷一个并把它放在自己的队列中

  > **fork/join 框架的优势**：
  >
  > - 一般的线程池中，若一个线程正在执行的任务由于某些原因无法继续运行，则该线程会处于等待状态
  > - fork/join 框架中，若某个子问题由于等待另一个子问题的完成而无法继续运行，则处理该子问题的线程会主动寻找其他尚未运行的子问题来执行，从而减少线程的等待时间，提高性能

  ```java
  @Test
  public void test1(){
      ForkJoinPool pool = new ForkJoinPool();
      ForkJoinTask<Long> task = new ForkJoinCalculate(0L, 10000000000L);
  
      long sum = pool.invoke(task);
      System.out.println(sum);
  }
  
  ```

  **ForkJoinCalculate 类：** 

  ```java
  public class ForkJoinCalculate extends RecursiveTask<Long>{
  	private static final long serialVersionUID = 13475679780L;
  	
  	private long start;
  	private long end;
  	
  	private static final long THRESHOLD = 10000L; //临界值
  	
  	public ForkJoinCalculate(long start, long end) {
  		this.start = start;
  		this.end = end;
  	}
  	
  	@Override
  	protected Long compute() {
  		long length = end - start;
  		
  		if(length <= THRESHOLD){
  			long sum = 0;
  			for (long i = start; i <= end; i++) {
  				sum += i;
  			}
  			return sum;
  		}else{
  			long middle = (start + end) / 2;
  			ForkJoinCalculate left = new ForkJoinCalculate(start, middle);
  			left.fork(); //拆分，并将该子任务压入线程队列
  			
  			ForkJoinCalculate right = new ForkJoinCalculate(middle+1, end);
  			right.fork();
  			
  			return left.join() + right.join();
  		}
  	}
  }
  
  ```

## 5. 接口中的默认方法与静态方法

- 接口中默认方法使用 `default` 关键字修饰

- 接口默认方法的**类优先原则**： 

  - **选择父类中的方法：** 若一个父类提供了具体的实现，则接口中具有相同名称和参数的默认方法会被忽略

    ```java
    public class TestDefaultInterface {
    	public static void main(String[] args) {
    		SubClass sc = new SubClass();
    		System.out.println(sc.getName());//打印 "嘿嘿嘿"
    	}
    }
    
    //SubClass 类
    public class SubClass extends MyClass implements MyFun{//”类优先”原则
    
    }
    
    //MyFun 接口
    public interface MyFun {
    	default String getName(){
    		return "哈哈哈";
    	}
    }
    
    //MyClass 类
    public class MyClass {
    	String getName(){
    		return "嘿嘿嘿";
    	}
    }
    
    ```

  - **接口冲突：** 若两个父接口提供了具有相同名称和参数列表的方法，则必须覆盖该方法来解决冲突

    ```java
    public class TestDefaultInterface {
    	public static void main(String[] args) {
    		SubClass sc = new SubClass();
    		System.out.println(sc.getName());
    		
    		MyInterface.show();
    	}
    }
    
    //SubClass 实现类
    public class SubClass implements MyFun, MyInterface{
    	@Override
    	public String getName() {
    		return MyInterface.super.getName();//必须指定哪个接口的默认方法
    	}
    }
    
    //MyInterface 接口
    public interface MyInterface {
    	default String getName(){
    		return "呵呵呵";
    	}
    	
    	public static void show(){
    		System.out.println("接口中的静态方法");
    	}
    }
    
    //MyFun 接口
    public interface MyFun {
    	default String getName(){
    		return "哈哈哈";
    	}
    }
    
    ```

## 6. 新时间日期API

### 1. 使用LocalDate、LocalTime、LocalDateTime

- LocalDate、LocalTime、LocalDateTime 类的实例是不可变的对象，分别表示使用ISO-8601日历系统的日期、时间、日期和时间

> **注：** ISO-8601日历系统是国际标准化组织制定的现代公民的日期和时间的表示法

```java
@Test
public void test1(){
    LocalDateTime ldt = LocalDateTime.now();
    System.out.println(ldt);

    LocalDateTime ld2 = LocalDateTime.of(2016, 11, 21, 10, 10, 10);
    System.out.println(ld2);

    LocalDateTime ldt3 = ld2.plusYears(20);
    System.out.println(ldt3);

    LocalDateTime ldt4 = ld2.minusMonths(2);
    System.out.println(ldt4);

    System.out.println(ldt.getYear());
    System.out.println(ldt.getMonthValue());
    System.out.println(ldt.getDayOfMonth());
    System.out.println(ldt.getHour());
    System.out.println(ldt.getMinute());
    System.out.println(ldt.getSecond());
}

```

![](C:/Users/lenovo/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/pics/java8_7.png)

### 2. Instant 时间戳

- **时间戳**： 以Unix元年(UTC时区1970年1月1日午夜)开始所经历的描述进行运算

```java
@Test
public void test2(){
    Instant ins = Instant.now();  //默认使用 UTC 时区(北京为东八区，时间为： UTC + 8h)
    System.out.println(ins);

    OffsetDateTime odt = ins.atOffset(ZoneOffset.ofHours(8));//对时区进行调整
    System.out.println(odt);

    System.out.println(ins.getNano());//纳秒

    Instant ins2 = Instant.ofEpochSecond(5);//相较于Unix元年增加 5 秒
    System.out.println(ins2);
}

```

### 3. Duration 和Period

- **Duration：** 用于计算两个**时间间隔**
- **Period：** 用于计算两个**日期间隔**

```java
@Test
public void test3(){
    Instant ins1 = Instant.now();
    try {
        Thread.sleep(1000);
    } catch (InterruptedException e) {}
    Instant ins2 = Instant.now();
    System.out.println("所耗费时间为：" + Duration.between(ins1, ins2));

    System.out.println("----------------------------------");

    LocalDate ld1 = LocalDate.now();
    LocalDate ld2 = LocalDate.of(2011, 1, 1);

    Period pe = Period.between(ld2, ld1);
    System.out.println(pe.getYears());
    System.out.println(pe.getMonths());
    System.out.println(pe.getDays());
}

```

### 4. 日期的操纵

- **TemporalAdjuster : ** 时间校正器，将日期调整到“下个周日”等操作
- **TemporalAdjusters : ** 通过静态方法提供了大量的常用 TemporalAdjuster 的实现

```java
@Test
public void test4(){
    LocalDateTime ldt = LocalDateTime.now();
    System.out.println(ldt);

    LocalDateTime ldt2 = ldt.withDayOfMonth(10);
    System.out.println(ldt2);

    LocalDateTime ldt3 = ldt.with(TemporalAdjusters.next(DayOfWeek.SUNDAY));
    System.out.println(ldt3);

    //自定义：下一个工作日
    LocalDateTime ldt5 = ldt.with((l) -> {
        LocalDateTime ldt4 = (LocalDateTime) l;

        DayOfWeek dow = ldt4.getDayOfWeek();

        if(dow.equals(DayOfWeek.FRIDAY)){
            return ldt4.plusDays(3);
        }else if(dow.equals(DayOfWeek.SATURDAY)){
            return ldt4.plusDays(2);
        }else{
            return ldt4.plusDays(1);
        }
    });
    System.out.println(ldt5);
}

```

### 5. 解析与格式化

`java.time.format.DateTimeFormatter` 类：

该类提供了三种格式化方法：

- **预定义的标准格式**
- **语言环境相关的格式**
- **自定义的格式**

```java
@Test
public void test5(){
//		DateTimeFormatter dtf = DateTimeFormatter.ISO_LOCAL_DATE;

    DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss E");

    LocalDateTime ldt = LocalDateTime.now();
    String strDate = ldt.format(dtf);

    System.out.println(strDate);

    LocalDateTime newLdt = ldt.parse(strDate, dtf);
    System.out.println(newLdt);
}

```

### 6. 时区的处理

- 带时区的时间为分别为：`ZonedDate`、`ZonedTime`、`ZonedDateTime` 

  > 每个时区都对应着 ID，地区 ID都为“**{区域}/{城市}**”的格式，例如：Asia/Shanghai 

- `ZoneId` ：包含了所有的时区信息

  - `getAvailableZoneIds()`：可以获取所有时区信息
  - `of(id) `：用指定的时区信息获取 ZoneId 对象

```java
@Test
public void test6(){
    Set<String> set = ZoneId.getAvailableZoneIds();
    set.forEach(System.out::println);
}

@Test
public void test7(){
    LocalDateTime ldt = LocalDateTime.now(ZoneId.of("Asia/Shanghai"));
    System.out.println(ldt);

    ZonedDateTime zdt = ZonedDateTime.now(ZoneId.of("US/Pacific"));
    System.out.println(zdt);
}

```

## 7. 其他新特性

### 1. Optional 类

> `Optional<T>`：一个容器类，代表一个值存在或不存在，替换 null ，可以避免空指针异常

**常用方法**：

- `Optional.of(T t)` ： 创建一个 Optional 实例

  ```java
  @Test
  public void test1(){
      Optional<Employee> op = Optional.of(new Employee());
      Employee emp = op.get();
      System.out.println(emp);
  }
  
  ```

- `Optional.empty()` ： 创建一个空的 Optional 实例

  ```java
  Optional<Employee> op = Optional.empty();
  System.out.println(op.get());//会报错，因为构建的容器为空，所以不能 get()
  
  ```

- `Optional.ofNullable(T t)`： 若 t 不为null，创建 Optional 实例，否则创建空实例

  ```java
  Optional<Employee> op = Optional.ofNullable(null);
  System.out.println(op.get());//会报错，因为构建的容器为空，所以不能 get()
  
  Optional<Employee> op = Optional.ofNullable(new Employee());
  System.out.println(op.get());
  ```

- `isPresent() `： 判断是否包含值

- `orElse(T t)` ： 如果调用对象包含值，返回该值，否则返回 t

- `orElseGet(Supplier s)` ： 如果调用对象包含值，返回该值，否则返回s 获取的值

- `map(Function f)` ： 如果有值对其处理，并返回处理后的Optional，否则返回Optional.empty()

- `flatMap(Function mapper)` ： 与map 类似，要求返回值必须是Optional

  ```java
  Optional<Employee> op = Optional.of(new Employee(101, "张三", 18, 9999.99));
  		
  Optional<String> op2 = op.map(Employee::getName);
  System.out.println(op2.get());
  
  Optional<String> op3 = op.flatMap((e) -> Optional.of(e.getName()));
  System.out.println(op3.get());
  ```

### 2. 重复注解与类型注解

![](../pics/java8_6.png)