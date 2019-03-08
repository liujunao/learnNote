# 一、数据类型

## 包装类型

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
Integer x = 2;     // 装箱
int y = x;         // 拆箱
```

## 缓存池

new Integer(123) 与 Integer.valueOf(123) 的区别在于：

- new Integer(123) 每次都会新建一个对象；
- Integer.valueOf(123) 会使用缓存池中的对象，多次调用会取得同一个对象的引用。

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



valueOf() 方法的实现比较简单，就是先判断值是否在缓存池中，如果在的话就直接返回缓存池的内容。

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

在使用这些基本类型对应的包装类型时，就可以直接使用缓冲池中的对象。



# 二、String

## 概览

String 被声明为 final，因此它不可被继承。

内部使用 char 数组存储数据，该数组被声明为 final，这意味着 value 数组初始化之后就不能再引用其它数组。并且 String 内部没有改变 value 数组的方法，因此可以保证 String 不可变。

```java
public final class String
    implements java.io.Serializable, Comparable<String>, CharSequence {
    /** The value is used for character storage. */
    private final char value[];
```

## 不可变的好处

**1. 可以缓存 hash 值** 

因为 String 的 hash 值经常被使用，例如 String 用做 HashMap 的 key。不可变的特性可以使得 hash 值也不可变，因此只需要进行一次计算。

**2. String Pool 的需要** 

如果一个 String 对象已经被创建过了，那么就会从 String Pool 中取得引用。只有 String 是不可变的，才可能使用 String Pool。

<div align="center"> <img src="../pics//f76067a5-7d5f-4135-9549-8199c77d8f1c.jpg" width=""/> </div><br>

**3. 安全性** 

String 经常作为参数，String 不可变性可以保证参数不可变。例如在作为网络连接参数的情况下如果 String 是可变的，那么在网络连接过程中，String 被改变，改变 String 对象的那一方以为现在连接的是其它主机，而实际情况却不一定是。

**4. 线程安全** 

String 不可变性天生具备线程安全，可以在多个线程中安全地使用。

[Program Creek : Why String is immutable in Java?](https://www.programcreek.com/2013/04/why-string-is-immutable-in-java/)

## String, StringBuffer and StringBuilder

**1. 可变性** 

- String 不可变
- StringBuffer 和 StringBuilder 可变

**2. 线程安全** 

- String 不可变，因此是线程安全的
- StringBuilder 不是线程安全的
- StringBuffer 是线程安全的，内部使用 synchronized 进行同步

[StackOverflow : String, StringBuffer, and StringBuilder](https://stackoverflow.com/questions/2971315/string-stringbuffer-and-stringbuilder)

## String Pool

字符串常量池（String Poll）保存着所有字符串字面量（literal strings），这些字面量在编译时期就确定。不仅如此，还可以使用 String 的 intern() 方法在运行过程中将字符串添加到 String Poll 中。

当一个字符串调用 intern() 方法时，如果 String Poll 中已经存在一个字符串和该字符串值相等（使用 equals() 方法进行确定），那么就会返回 String Poll 中字符串的引用；否则，就会在 String Poll 中添加一个新的字符串，并返回这个新字符串的引用。

下面示例中，s1 和 s2 采用 new String() 的方式新建了两个不同字符串，而 s3 和 s4 是通过 s1.intern() 方法取得一个字符串引用。intern() 首先把 s1 引用的字符串放到 String Pool 中，然后返回这个字符串引用。因此 s3 和 s4 引用的是同一个字符串。

```java
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
System.out.println(s4 == s5);  // true
```

1. 如果是采用 "bbb" 这种字面量的形式创建字符串，会自动地将字符串放入 String Pool 中
2. 如果采用 new String() 的方式创建的字符对象不进入字符串池中
3. 如果采用表达式中含有变量，则不会进入字符串池中

在 Java 7 之前，String Poll 被放在运行时常量池中，它属于永久代。而在 Java 7，String Poll 被移到堆中。这是因为永久代的空间有限，在大量使用字符串的场景下会导致 OutOfMemoryError 错误。

- [String中intern的方法](https://www.cnblogs.com/wanlipeng/archive/2010/10/21/1857513.html) 
- [深入解析 String#intern](https://tech.meituan.com/in_depth_understanding_string_intern.html)

## new String("abc")

使用这种方式一共会创建两个字符串对象（前提是 String Poll 中还没有 "abc" 字符串对象）。

- "abc" 属于字符串字面量，因此编译时期会在 String Poll 中创建一个字符串对象，指向这个 "abc" 字符串字面量
- 使用 new 的方式会在堆中创建一个字符串对象

创建一个测试类，其 main 方法中使用这种方式来创建字符串对象。

```java
public class NewStringTest {
    public static void main(String[] args) {
        String s = new String("abc");
    }
}
```

使用 javap -verbose 进行反编译，得到以下内容：

```java
// ...
Constant pool:
// ...
   #2 = Class              #18            // java/lang/String
   #3 = String             #19            // abc
// ...
  #18 = Utf8               java/lang/String
  #19 = Utf8               abc
// ...

  public static void main(java.lang.String[]);
    descriptor: ([Ljava/lang/String;)V
    flags: ACC_PUBLIC, ACC_STATIC
    Code:
      stack=3, locals=2, args_size=1
         0: new           #2                  // class java/lang/String
         3: dup
         4: ldc           #3                  // String abc
         6: invokespecial #4                  // Method java/lang/String."<init>":(Ljava/lang/String;)V
         9: astore_1
// ...
```

在 Constant Poll 中，#19 存储这字符串字面量 "abc"，#3 是 String Poll 的字符串对象，它指向 #19 这个字符串字面量。在 main 方法中，0: 行使用 new #2 在堆中创建一个字符串对象，并且使用 ldc #3 将 String Poll 中的字符串对象作为 String 构造函数的参数。

以下是 String 构造函数的源码，可以看到，在将一个字符串对象作为另一个字符串对象的构造函数参数时，并不会完全复制 value 数组内容，而是都会指向同一个 value 数组。

```java
public String(String original) {
    this.value = original.value;
    this.hash = original.hash;
}
```

# 三、运算

## 参数传递

Java 的参数是以值传递的形式传入方法中，而不是引用传递。

- 形参是基本数据类型的：将实参的值传递给形参的基本数据类型的变量
- 形参是引用数据类型的：将实参的引用类型变量的值（对应的堆空间的对象实体的首地址值）传递给形参的引用类型变量。

以下代码中 Dog dog 的 dog 是一个指针，存储的是对象的地址。在将一个参数传入一个方法时，本质上是将对象的地址以值的方式传递到形参中。因此在方法中改变指针引用的对象，那么这两个指针此时指向的是完全不同的对象，一方改变其所指向对象的内容对另一方没有影响。

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

    private static void func(Dog dog) {
        System.out.println(dog.getObjectAddress()); // Dog@4554617c
        dog = new Dog("B");
        System.out.println(dog.getObjectAddress()); // Dog@74a14482
        System.out.println(dog.getName());          // B
    }
}
```

但是如果在方法中改变对象的字段值会改变原对象该字段值，因为改变的是同一个地址指向的内容。

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

## float 与 double

1.1 字面量属于 double 类型，不能直接将 1.1 直接赋值给 float 变量，因为这是向下转型。Java 不能隐式执行向下转型，因为这会使得精度降低。

```java
// float f = 1.1;
```

1.1f 字面量才是 float 类型。

```java
float f = 1.1f;
```

## 隐式类型转换

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

## switch

从 Java 7 开始，可以在 switch 条件判断语句中使用 String 对象。

```java
String s = "a";
switch (s) {
    case "a":
        System.out.println("aaa");
        break;
    case "b":
        System.out.println("bbb");
        break;
}
```

switch 不支持 long，是因为 switch 的设计初衷是对那些只有少数的几个值进行等值判断，如果值过于复杂，那么还是用 if 比较合适。

```java
// long x = 111;
// switch (x) { // Incompatible types. Found: 'long', required: 'char, byte, short, int, Character, Byte, Short, Integer, String, or an enum'
//     case 111:
//         System.out.println(111);
//         break;
//     case 222:
//         System.out.println(222);
//         break;
// }
```

[StackOverflow : Why can't your switch statement data type be long, Java?](https://stackoverflow.com/questions/2676210/why-cant-your-switch-statement-data-type-be-long-java)

# 四、继承

## 访问权限

Java 中有三个访问权限修饰符：private、protected 以及 public，如果不加访问修饰符，表示包级可见。

可以对类或类中的成员（字段以及方法）加上访问修饰符。

- 类可见表示其它类可以用这个类创建实例对象。
- 成员可见表示其它类可以用这个类的实例对象访问到该成员；

protected 用于修饰成员，表示在继承体系中成员对于子类可见，但是这个访问修饰符对于类没有意义。

设计良好的模块会隐藏所有的实现细节，把它的 API 与它的实现清晰地隔离开来。模块之间只通过它们的 API 进行通信，一个模块不需要知道其他模块的内部工作情况，这个概念被称为信息隐藏或封装。因此访问权限应当尽可能地使每个类或者成员不被外界访问。

如果子类的方法重写了父类的方法，那么子类中该方法的访问级别不允许低于父类的访问级别。这是为了确保可以使用父类实例的地方都可以使用子类实例，也就是确保满足[里氏替换原则](http://wiki.jikexueyuan.com/project/java-design-pattern-principle/principle-2.html) 

字段决不能是公有的，因为这么做的话就失去了对这个字段修改行为的控制，客户端可以对其随意修改。例如下面的例子中，AccessExample 拥有 id 共有字段，如果在某个时刻，我们想要使用 int 去存储 id 字段，那么就需要去修改所有的客户端代码。

```java
public class AccessExample {
    public String id;
}
```

可以使用公有的 getter 和 setter 方法来替换公有字段，这样的话就可以控制对字段的修改行为。


```java
public class AccessExample {

    private int id;

    public String getId() {
        return id + "";
    }

    public void setId(String id) {
        this.id = Integer.valueOf(id);
    }
}
```

但是也有例外，如果是包级私有的类或者私有的嵌套类，那么直接暴露成员不会有特别大的影响。

```java
public class AccessWithInnerClassExample {

    private class InnerClass {
        int x;
    }

    private InnerClass innerClass;

    public AccessWithInnerClassExample() {
        innerClass = new InnerClass();
    }

    public int getValue() {
        return innerClass.x;  // 直接访问
    }
}
```

## 抽象类与接口

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

## super

- 访问父类的构造函数：可以使用 super() 函数访问父类的构造函数，从而委托父类完成一些初始化的工作。
- 访问父类的成员：如果子类重写了父类的中某个方法的实现，可以通过使用 super 关键字来引用父类的方法实现。

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

## this

1. 使用在类中，可以用来修饰属性、方法、构造器
2. 表示当前对象或者是当前正在创建的对象
3. 当形参与成员变量重名时，如果在方法内部需要使用成员变量，必须添加this来表明该变量时类成员
4. 在任意方法内，如果使用当前类的成员变量或成员方法可以在其前面添加this，增强程序的阅读性
5. 在构造器中使用“this(形参列表)”显式的调用本类中重载的其它的构造器
   1. 要求“this(形参列表)”要声明在构造器的首行！
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

## 重写与重载

**1. 重写（Override）** 

存在于继承体系中，指子类实现了一个与父类在方法声明上完全相同的一个方法。

为了满足里式替换原则，重写有有以下两个限制：

- 子类方法的访问权限必须大于等于父类方法；
- 子类方法的返回类型必须是父类方法返回类型或为其子类型。

使用 @Override 注解，可以让编译器帮忙检查是否满足上面的两个限制条件。

**2. 重载（Overload）** 

存在于同一个类中，指一个方法与已经存在的方法名称上相同，但是参数类型、个数、顺序至少有一个不同。

应该注意的是，返回值不同，其它都相同不算是重载。

# 五、Object 通用方法

## 概览

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

## equals()

**1. 等价关系** 

Ⅰ 自反性

```java
x.equals(x); // true
```

Ⅱ 对称性

```java
x.equals(y) == y.equals(x); // true
```

Ⅲ 传递性

```java
if (x.equals(y) && y.equals(z))
    x.equals(z); // true;
```

Ⅳ 一致性

多次调用 equals() 方法结果不变

```java
x.equals(y) == x.equals(y); // true
```

Ⅴ 与 null 的比较

对任何不是 null 的对象 x 调用 x.equals(null) 结果都为 false

```java
x.equals(null); // false;
```

**2. 等价与相等** 

- 对于基本类型，== 判断两个值是否相等，基本类型没有 equals() 方法。
- 对于引用类型，== 判断两个变量是否引用同一个对象，而 equals() 判断引用的对象是否等价。

```java
Integer x = new Integer(1);
Integer y = new Integer(1);
System.out.println(x.equals(y)); // true
System.out.println(x == y);      // false
```

**3. 实现** 

- 检查是否为同一个对象的引用，如果是直接返回 true；
- 检查是否是同一个类型，如果不是，直接返回 false；
- 将 Object 对象进行转型；
- 判断每个关键域是否相等。

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
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        EqualExample that = (EqualExample) o;

        if (x != that.x) return false;
        if (y != that.y) return false;
        return z == that.z;
    }
}
```

## hashCode()

hashCode() 返回散列值，而 equals() 是用来判断两个对象是否等价。等价的两个对象散列值一定相同，但是散列值相同的两个对象不一定等价。

在覆盖 equals() 方法时应当总是覆盖 hashCode() 方法，保证等价的两个对象散列值也相等。

下面的代码中，新建了两个等价的对象，并将它们添加到 HashSet 中。我们希望将这两个对象当成一样的，只在集合中添加一个对象，但是因为 EqualExample 没有实现 hasCode() 方法，因此这两个对象的散列值是不同的，最终导致集合添加了两个等价的对象。

```java
EqualExample e1 = new EqualExample(1, 1, 1);
EqualExample e2 = new EqualExample(1, 1, 1);
System.out.println(e1.equals(e2)); // true
HashSet<EqualExample> set = new HashSet<>();
set.add(e1);
set.add(e2);
System.out.println(set.size());   // 2
```

理想的散列函数应当具有均匀性，即不相等的对象应当均匀分布到所有可能的散列值上。这就要求了散列函数要把所有域的值都考虑进来。可以将每个域都当成 R 进制的某一位，然后组成一个 R 进制的整数。R 一般取 31，因为它是一个奇素数，如果是偶数的话，当出现乘法溢出，信息就会丢失，因为与 2 相乘相当于向左移一位。

一个数与 31 相乘可以转换成移位和减法：`31*x == (x<<5)-x`，编译器会自动进行这个优化。

```java
@Override
public int hashCode() {
    int result = 17;
    result = 31 * result + x;
    result = 31 * result + y;
    result = 31 * result + z;
    return result;
}
```

## toString()

默认返回 ToStringExample@4554617c 这种形式，其中 @ 后面的数值为散列码的无符号十六进制表示。

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

## clone()

**1. cloneable** 

clone() 是 Object 的 protected 方法，它不是 public，一个类不显式去重写 clone()，其它类就不能直接去调用该类实例的 clone() 方法。

```java
public class CloneExample {
    private int a;
    private int b;
}
```

```java
CloneExample e1 = new CloneExample();
// CloneExample e2 = e1.clone(); // 'clone()' has protected access in 'java.lang.Object'
```

重写 clone() 得到以下实现：

```java
public class CloneExample {
    private int a;
    private int b;

    @Override
    protected CloneExample clone() throws CloneNotSupportedException {
        return (CloneExample)super.clone();
    }
}
```

```java
CloneExample e1 = new CloneExample();
try {
    CloneExample e2 = e1.clone();
} catch (CloneNotSupportedException e) {
    e.printStackTrace();
}
```

```html
java.lang.CloneNotSupportedException: CloneExample
```

以上抛出了 CloneNotSupportedException，这是因为 CloneExample 没有实现 Cloneable 接口。

应该注意的是，clone() 方法并不是 Cloneable 接口的方法，而是 Object 的一个 protected 方法。Cloneable 接口只是规定，如果一个类没有实现 Cloneable 接口又调用了 clone() 方法，就会抛出 CloneNotSupportedException。

```java
public class CloneExample implements Cloneable {
    private int a;
    private int b;

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
```

**2. 浅拷贝** 

拷贝对象和原始对象的引用类型引用同一个对象。

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
```

```java
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

**3. 深拷贝** 

拷贝对象和原始对象的引用类型引用不同对象。

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
```

```java
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

**4. clone() 的替代方案** 

使用 clone() 方法来拷贝一个对象即复杂又有风险，它会抛出异常，并且还需要类型转换。Effective Java 书上讲到，最好不要去使用 clone()，可以使用拷贝构造函数或者拷贝工厂来拷贝一个对象。

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
```

```java
CloneConstructorExample e1 = new CloneConstructorExample();
CloneConstructorExample e2 = new CloneConstructorExample(e1);
e1.set(2, 222);
System.out.println(e2.get(2)); // 2
```

# 六、关键字

## final

**1. 数据** 

声明数据为常量，可以是编译时常量，也可以是在运行时被初始化后不能被改变的常量。

- 对于基本类型，final 使数值不变；
- 对于引用类型，final 使引用不变，也就不能引用其它对象，但是被引用的对象本身是可以修改的。

```java
final int x = 1;
// x = 2;  // cannot assign value to final variable 'x'
final A y = new A();
y.a = 1;
```

**2. 方法** 

声明方法不能被子类重写。

private 方法隐式地被指定为 final，如果在子类中定义的方法和基类中的一个 private 方法签名相同，此时子类的方法不是重写基类方法，而是在子类中定义了一个新的方法。

**3. 类** 

声明类不允许被继承。

## static

**1. 静态变量** 

- 静态变量：又称为类变量，也就是说这个变量属于类的，类所有的实例都共享静态变量，可以直接通过类名来访问它。静态变量在内存中只存在一份。
- 实例变量：每创建一个实例就会产生一个实例变量，它与该实例同生共死。

```java
public class A {
    private int x;         // 实例变量
    private static int y;  // 静态变量

    public static void main(String[] args) {
        // int x = A.x;  // Non-static field 'x' cannot be referenced from a static context
        A a = new A();
        int x = a.x;
        int y = A.y;
    }
}
```

**2. 静态方法** 

静态方法在类加载的时候就存在了，它不依赖于任何实例。所以静态方法必须有实现，也就是说它不能是抽象方法。

```java
public abstract class A {
    public static void func1(){
    }
    // public abstract static void func2();  // Illegal combination of modifiers: 'abstract' and 'static'
}
```

只能访问所属类的静态字段和静态方法，方法中不能有 this 和 super 关键字。

```java
public class A {
    private static int x;
    private int y;

    public static void func1(){
        int a = x;
        // int b = y;  // Non-static field 'y' cannot be referenced from a static context
        // int b = this.y;     // 'A.this' cannot be referenced from a static context
    }
}
```

**3. 静态语句块** 

静态语句块在类初始化时运行一次。

```java
public class A {
    static {
        System.out.println("123");
    }

    public static void main(String[] args) {
        A a1 = new A();
        A a2 = new A();
    }
}
```

```html
123
```

**4. 静态内部类** 

非静态内部类依赖于外部类的实例，而静态内部类不需要。

```java
public class OuterClass {
    class InnerClass {
    }

    static class StaticInnerClass {
    }

    public static void main(String[] args) {
        // InnerClass innerClass = new InnerClass(); // 'OuterClass.this' cannot be referenced from a static context
        OuterClass outerClass = new OuterClass();
        InnerClass innerClass = outerClass.new InnerClass();
        StaticInnerClass staticInnerClass = new StaticInnerClass();
    }
}
```

静态内部类不能访问外部类的非静态的变量和方法。

**5. 静态导包** 

在使用静态变量和方法时不用再指明 ClassName，从而简化代码，但可读性大大降低。

```java
import static com.xxx.ClassName.*
```

**6. 初始化顺序** 

静态变量和静态语句块优先于实例变量和普通语句块，静态变量和静态语句块的初始化顺序取决于它们在代码中的顺序。

```java
public static String staticField = "静态变量";
```

```java
static {
    System.out.println("静态语句块");
}
```

```java
public String field = "实例变量";
```

```java
{
    System.out.println("普通语句块");
}
```

最后才是构造函数的初始化。

```java
public InitialOrderTest() {
    System.out.println("构造函数");
}
```

存在继承的情况下，初始化顺序为：

- 父类（静态变量、静态语句块）
- 子类（静态变量、静态语句块）
- 父类（实例变量、普通语句块）
- 父类（构造函数）
- 子类（实例变量、普通语句块）
- 子类（构造函数）




## native 

> 实现 java 与其他语言的交互（如：C，C++）

[java中的native关键字](http://www.blogjava.net/shiliqiang/articles/287920.html)

[Java中Native关键字的作用](https://www.cnblogs.com/Qian123/p/5702574.html) 



## transient 

**为了数据安全，避免序列化和反序列化** 

>  当对象被序列化时，被transient关键字修饰的变量不会被序列化到目标文件
>
>  当对象从序列化文件重构对象时（反序列化过程），被transient字段修饰的变量不会被恢复

注： 

1. 一旦变量被transient修饰，变量将不再是对象持久化的一部分，该变量内容在序列化后无法获得访问
2. transient关键字只能修饰变量，而不能修饰方法和类。注意，本地变量是不能被transient关键字修饰的。变量如果是用户自定义类变量，则该类需要实现Serializable接口
3. 被transient关键字修饰的变量不再能被序列化，一个静态变量不管是否被transient修饰，均不能被序列化



## abstract

abstract：抽象的，可以用来修饰类、方法

1. abstract修饰类：抽象类。当我们设计一个类，不需要创建此类的实例时候，就可以考虑将其设置为抽象的，由其子类实现这个类的抽象方法以后，就行实例化
   1. 不可被实例化
   2. 抽象类有构造器 (凡是类都有构造器)
   3. 抽象方法所在的类，一定是抽象类。
   4. 抽象类中可以没有抽象方法。


2. abstract修饰方法：抽象方法
   1. 格式：没有方法体，包括{}.如：public abstract void eat();
   2. 抽象方法只保留方法的功能，而具体的执行，交给继承抽象类的子类，由子类重写此抽象方法。
   3. 若子类继承抽象类，并重写了所有的抽象方法，则此类是一个"实体类",即可以实例化
   4. 若子类继承抽象类，没有重写所有的抽象方法，意味着此类中仍有抽象方法，则此类必须声明为抽象的！



## interface

接口（interface）  是与类并行的一个概念

1. 接口可以看做是一个特殊的抽象类。是常量与抽象方法的一个集合，不能包含变量、一般的方法。
2. 接口是没有构造器的。
3. 接口定义的就是一种功能。此功能可以被类所实现（implements）。
4. 实现接口的类，必须要重写其中的所有的抽象方法，方可实例化。若没有重写所有的抽象方法，则此类仍为一个抽象类
5. 类可以实现多个接口。----java 中的类的继承是单继承的
6. 接口与接口之间也是继承的关系，而且可以实现多继承
7. 接口与具体的实现类之间也存在多态性



## 代码块

执行顺序：（优先级从高到低。）静态代码块>mian方法>构造代码块>构造方法。

代码块：是类的第4个成员
作用：用来初始化类的属性


1. **静态代码块**： 
   1. 使用static关键字声明的代码块
   2. 里面可以有输出语句
   3. 随着类的加载而加载，而且只被加载一次
   4. 多个静态代码块之间按照顺序结构执行
   5. 静态代码块的执行要早于非静态代码块的执行
   6. 静态的代码块中只能执行静态的结构(类属性，类方法)
   7. 静态代码块不能存在于任何方法体内
   8. 静态代码块不能直接访问静态实例变量和实例方法，需要通过类的实例对象来访问
2. **非静态代码块：** 普通代码块，构造代码块，同步代码块

   1. 可以对类的属性(静态的 & 非静态的)进行初始化操作，同时也可以调用本类声明的方法(静态的 & 非静态的)
   2. 里面可以有输出语句
   3. 一个类中可以有多个非静态的代码块，多个代码块之间按照顺序结构执行
   4. 每创建一个类的对象，非静态代码块就加载一次。
   5. 非静态代码块的执行要早于构造器

 undefined关于属性赋值的操作：
  ①默认的初始化
  ②显式的初始化或代码块初始化(此处两个结构按照顺序执行) 
  ③构造器中；
  —————————以上是对象的属性初始化的过程—————————————
  ④通过方法对对象的相应属性进行修改



# 七、反射

- Java Reflection     
  - Reflection（反射）是被视为动态语言的关键，反射机制允许程序在执行期借助于Reflection API取得任何类的内部信息，并能直接操作任意对象的内部属性及方法
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

1. 前提：若已知具体的类，通过类的class属性获取，该方法最为安全可靠，程序性能最高       

  实例：Class clazz = String.class

2. 前提：已知某个类的实例，调用该实例的getClass()方法获取Class对象       

   实例：Class clazz = person.getClass()

3. 前提：已知一个类的全类名，且该类在类路径下，可通过Class类的静态方法forName()获取，可能抛出ClassNotFoundException       

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

- 引导类加载器：用C++编写的，是JVM自带的类加载器，负责Java平台核心库，用来加载核心类库。该加载器无法直接获取
- 扩展类加载器：负责jre/lib/ext目录下的jar包或 –D java.ext.dirs 指定目录下的jar包装入工作库
- 系统类加载器：负责java –classpath 或 –D java.class.path所指的目录下的类与jar包装入工作 ，是最常用的加载器

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
   2. 类的构造器的访问权限需要足够。

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

    //getDeclaredMethod(String methodName,Class ... params):获取运行时类中声明了的指定的方法
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

- public void setAccessible(true)访问私有属性时，让这个属性可见。 

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

//		Field id = clazz.getField("id");
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
- Proxy ：专门完成代理的操作类，是所有动态代理类的父类。通过此类为一个或多个接口动态地生成实现类。
- 提供用于创建动态代理类和动态代理对象的静态方法
  - `static Class<?>   getProxyClass(ClassLoader loader, Class<?>... interfaces)`  创建一个动态代理类所对应的Class对象
  - `static Object   newProxyInstance(ClassLoader loader, Class<?>[] interfaces, InvocationHandler h)`  直接创建一个动态代理对象

### 2. 动态代理步骤

1. 创建一个实现接口 InvocationHandler 的类，它必须实现 invoke 方法，以完成代理的具体操作
2. 创建被代理的类以及接口
3. 通过Proxy的静态方法`newProxyInstance(ClassLoader loader,Class[] interfaces,InvocationHandler h)` 创建一个Subject接口代理
4. 通过 Subject 代理调用RealSubject 实现类的方法

**静态代理模式：** 

```java
//接口
interface ClothFactory{
	void productCloth();
}
//被代理类
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

public class TestClothProduct {
	public static void main(String[] args) {
		NikeClothFactory nike = new NikeClothFactory();//创建被代理类的对象
		ProxyFactory proxy = new ProxyFactory(nike);//创建代理类的对象
		proxy.productCloth();
	}
}
```

**动态代理：** 体会反射是动态语言的关键

```java
interface Subject {
	void action();
}

// 被代理类
class RealSubject implements Subject {
	public void action() {
		System.out.println("我是被代理类，记得要执行我哦！么么~~");
	}
}

class MyInvocationHandler implements InvocationHandler {
	Object obj;// 实现了接口的被代理类的对象的声明

	// ①给被代理的对象实例化  ②返回一个代理类的对象
	public Object blind(Object obj) {
		this.obj = obj;
		return Proxy.newProxyInstance(obj.getClass().getClassLoader(), obj
				.getClass().getInterfaces(), this);
	}
	//当通过代理类的对象发起对被重写的方法的调用时，都会转换为对如下的invoke方法的调用
	@Override
	public Object invoke(Object proxy, Method method, Object[] args)
			throws Throwable {
		//method方法的返回值时returnVal
		Object returnVal = method.invoke(obj, args);
		return returnVal;
	}
}

public class TestProxy {
	public static void main(String[] args) {
		//1.被代理类的对象
		RealSubject real = new RealSubject();
		//2.创建一个实现了InvacationHandler接口的类的对象
		MyInvocationHandler handler = new MyInvocationHandler();
		//3.调用blind()方法，动态的返回一个同样实现了real所在类实现的接口Subject的代理类的对象。
		Object obj = handler.blind(real);
		Subject sub = (Subject)obj;//此时sub就是代理类的对象
		
		sub.action();//转到对InvacationHandler接口的实现类的invoke()方法的调用
		
		//再举一例
		NikeClothFactory nike = new NikeClothFactory();
		ClothFactory proxyCloth = (ClothFactory)handler.blind(nike);//proxyCloth即为代理类的对象
		proxyCloth.productCloth();
	}
}
```

### 3. 动态代理与AOP

- 使用Proxy生成一个动态代理时，往往并不会凭空产生一个动态代理，这样没有太大的意义。通常都是为指定的目标对象生成动态代理
- 这种动态代理在AOP中被称为AOP代理，AOP代理可代替目标对象，AOP代理包含了目标对象的全部方法。但AOP代理中的方法与目标对象的方法存在差异：AOP代理里的方法可以在执行目标方法之前、之后插入一些通用处理

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


- [Trail: The Reflection API](https://docs.oracle.com/javase/tutorial/reflect/index.html)
- [深入解析 Java 反射（1）- 基础](http://www.sczyh30.com/posts/Java/java-reflection-1/)

# 八、异常

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

- [Java 入门之异常处理](https://www.tianmaying.com/tutorial/Java-Exception)
- [Java 异常的面试问题及答案 -Part 1](http://www.importnew.com/7383.html)

# 九、泛型

1. 对象实例化时不指定泛型，默认为：Object。

2. 泛型不同的引用不能相互赋值。

3. 加入集合中的对象类型必须与指定的泛型类型一致。

4. 静态方法中不能使用类的泛型。

5. 如果泛型类是一个接口或抽象类，则不可创建泛型类的对象。

6. 不能在catch中使用泛型

7. 从泛型类派生子类，泛型类型需具体化

8. 泛型与继承的关系

   A类是B类的子类，G是带泛型声明的类或接口。那么G\<A>不是G\<B>的子类！

9. 通配符:?

   A类是B类的子类，G是带泛型声明的类或接口。则G\<?> 是G\<A>、G\<B>的父类！
   ①以List<?>为例，能读取其中的数据。因为不管存储的是什么类型的元素，其一定是Object类的或其子类的。
   ①以List<?>为例，不可以向其中写入数据。因为没有指明可以存放到其中的元素的类型！唯一例外的是：null

10. List<？ extends A> :可以将List\<A>的对象或List\<B>的对象赋给List<? extends A>。其中B 是A的子类

    ? super A:可以将List\<A>的对象或List\<B>的对象赋给List<? extends A>。其中B 是A的父类

- [Java 泛型详解](http://www.importnew.com/24029.html)
- [10 道 Java 泛型面试题](https://cloud.tencent.com/developer/article/1033693) 



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

   [JAVA自定义注解、元注解介绍及自定义注解使用场景](https://blog.csdn.net/bluuusea/article/details/79996572) 

[注解 Annotation 实现原理与自定义注解例子](https://www.cnblogs.com/acm-bingzi/p/javaAnnotation.html) 

---

Java 注解是附加在代码中的一些元信息，用于一些工具在编译、运行时进行解析和使用，起到说明、配置的功能。注解不会也不能影响代码的实际逻辑，仅仅起到辅助性的作用。

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

- Java 是 Internet 上的语言，它从语言级上提供了对网络应用程序的支持，程序员能够很容易开发常见的网络应用程序
- Java 提供的网络类库，可以实现无痛的网络连接，联网的底层细节被隐藏在 Java 的本机安装系统里，由 JVM 进行控制。并且 Java 实现了一个跨平台的网络库，程序员面对的是一个统一的网络编程环境

### 1. 网络基础

- 计算机网络：把分布在不同地理区域的计算机与专门的外部设备用通信线路互连成一个规模大、功能强的网络系统，从而使众多的计算机可以方便地互相传递信息、共享硬件、软件、数据信息等资源
- 网络编程的目的：直接或间接地通过网络协议与其它计算机进行通讯
- 网络编程中有两个主要的问题：
  - 如何准确地定位网络上一台或多台主机
  - 找到主机后如何可靠高效地进行数据传输
- 如何实现网络中的主机互相通信：
  - 通信双方地址 
  - 一定的规则（有两套参考模型）
    1. OSI 参考模型：模型过于理想化，未能在因特网上进行广泛推广
    2. TCP/IP参考模型(或TCP/IP协议)：事实上的国际标准

### 2. 网络通信协议

![](../pics/internet2.png)

![](../pics/internet.png)

## 2. 通讯要素

### 1.  IP和端口号

- IP 地址：InetAddress
  - 唯一的标识 Internet 上的计算机
  - 本地回环地址(hostAddress)：127.0.0.1      主机名(hostName)：localhost
  - 不易记忆
- 端口号标识正在计算机上运行的进程（程序）
  - 不同的进程有不同的端口号
  - 被规定为一个 16 位的整数 0~65535。其中，0~1023被预先定义的服务通信占用（如MySql占用端口3306，http占用端口80等）。除非我们需要访问这些特定服务，否则，就应该使用 1024~65535 这些端口中的某一个进行通信，以免发生端口冲突 
- 端口号与IP地址的组合得出一个网络套接字

### 2. 网络通信协议

- 网络通信协议  

  计算机网络中实现通信必须有一些约定，即通信协议，对速率、传输代码、代码结构、传输控制步骤、出错控制等制定标准

- 通信协议分层的思想  

  由于结点之间联系很复杂，在制定协议时，把复杂成份分解成一些简单的成份，再将它们复合起来。最常用的复合方式是层次方式，即同层间可以通信、上一层可以调用下一层，而与再下一层不发生关系。各层互不影响，利于系统的开发和扩展

## 3. InetAddress类

- Internet上的主机有两种方式表示地址：
  - 域名(hostName)：www.baidu.com
  - IP 地址(hostAddress)：111.13.100.92
- InetAddress类主要表示IP地址，两个子类：Inet4Address、Inet6Address。
- InetAddress 类对象含有一个 Internet 主机地址的域名和IP地址：www.baidu.com 和 111.13.100.92
- 域名容易记忆，当在连接网络时输入一个主机的域名后，域名服务器（DNS）负责将域名转化成IP地址，这样才能和主机建立连接。 -------域名解析

> InetAddress:位于java.net包下
>
>  * InetAddress用来代表IP地址。一个InetAdress的对象就代表着一个IP地址
>  * 如何创建InetAddress的对象：getByName(String host)
>  * getHostName(): 获取IP地址对应的域名
>  * getHostAddress():获取IP地址

```java
public class TestInetAddress {
	public static void main(String[] args) throws Exception {
		//创建一个InetAddress对象：getByName()
		InetAddress inet = InetAddress.getByName("www.baidu.com");
		//inet = InetAddress.getByName("111.13.100.92");
		System.out.println(inet);
		//两个方法
		System.out.println(inet.getHostName());
		System.out.println(inet.getHostAddress());
		//获取本机的IP:getLocalHost()
		InetAddress inet1 = InetAddress.getLocalHost();
		System.out.println(inet1);
		System.out.println(inet1.getHostName());
		System.out.println(inet1.getHostAddress());
	}
}
```



## 4. TCP网络通信(传输控制协议)

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
