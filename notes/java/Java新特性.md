- 推荐阅读： [java 8 新特性]([https://snailclimb.gitee.io/javaguide/#/java/What's%20New%20in%20JDK8/Java8Tutorial](https://snailclimb.gitee.io/javaguide/#/java/What's New in JDK8/Java8Tutorial)) 

# 一、Java8

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

  - **语法格式五**：若 Lambda 体中只有一条语句， return 和 大括号都可以省略不写

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

1. **消费型接口：** `void Consumer<T>`，对类型为 T 的对象应用操作，包含方法：`void accept(T t)` 

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
   
3. **函数型接口：** `R Function<T, R>`，对类型为T的对象应用操作，并返回结果，结果是 R 类型的对象，包含方法：`R apply(T t)`

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

![](../../pics/java/javaN_1.png)

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

## 4、Stream API

推荐阅读：**[Java 8 中的 Streams API 详解(IBM)](https://developer.ibm.com/zh/technologies/java/articles/j-lo-java8streamapi/)** 

### (1) Stream 简介

- Stream 如同一个迭代器，单向、不可往复，数据只能遍历一次，遍历过一次后即用尽

**Stream 优势**：

- Stream 专注对集合对象进行各种非常便利、高效的聚合操作，或大批量数据操作

- 提供串行和并行模式操作，其中并发模式能充分利用多核处理器的优势，使用 fork/join 并行方式来拆分任务和加速处理过程

    > Stream 的并行操作依赖于 Java7 中引入的 Fork/Join 框架来拆分任务和加速处理过程

**聚合操作**：Java 代码经常不得不依赖于关系型数据库的聚合操作

- Java 7 的排序、取值实现

    ```java
    List<Transaction> groceryTransactions = new Arraylist<>();
    for(Transaction t: transactions){
    	if(t.getType() == Transaction.GROCERY){
    		groceryTransactions.add(t);
    	}
    }
    
    Collections.sort(groceryTransactions, new Comparator(){
    	public int compare(Transaction t1, Transaction t2){
    		return t2.getValue().compareTo(t1.getValue());
    	}
    });
    
    List<Integer> transactionIds = new ArrayList<>();
    for(Transaction t: groceryTransactions){
    	transactionsIds.add(t.getId());
    }
    ```

- Java 8 的排序、取值实现

    ```java
    List<Integer> transactionsIds = transactions.parallelStream()
    											.filter(t -> t.getType() == Transaction.GROCERY)
     											.sorted(comparing(Transaction::getValue).reversed())
    											.map(Transaction::getId)
    											.collect(toList());
    ```

### (2) Stream 的操作

**使用流的三个基本步骤**：

- **创建Stream**：一个数据源（如：集合、数组），获取一个流
- **中间操作**：一个中间操作链，对数据源的数据进行处理
- **终止操作(终端操作)**：一个终止操作，执行中间操作链，并产生结果

> 每次转换原有 Stream 对象不改变，返回一个新的 Stream 对象(可以多次转换)，即允许操作像链条一样排列，变成一个管道

<img src="../../pics/java/javaN_2.png" align=left>

---

**生成 Stream Source 的多种方式**：

- 从 Collection 和数组
    - `Collection.stream()`
    - `Collection.parallelStream()`
    - `Arrays.stream(T array) or Stream.of()`

- 从 BufferedReader：`java.io.BufferedReader.lines()`

- 静态工厂
    - `java.util.stream.IntStream.range()`
    - `java.nio.file.Files.walk()`
- 自己构建：`java.util.Spliterator`

- 其它
    - `Random.ints()`
    - `BitSet.stream()`
    - `Pattern.splitAsStream(java.lang.CharSequence)`
    - `JarFile.stream()`

---

**流的两种操作类型**：

- **Intermediate** ：一个流可以后面跟随零个或多个 intermediate 操作

    > - 目的：打开流，做出某种程度的数据映射/过滤，然后返回一个新的流，交给下一个操作使用
    >
    > - 操作惰性化：即仅仅调用到这类方法，并没有真正开始流的遍历

- **Terminal** ：一个流只能有一个 terminal 操作，当这个操作执行后，流就无法再被操作

    > Terminal 操作的执行，才会真正开始流的遍历，并且会生成一个结果，或一个 side effect

-  **short-circuiting**：用于操作一个无限大的 Stream，并在有限时间内完成操作

    - 对于一个 intermediate 操作，若接受一个无限大的 Stream，则返回一个有限的新 Stream
    - 对于一个 terminal 操作，若接受一个无限大的 Stream，则能在有限的时间计算出结果

### (3) Stream 的构造与转换

常见构造 Stream 的方法：

```java
// 1. Individual values
Stream stream = Stream.of("a", "b", "c");
// 2. Arrays
String [] strArray = new String[] {"a", "b", "c"};
stream = Stream.of(strArray);
stream = Arrays.stream(strArray);
// 3. Collections
List<String> list = Arrays.asList(strArray);
stream = list.stream();
// 4. concat
Stream<Integer> firstStream = Stream.of(1, 2, 3);
Stream<Integer> secondStream = Stream.of(4, 5, 6);
Stream<Integer> resultingStream = Stream.concat(firstStream, secondStream);
```

> 三种基本数值型对应的包装类型 Stream：`IntStream、LongStream、DoubleStream`

- **数值流的构造**：

    ```java
    IntStream.of(new int[]{1, 2, 3}).forEach(System.out::println);
    IntStream.range(1, 3).forEach(System.out::println);
    IntStream.rangeClosed(1, 3).forEach(System.out::println);
    ```

- **流转换为其他数据结构**：

    ```java
    // 1. Array
    String[] strArray1 = stream.toArray(String[]::new);
    // 2. Collection
    List<String> list1 = stream.collect(Collectors.toList());
    List<String> list2 = stream.collect(Collectors.toCollection(ArrayList::new));
    Set set1 = stream.collect(Collectors.toSet());
    Stack stack1 = stream.collect(Collectors.toCollection(Stack::new));
    // 3. String
    String str = stream.collect(Collectors.joining()).toString();
    ```

### (4) Stream 的操作

#### 1. Intermediate

- `map/flatMap`：将 input Stream 的每一个元素，映射成 output Stream 的另一个元素

    > - 转换大写：
    >
    >     ```java
    >     List<String> output = wordList.stream().map(String::toUpperCase).collect(Collectors.toList());
    >     ```
    >
    > - 平方数：
    >
    >     ```java
    >     List<Integer> nums = Arrays.asList(1, 2, 3, 4);
    >     List<Integer> squareNums = nums.stream().map(n -> n * n).collect(Collectors.toList());
    >     ```
    >
    > - 一对多：flatMap 将 input Stream 层级结构扁平化，即将最底层元素抽出放到一起，最终 output Stream 都是直接的数字
    >
    >     ```java
    >     Stream<List<Integer>> inputStream = Stream.of(
    >     	Arrays.asList(1),
    >     	Arrays.asList(2, 3),
    >     	Arrays.asList(4, 5, 6)
    >     );
    >     Stream<Integer> outputStream = inputStream.flatMap(childList -> childList.stream()); //List 合并
    >     outputStream.forEach(System.out::print);
    >     //结果：123456
    >     ```

- `filter`：对原始 Stream 进行某项测试，通过测试的元素被留下来生成一个新 Stream

    > - 留下偶数：
    >
    >     ```java
    >     Integer[] sixNums = {1, 2, 3, 4, 5, 6};
    >     Integer[] evens = Stream.of(sixNums).filter(n -> n % 2 == 0).toArray(Integer[]::new);
    >     ```
    >
    > - 把单词挑出来：
    >
    >     ```java
    >     List<String> output = reader.lines()
    >         						.flatMap(line -> Stream.of(line.split(REGEXP)))
    >     							.filter(word -> word.length() > 0)
    >     							.collect(Collectors.toList());
    >     ```

- `peek`：对每个元素执行操作并**返回一个新的 Stream**，forEach 会销毁原有的 Stream

    > ```java
    > Stream.of("one", "two", "three", "four")
    > 		.filter(e -> e.length() > 3)
    > 		.peek(e -> System.out.println("Filtered value: " + e))
    >  		.map(String::toUpperCase)
    >  		.peek(e -> System.out.println("Mapped value: " + e))
    >  		.collect(Collectors.toList());
    > //结果
    > Filtered value: three
    > Mapped value: THREE
    > Filtered value: four
    > Mapped value: FOUR
    > ```

- `limit/skip`：limit 返回 Stream 的前面 n 个元素；skip 则是扔掉前 n 个元素

    > -  **limit 和 skip 对运行次数的影响** 
    >
    >     > 这是一个有 10，000 个元素的 Stream，但在 short-circuiting 操作 limit 和 skip 的作用下：
    >     >
    >     > - 管道中 map 操作指定的 getName() 方法的执行次数为 limit 所限定的 10 次
    >     > - 而最终返回结果在跳过前 3 个元素后只有后面 7 个返回
    >
    >     ```java
    >     public void testLimitAndSkip() {
    >         List<Person> persons = new ArrayList();
    >         for (int i = 1; i <= 10000; i++) {
    >             Person person = new Person(i, "name" + i);
    >             persons.add(person);
    >         }
    >         List<String> personList2 = persons.stream()
    >                 				.map(Person::getName).limit(10).skip(3).collect(Collectors.toList());
    >         System.out.println(personList2);
    >     }
    >     
    >     private class Person {
    >         public int no;
    >         private String name;
    >     
    >         public Person(int no, String name) {
    >             this.no = no;
    >             this.name = name;
    >         }
    >     
    >         public String getName() {
    >             System.out.println(name);
    >             return name;
    >         }
    >     }
    >     //结果
    >     name1
    >     name2
    >     name3
    >     name4
    >     name5
    >     name6
    >     name7
    >     name8
    >     name9
    >     name10
    >     [name4, name5, name6, name7, name8, name9, name10]
    >     ```
    >
    > - **limit 和 skip 对 sorted 后的运行次数无影响** 
    >
    >     > 注意：对于 parallel 的 Steam 管道，如果其元素有序，则 limit 操作的成本会比较大
    >
    >     ```java
    >     public void testLimitAndSkip() {
    >         List<Person> persons = new ArrayList();
    >         for (int i = 1; i <= 5; i++) {
    >             Person person = new Person(i, "name" + i);
    >             persons.add(person);
    >         }
    >         List<Person> personList = persons.stream()
    >                                          .sorted(Comparator.comparing(Person::getName)) //先排序再截断
    >                                          .limit(2)
    >                                          .collect(Collectors.toList());
    >         System.out.println(personList);
    >     }
    >     //结果
    >     name2
    >     name1
    >     name3
    >     name2
    >     name4
    >     name3
    >     name5
    >     name4
    >     [org.example.Main$Person@34c45dca, org.example.Main$Person@52cc8049]
    >     ```

- `sorted`：首先对 Stream 进行各类 map、filter、limit、skip 甚至 distinct 来减少元素数量后，再排序，能明显缩短执行时间

    >  上面代码的改进：不要求排序后再取值
    >
    > ```java
    > public void testLimitAndSkip() {
    >     List<Person> persons = new ArrayList();
    >     for (int i = 1; i <= 5; i++) {
    >         Person person = new Person(i, "name" + i);
    >         persons.add(person);
    >     }
    > 
    >     List<Person> personList = persons.stream()
    >         							 .limit(2) //先截断，再排序
    >         							 .sorted(Comparator.comparing(Person::getName))
    >         							 .collect(Collectors.toList());
    >     System.out.println(personList);
    > }
    > //结果
    > name2
    > name1
    > [stream.StreamDW$Person@6ce253f1, stream.StreamDW$Person@53d8d10a]
    > ```


#### 2. Terminal

- `forEach`：接收一个 Lambda 表达式，然后在 Stream 的每一个元素上执行该表达式

    > ```java
    > // Java 8
    > roster.stream()
    >       .filter(p -> p.getGender() == Person.Sex.MALE)
    >       .forEach(p -> System.out.println(p.getName()));
    > 
    > // Pre-Java 8
    > for (Person p : roster) {
    > 	if (p.getGender() == Person.Sex.MALE) {
    >         System.out.println(p.getName());
    > 	}
    > }
    > ```
    >
    > 注意：forEach 是 Terminal 操作，Input Steam 执行一次后，就会被销毁
    >
    > ```java
    > Stream<Integer> stream = Stream.of(1, 2, 3, 4);
    > stream.forEach(System.out::println);
    > stream.forEach(System.out::println); //第二次执行时，会报错
    > 
    > List<Integer> list = Arrays.asList(1, 2, 3, 4);
    > list.forEach(System.out::println);
    > list.forEach(System.out::println); //第二次执行时，不会报错
    > 
    > List<Integer> listStream = Arrays.asList(1, 2, 3, 4);
    > listStream.stream().forEach(System.out::println);
    > listStream.stream().forEach(System.out::println); //第二次执行时，不会报错
    > ```
    >
    > - forEach 不能修改自己的本地变量，也不能用 break 关键字提前结束循环，但可以用 return 跳过本次循环(类似 continue)

- `forEachOrdered`：在并行 Stream 中，严格按照顺序取数据

- `reduce`：提供一个起始值，然后依照运算规则，把 Stream 元素组合起来

    > ```java
    > // 字符串连接，concat = "ABCD"
    > String concat = Stream.of("A", "B", "C", "D").reduce("", String::concat);
    > 
    > // 求最小值，minValue = -3.0
    > double minValue = Stream.of(-1.5, 1.0, -3.0, -2.0).reduce(Double.MAX_VALUE, Double::min);
    > 
    > // 求和，sumValue = 10, 有起始值
    > int sumValue = Stream.of(1, 2, 3, 4).reduce(0, Integer::sum);
    > 
    > // 求和，sumValue = 10, 无起始值
    > sumValue = Stream.of(1, 2, 3, 4).reduce(Integer::sum).get();
    > 
    > // 过滤，字符串连接，concat = "ace"
    > concat = Stream.of("a", "B", "c", "D", "e", "F")
    >     			.filter(x -> x.compareTo("Z") > 0)
    >     			.reduce("", String::concat);
    > ```

- `max/min/distinct`：

    > - 找出最长一行的长度：
    >
    >     ```java
    >     BufferedReader br = new BufferedReader(new FileReader("SUService.log"));
    >     int longest = br.lines()
    >     				.mapToInt(String::length)
    >     				.max()
    >     				.getAsInt();
    >     br.close();
    >     System.out.println(longest);
    >     ```
    >
    > - 找出全文的单词，转小写，并排序
    >
    >     ```java
    >     List<String> words = br.lines()
    >     					   .flatMap(line -> Stream.of(line.split(" ")))
    >     					   .filter(word -> word.length() > 0)
    >      					   .map(String::toLowerCase)
    >     					   .distinct()
    >      					   .sorted()
    >      					   .collect(Collectors.toList());
    >     br.close();
    >     System.out.println(words);
    >     ```

- Match：

    - `anyMatch`：Stream 中只要有一个元素符合传入的 predicate，返回 true
    - `allMatch`：Stream 中全部元素符合传入的 predicate，返回 true
    - `noneMatch`：Stream 中没有一个元素符合传入的 predicate，返回 true

    > ```java
    > List<Person> persons = new ArrayList();
    > persons.add(new Person(1, "name" + 1, 10));
    > persons.add(new Person(2, "name" + 2, 21));
    > persons.add(new Person(3, "name" + 3, 34));
    > persons.add(new Person(4, "name" + 4, 6));
    > persons.add(new Person(5, "name" + 5, 55));
    > 
    > boolean isAllAdult = persons.stream().allMatch(p -> p.getAge() > 18);
    > System.out.println("All are adult? " + isAllAdult);
    > 
    > boolean isThereAnyChild = persons.stream().anyMatch(p -> p.getAge() < 12);
    > System.out.println("Any child? " + isThereAnyChild);
    > 
    > //结果
    > All are adult? false
    > Any child? true
    > ```

- `collect`：将流转换为其他形式，接收一个Collector接口的实现，用于汇总Stream元素

    >  <img src="../../pics/java/javaN_4.png" align=left>
    >
    >  <img src="../../pics/java/javaN_5.png" align=left>
    >
    >  ```java
    >  public void test3(){
    >      List<String> list = emps.stream().map(Employee::getName).collect(Collectors.toList());
    >      list.forEach(System.out::println);
    >      System.out.println("----------------------------------");
    >  
    >      Set<String> set = emps.stream().map(Employee::getName).collect(Collectors.toSet());
    >      set.forEach(System.out::println);
    >      System.out.println("----------------------------------");
    >  
    >      HashSet<String> hs = emps.stream()
    >          					 .map(Employee::getName)
    >          					 .collect(Collectors.toCollection(HashSet::new));
    >      hs.forEach(System.out::println);
    >  }
    >  
    >  public void test4(){
    >      Optional<Double> max = emps.stream()
    >          					   .map(Employee::getSalary)
    >          					   .collect(Collectors.maxBy(Double::compare));
    >      System.out.println(max.get());
    >  
    >      Optional<Employee> op = emps.stream()
    >          						.collect(Collectors.minBy((e1, e2) -> 
    >                          						Double.compare(e1.getSalary(), e2.getSalary())));
    >      System.out.println(op.get());
    >  
    >      Double sum = emps.stream().collect(Collectors.summingDouble(Employee::getSalary));
    >      System.out.println(sum);
    >  
    >      Double avg = emps.stream().collect(Collectors.averagingDouble(Employee::getSalary));
    >      System.out.println(avg);
    >  
    >      Long count = emps.stream().collect(Collectors.counting());
    >      System.out.println(count);
    >  
    >      DoubleSummaryStatistics dss = emps.stream()
    >          							  .collect(Collectors.summarizingDouble(Employee::getSalary));
    >      System.out.println(dss.getMax());
    >  }
    >  
    >  //分组
    >  public void test5(){
    >      Map<Status, List<Employee>> map = emps.stream()
    >  										  .collect(Collectors.groupingBy(Employee::getStatus));
    >      System.out.println(map);
    >  }
    >  
    >  //多级分组
    >  public void test6(){
    >      Map<Status, Map<String, List<Employee>>> map = emps.stream()
    >          		.collect(Collectors.groupingBy(Employee::getStatus, Collectors.groupingBy((e) -> {
    >                      if(e.getAge() >= 60)
    >                          return "老年";
    >                      else if(e.getAge() >= 35)
    >                          return "中年";
    >                      else
    >                          return "成年";
    >                  })));
    >      System.out.println(map);
    >  }
    >  
    >  //分区
    >  @Test
    >  public void test7(){
    >      Map<Boolean, List<Employee>> map = emps.stream()
    >          						.collect(Collectors.partitioningBy((e) -> e.getSalary() >= 5000));
    >      System.out.println(map);
    >  }
    >  
    >  public void test8(){
    >      String str = emps.stream()
    >          			 .map(Employee::getName)
    >          			 .collect(Collectors.joining("," , "----", "----"));
    >      System.out.println(str);
    >  }
    >  
    >  public void test9(){
    >      Optional<Double> sum = emps.stream()
    >          					   .map(Employee::getSalary)
    >          					   .collect(Collectors.reducing(Double::sum));
    >      System.out.println(sum.get());
    >  }
    >  ```

- `findFirst`：termimal 兼 short-circuiting 操作，总是返回 Stream 的第一个元素或空，返回值类型为 `Optional` 

- `findAny`：返回当前流中的任意元素

- `count`：返回流中元素总数

- `iterator`：返回一个迭代器

- `toArray`：返回数组(默认为 Object)

#### 3. Short-circuiting

- `anyMatch、 allMatch、 noneMatch、 findFirst、 findAny、 limit`



#### 4. 进阶操作

- **自己生成流**：通过实现 Supplier 接口控制流的生成

    > - **Stream.generate**：
    >
    >     - 生成 10 个随机数
    >
    >         ```java
    >         Random seed = new Random();
    >         Supplier<Integer> random = seed::nextInt;
    >         Stream.generate(random).limit(10).forEach(System.out::println);
    >         //Another way
    >         IntStream.generate(()->(int)(System.nanoTime() % 100)).limit(10).forEach(System.out::println);
    >         ```
    >
    >     - 自实现 Supplier
    >
    >         ```java
    >         Stream.generate(new PersonSupplier())
    >         	  .limit(10).
    >         	  .forEach(p -> System.out.println(p.getName() + ", " + p.getAge()));
    >         
    >         private class PersonSupplier implements Supplier<Person> {
    >          	private int index = 0;
    >          	private Random random = new Random();
    >             
    >             @Override
    >             public Person get() {
    >                 return new Person(index++, "StormTestUser" + index, random.nextInt(100));
    >             }
    >         }
    >         
    >         //结果
    >         StormTestUser1, 9
    >         StormTestUser2, 12
    >         StormTestUser3, 88
    >         StormTestUser4, 51
    >         StormTestUser5, 22
    >         StormTestUser6, 28
    >         StormTestUser7, 81
    >         StormTestUser8, 51
    >         StormTestUser9, 4
    >         StormTestUser10, 76
    >         ```
    >
    > - **Stream.iterate**：iterate 类似 reduce 操作，接受一个种子值，和一个 UnaryOperator(如：f)，然后种子值成为 Stream 的第一个元素、f(seed) 为第二个、f(f(seed)) 第三个，以此类推
    >
    >     ```java
    >     Stream.iterate(0, n -> n + 3).limit(10). forEach(x -> System.out.print(x + " "));
    >     //结果
    >     0 3 6 9 12 15 18 21 24 27
    >     ```

- **用 Collectors 进行 reduction 操作**

    > **groupingBy/partitioningBy**：
    >
    > - 按照年龄归组
    >
    >     ```java
    >     Map<Integer, List<Person>> personGroups = Stream.generate(new PersonSupplier())
    >      												.limit(100)
    >      												.collect(Collectors.groupingBy(Person::getAge));
    >     Iterator it = personGroups.entrySet().iterator();
    >     while (it.hasNext()) {
    >      	Map.Entry<Integer, List<Person>> persons = (Map.Entry) it.next();
    >      	System.out.println("Age " + persons.getKey() + " = " + persons.getValue().size());
    >     }
    >     //结果
    >     Age 0 = 2
    >     Age 1 = 2
    >     Age 5 = 2
    >     Age 8 = 1
    >     Age 9 = 1
    >     Age 11 = 2
    >     ......
    >     ```
    >
    > - 按照未成年人和成年人归组
    >
    >     ```java
    >     Map<Boolean, List<Person>> children = Stream.generate(new PersonSupplier())
    >      										.limit(100)
    >      										.collect(Collectors.partitioningBy(p -> p.getAge() < 18));
    >     System.out.println("Children number: " + children.get(true).size());
    >     System.out.println("Adult number: " + children.get(false).size());
    >     //结果
    >     Children number: 23
    >     Adult number: 77
    >     ```

### (5) 并行流与串行流

- **并行流**：把一个内容分成多个数据块，并用不同的线程分别处理每个数据块的流

- Stream API 可以声明性地通过 `parallel() 与sequential()` 在并行流与顺序流之间进行切换

- **Fork/Join 框架**：就是在必要的情况下，将一个大任务，进行拆分(fork)成若干个小任务（拆到不可再拆时），再将一个个的小任务运算的结果进行join 汇总.

  ![](../../pics/java/javaN_3.png)

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

## 5、接口中的默认方法与静态方法

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

## 6、新时间日期API

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

![](../../pics/java/javaN_7.png)

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

## 7、其他新特性

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

![](../../pics/java/javaN_6.png)