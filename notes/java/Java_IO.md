-  **按操作方式分类结构图**：

  ![](../../pics/io/io_4.png) 

- **按操作对象分类结构图**：

  ![](../../pics/io/io_5.png)

# 一、概览

Java 的 I/O 大概可以分成以下几类：

- **磁盘操作**：File
- **字节操作**：InputStream 和 OutputStream
- **字符操作**：Reader 和 Writer
- **对象操作**：Serializable
- **网络操作**：Socket
- **新的输入/输出**：NIO

![](../../pics/io/io_6.png) 

# 二、磁盘操作

- File 类可以用于表示文件和目录信息，但不能访问文件内容本身

- 如需访问文件内容本身，则需使用输入/输出流

```java
//递归地列出一个目录下所有文件
public static void listAllFiles(File dir) {
    if (dir == null || !dir.exists()) {
        return;
    }
    if (dir.isFile()) {
        System.out.println(dir.getName());
        return;
    }
    for (File file : dir.listFiles()) {
        listAllFiles(file);
    }
}
```

# 三、节点流(文件流)

## 1. 字节操作

- `FileInputStream `

  ```java
  public void testFileInputStream() { 
      FileInputStream fis = null;
      try {
          File file = new File("hello.txt");
          fis = new FileInputStream(file);
          byte[] b = new byte[5];// 读取到的数据要写入的数组
          int len;// 每次读入到byte中的字节的长度
          while ((len = fis.read(b)) != -1) {
              String str = new String(b, 0, len);
              System.out.print(str);
          }
      } catch (IOException e) {
          e.printStackTrace();
      } finally {
          if (fis != null) {
              try {
                  fis.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
      }
  }
  ```

- `FileOutputStream`

  ```java
  public void testFileOutputStream() {
      // 1.创建一个File对象，表明要写入的文件位置
      // 输出的物理文件可以不存在，当执行过程中，若不存在，会自动的创建;若存在，会将原有的文件覆盖
      File file = new File("hello2.txt");
      // 2.创建一个FileOutputStream对象，将file对象作为形参传递给FileOutputStream的构造器
      FileOutputStream fos = null;
      try {
          fos = new FileOutputStream(file);
          // 3.写入的操作
          fos.write(new String("I love China！").getBytes());
      } catch (Exception e) {
          e.printStackTrace();
      } finally {
          // 4.关闭输出流
          if (fos != null) {
              try {
                  fos.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
      }
  }
  ```

- `FileInputStream` 与 `FileOutputStream` 同时使用： 实现文件复制

  ```java
  // 从硬盘读取一个文件，并写入到另一个位置（相当于文件的复制）
  public void testFileInputOutputStream() {
      // 1.提供读入、写出的文件
      File file1 = new File("1.jpg");
      File file2 = new File("2.jpg");
      // 2.提供相应的流
      FileInputStream fis = null;
      FileOutputStream fos = null;
      try {
          fis = new FileInputStream(file1);
          fos = new FileOutputStream(file2);
          // 3.实现文件的复制
          byte[] b = new byte[20];
          int len;
          while ((len = fis.read(b)) != -1) {
              fos.write(b, 0, len);//写入
          }
      } catch (Exception e) {
          e.printStackTrace();
      } finally {
          if (fos != null) {
              try {
                  fos.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
          if (fis != null) {
              try {
                  fis.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
      }
  }
  ```

## 2. 字符操作 

- `FileReader` 操作： 

  ```java
  public void testFileReader(){
      FileReader fr = null;
      try {
          File file = new File("dbcp.txt");
          fr = new FileReader(file);
          char[] c = new char[24];
          int len;
          while((len = fr.read(c)) != -1){
              String str = new String(c, 0, len);
              System.out.print(str);
          }
      }catch (IOException e) {
          e.printStackTrace();
      }finally{
          if(fr != null){
              try {
                  fr.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
      }
  }
  ```

- `FileReader 与 FileWriter` 操作： 

  ```java
  //字符流只能处理文本文件，对于非文本文件（如：视频文件，音频文件，图片等）只能使用字节流
  public void testFileReaderWriter(){
      //输入流对应的文件src一定要存在，否则抛异常
      //输出流对应的文件dest可以不存在，执行过程中会自动创建
      FileReader fr = null;
      FileWriter fw = null;
      try{
          File src = new File("dbcp.txt");
          File dest = new File("dbcp1.txt");
          //2.
          fr = new FileReader(src);
          fw = new FileWriter(dest);
          //3.
          char[] c = new char[24];
          int len;
          while((len = fr.read(c)) != -1){
              fw.write(c, 0, len);
          }
      }catch(Exception e){
          e.printStackTrace();
      }finally{
          if(fw != null){
              try {
                  fw.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
          if(fr != null){
              try {
                  fr.close();
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
      }
  }
  ```

# 四、缓冲流(处理流)

## 1. 字节流

```java
//使用BufferedInputStream和BufferedOutputStream实现非文本文件的复制
public void testBufferedInputOutputStream(){
    BufferedInputStream bis = null;
    BufferedOutputStream bos = null;
    try {
        //1.提供读入、写出的文件
        File file1 = new File("1.jpg");
        File file2 = new File("2.jpg");
        //2.想创建相应的节点流：FileInputStream、FileOutputStream
        FileInputStream fis = new FileInputStream(file1);
        FileOutputStream fos = new FileOutputStream(file2);
        //3.将创建的节点流的对象作为形参传递给缓冲流的构造器中
        bis = new BufferedInputStream(fis);
        bos = new BufferedOutputStream(fos);
        //4.具体的实现文件复制的操作
        byte[] b = new byte[1024];
        int len;
        while((len = bis.read(b)) != -1){
            bos.write(b, 0, len);
            bos.flush();
        }
    }catch (IOException e) {
        e.printStackTrace();
    }finally{
        //5.关闭相应的流
        if(bos != null){
            try {
                bos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(bis != null){
            try {
                bis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 2. 字符流

```java
public void testBufferedReader(){
    BufferedReader br = null;
    BufferedWriter bw = null;
    try {
        File file = new File("dbcp.txt");
        File file1 = new File("dbcp3.txt");
        
        FileReader fr = new FileReader(file);
        FileWriter fw = new FileWriter(file1);
        
        br = new BufferedReader(fr);
        bw = new BufferedWriter(fw);

        String str;
        while((str = br.readLine()) != null){
            bw.write(str + "\n");
            bw.flush();
        }
    }catch (IOException e) {
        e.printStackTrace();
    }finally{
        if(bw != null){
            try {
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(br != null){
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

# 五、其他流

## 1. 转换流

不管是磁盘还是网络传输，**最小的存储单元都是字节**： 

- `InputStreamReader` 实现从字节流解码成字符流
- `OutputStreamWriter` 实现字符流编码成为字节流

```java
/*
 * 如何实现字节流与字符流之间的转换：
 * 转换流：InputStreamReader  OutputStreamWriter
 * 编码：字符串  --->字节数组
 * 解码：字节数组--->字符串
 */
public void test1(){
    BufferedReader br = null;
    BufferedWriter bw = null;
    try {
        //解码
        File file = new File("dbcp.txt");
        FileInputStream fis = new FileInputStream(file);
        InputStreamReader isr = new InputStreamReader(fis, "GBK");
        br = new BufferedReader(isr);
        //编码
        File file1 = new File("dbcp4.txt");
        FileOutputStream fos = new FileOutputStream(file1);
        OutputStreamWriter osw = new OutputStreamWriter(fos, "GBK");
        bw = new BufferedWriter(osw);
        String str;
        while((str = br.readLine()) != null){
            bw.write(str);
            bw.newLine();
            bw.flush();
        }
    }catch (IOException e) {
        e.printStackTrace();
    }finally{
        if(bw != null){
            try {
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(br != null){
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 2. 标准输入输出流

- `System.in 和 System.out` 分别代表系统标准的输入和输出设备
   - `System.in` 的类型是 `InputStream`
   - `System.out` 的类型是 `PrintStream`
- 默认输入设备是键盘，输出设备是显示器
- 通过 System 类的 `setIn，setOut` 方法对默认设备进行改变
   - `public static void setIn(InputStream in)`
   - `public static void setOut(PrintStream out)`

```java
/*
 * 标准的输入输出流：
 * 标准的输出流：System.out
 * 标准的输入流：System.in
 * 
 * 题目：
 * 从键盘输入字符串，要求将读取到的整行字符串转成大写输出。然后继续进行输入操作，
 * 直至当输入“e”或者“exit”时，退出程序。
 */
public void test2(){
    BufferedReader br = null;
    try {
        InputStream is = System.in;
        InputStreamReader isr = new InputStreamReader(is);
        br = new BufferedReader(isr);
        String str;
        while(true){
            System.out.println("请输入字符串：");
            str = br.readLine();
            if(str.equalsIgnoreCase("e") || str.equalsIgnoreCase("exit")){
                break;
            }
            String str1 = str.toUpperCase();
            System.out.println(str1);
        }
    } catch (IOException e) {
        e.printStackTrace();
    }finally{
        if(br != null){
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 3. 打印流

**PrintStream(字节打印流)和PrintWriter(字符打印流)**

- 提供了一系列重载的print和println方法，用于多种数据类型的输出
- `PrintStream` 和 `PrintWriter` 的输出不会抛出异常
- `PrintStream` 和 `PrintWriter` 有自动flush功能
- `System.out` 返回的是 `PrintStream` 的实例

```java
// 打印流=> 字节流：PrintStream 字符流：PrintWriter
public void printStreamWriter() {
    FileOutputStream fos = null;
    try {
        fos = new FileOutputStream(new File("print.txt"));
    } catch (FileNotFoundException e) {
        e.printStackTrace();
    }
    // 创建打印输出流,设置为自动刷新模式(写入换行符或字节 '\n' 时都会刷新输出缓冲区)
    PrintStream ps = new PrintStream(fos, true);
    if (ps != null) { // 把标准输出流(控制台输出)改成文件
        System.setOut(ps);
    }
    for (int i = 0; i <= 255; i++) { // 输出ASCII字符
        System.out.print((char) i);
        if (i % 50 == 0) { // 每50个数据一行
            System.out.println(); // 换行
        }
    }
    ps.close();
}
```

## 4. 数据流

数据流有两个类：(用于读取和写出基本数据类型的数据）

- `DataInputStream` 和 `DataOutputStream` 

```java
//数据流：用来处理基本数据类型、String、字节数组的数据:DataInputStream DataOutputStream
public void testData(){
    DataOutputStream dos = null;
    try {
        FileOutputStream fos = new FileOutputStream("data.txt");
        dos = new DataOutputStream(fos);

        dos.writeUTF("我爱你，而你却不知道！");
        dos.writeBoolean(true);
        dos.writeLong(1432522344);
    }catch (IOException e) {
        e.printStackTrace();
    }finally{
        if(dos != null){
            try {
                dos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}

public void testData1(){
    DataInputStream dis = null;
    try{
        dis = new DataInputStream(new FileInputStream(new File("data.txt")));
        String str = dis.readUTF();
        System.out.println(str);
        boolean b = dis.readBoolean();
        System.out.println(b);
        long l = dis.readLong();
        System.out.println(l);
    }catch(Exception e){
        e.printStackTrace();
    }finally{
        if(dis != null){
            try {
                dis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## 5. 对象流

- `ObjectInputStream和OjbectOutputSteam`：用于存储和读取对象的处理流，可以把 Java 中的对象写入到数据源中，也能把对象从数据源中还原回来

- `ObjectOutputStream` 和 `ObjectInputStream` 不能序列化 `static和transient` 修饰的成员变量
  - **序列化(Serialize)**：用 `ObjectOutputStream` 类将一个Java对象写入IO流中
  - **反序列化(Deserialize)**：用 `ObjectInputStream` 类从IO流中恢复该Java对象

```java
public class TestObjectInputOutputStream {
	// 对象的反序列化过程：将硬盘中的文件通过ObjectInputStream转换为相应的对象
	public void testObjectInputStream() {
		ObjectInputStream ois = null;
		try {
			ois = new ObjectInputStream(new FileInputStream("person.txt"));
			Person p1 = (Person)ois.readObject();
			System.out.println(p1);
			Person p2 = (Person)ois.readObject();
			System.out.println(p2);
		}catch (Exception e) {
			e.printStackTrace();
		}finally{
			if(ois != null){
				try {
					ois.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	// 对象的序列化过程：将内存中的对象通过ObjectOutputStream转换为二进制流，存储在硬盘文件中
	public void testObjectOutputStream() {
		Person p1 = new Person("小米", 23,new Pet("花花"));
		Person p2 = new Person("红米", 21,new Pet("小花"));
		ObjectOutputStream oos = null;
		try {
			oos = new ObjectOutputStream(new FileOutputStream("person.txt"));
			oos.writeObject(p1);
			oos.flush();
			oos.writeObject(p2);
			oos.flush();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (oos != null) {
				try {
					oos.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}

/*
 * 要实现序列化的类： 
 * 1.要求此类是可序列化的：实现Serializable接口
 * 2.要求类的属性同样的要实现Serializable接口
 * 3.提供一个版本号：private static final long serialVersionUID
 * 4.使用static或transient修饰的属性，不可实现序列化
 */
class Person implements Serializable {
	private static final long serialVersionUID = 23425124521L;
	static String name;
	transient Integer age;
	Pet pet;
	public Person(String name, Integer age,Pet pet) {
		this.name = name;
		this.age = age;
		this.pet = pet;
	}
	@Override
	public String toString() {
		return "Person [name=" + name + ", age=" + age + ", pet=" + pet + "]";
	}
}
class Pet implements Serializable{
	String name;
	public Pet(String name){
		this.name = name;
	}
	@Override
	public String toString() {
		return "Pet [name=" + name + "]";
	}
}
```

## 6. RandomAccessFile 类

1. 构造器
   - `public RandomAccessFile(File file, String mode)` 
   - `public RandomAccessFile(String name, String mode)` 


2. 创建 RandomAccessFile 类实例需要指定一个 mode 参数，该参数指定 RandomAccessFile 的访问模式：
   - r：以只读方式打开
   - rw：打开以便读取和写入
   - rwd：打开以便读取和写入；同步文件内容的更新
   - rws：打开以便读取和写入；同步文件内容和元数据的更新

```java
/*
 * RandomAccessFile:支持随机访问
 * 1.既可以充当一个输入流，又可以充当一个输出流
 * 2.支持从文件的开头读取、写入
 * 3.支持从任意位置的读取、写入（插入）
 */
public class TestRandomAccessFile {
    //进行文件的读、写
	public void test1(){
		RandomAccessFile raf = null;
		try {			
            raf = new RandomAccessFile(new File("hello1.txt"),"rw");
			
			byte[] b = new byte[20];
			int len;
			while((len = raf1.read(b)) != -1){
				raf.write(b, 0, len);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(raf != null){
				try {
					raf.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
  
    //实现的实际上是覆盖的效果
	public void test2(){
		RandomAccessFile raf = null;
		try {
			raf = new RandomAccessFile(new File("hello1.txt"),"rw");
			raf.seek(4);
			raf.write("xy".getBytes());
		}catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(raf != null){
				try {
					raf.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
  
    //实现插入的效果：在d字符后面插入“xy”
	public void test3(){
		RandomAccessFile raf = null;
		try {
			raf = new RandomAccessFile(new File("hello1.txt"),"rw");
			raf.seek(4);
			String str = raf.readLine();
			raf.seek(4);
			raf.write("xy".getBytes());
			raf.write(str.getBytes());
		}catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(raf != null){
				try {
					raf.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
  
    //相较于test3，更通用
	public void test4(){
		RandomAccessFile raf = null;
		try {
			raf = new RandomAccessFile(new File("hello1.txt"),"rw");
			raf.seek(4);
			byte[] b = new byte[10];
			int len;
			StringBuffer sb = new StringBuffer();
			while((len = raf.read(b)) != -1){
				sb.append(new String(b,0,len));
			}
			raf.seek(4);
			raf.write("xy".getBytes());
			raf.write(sb.toString().getBytes());
		}catch (IOException e) {
			e.printStackTrace();
		}finally{
			if(raf != null){
				try {
					raf.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}	
}
```

# 六、对象操作

## 1. 序列化

- **序列化**： 把 Java 对象转换成平台无关的二进制流，从而持久地保存在磁盘上或通过网络传输

- **好处**： 可将任何实现了 `Serializable` 接口的对象转化为字节数据，使其在保存和传输时可被还原

- **实现**： 继承 `Serializable 或 Externalizable` 接口

  > 实现 Serializable 接口的类，需设置 `serialVersionUID` 用来表明类的不同版本间的兼容性

- **方式**： 
  - 序列化：`ObjectOutputStream.writeObject()`
  - 反序列化：`ObjectInputStream.readObject()`

## 2. Serializable 

- 实现 Serializable 接口来序列化类
- 若不实现它的话而进行序列化，会抛出异常

```java
public static void main(String[] args) throws IOException, ClassNotFoundException {
    A a1 = new A(123, "abc");
    String objectFile = "file/a1";

    ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(objectFile));
    objectOutputStream.writeObject(a1);
    objectOutputStream.close();

    ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(objectFile));
    A a2 = (A) objectInputStream.readObject();
    objectInputStream.close();
    System.out.println(a2);
}

private static class A implements Serializable {
    private int x;
    private String y;

    A(int x, String y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public String toString() {
        return "x = " + x + "  " + "y = " + y;
    }
}
```

## 3. transient 

transient 关键字可以使属性不被序列化

```java
private transient Object[] elementData;
```

# 七、网络操作

Java 中的网络支持：

- `InetAddress`：用于表示网络上的硬件资源，即 IP 地址
- `URL`：统一资源定位符
- `Sockets`：使用 TCP 协议实现网络通信
- `Datagram`：使用 UDP 协议实现网络通信

## 1. InetAddress

没有公有的构造函数，只能**通过静态方法来创建实例**

```java
InetAddress.getByName(String host);
InetAddress.getByAddress(byte[] address);
```

## 2. URL

可以直接从 URL 中读取字节流数据

```java
public static void main(String[] args) throws IOException {
    URL url = new URL("http://www.baidu.com");
    /* 字节流 */
    InputStream is = url.openStream();
    /* 字符流 */
    InputStreamReader isr = new InputStreamReader(is, "utf-8");
    /* 提供缓存功能 */
    BufferedReader br = new BufferedReader(isr);

    String line;
    while ((line = br.readLine()) != null) {
        System.out.println(line);
    }
    br.close();
}
```

## 3. Sockets

- `ServerSocket`：服务器端类
- `Socket`：客户端类

服务器和客户端通过 InputStream 和 OutputStream 进行输入输出

![](../../pics/io/io_7.png)

## 4. Datagram

- `DatagramSocket`：通信类
- `DatagramPacket`：数据包类

# 八、NIO

- [Java NIO 浅析](https://tech.meituan.com/nio.html)
- [IBM: NIO 入门](https://www.ibm.com/developerworks/cn/education/java/j-nio/j-nio.html) 

## 1. 流与块

- I/O 与 NIO 区别： 数据打包和传输的方式，**I/O 以流的方式处理数据，而 NIO 以块的方式处理数据**

  - 面向流的 I/O 一次处理一个字节数据：一个输入流产生一个字节数据，一个输出流消费一个字节数据

  - 面向块的 I/O 一次处理一个数据块： 按块处理数据比按流处理数据要快得多

## 2. 通道与缓冲区

### 1. 缓冲区(Buffer)

> Buffer 负责存储

- **通道间进行读写数据，要先经过缓冲区**

- **缓冲区提供了对数据的结构化访问，而且还可以跟踪系统的读/写进程**

- **缓冲区实质上是一个数组**

#### 1. 缓冲区(Buffer)简介

在 Java NIO 中，缓冲区负责数据的存取，用于存储不同数据类型的数据

 * 根据数据类型不同（boolean 除外），提供了相应类型的缓冲区：
 * ByteBuffer
 * CharBuffer
 * ShortBuffer
 * IntBuffer
 * LongBuffer
 * FloatBuffer
 * DoubleBuffer

上述缓冲区的管理方式几乎一致，**通过 allocate() 获取缓冲区**

#### 2. 缓冲区存取数据的核心方法

 * `put()`: 存入数据到缓冲区中
 * `get()`: 获取缓冲区中的数据

```java
public void test2(){
    String str = "abcde";

    ByteBuffer buf = ByteBuffer.allocate(1024);
    buf.put(str.getBytes());
    buf.flip();

    byte[] dst = new byte[buf.limit()];
    buf.get(dst, 0, 2);
    System.out.println(new String(dst, 0, 2));
    System.out.println(buf.position());
    //mark() : 标记
    buf.mark();
    buf.get(dst, 2, 2);
    System.out.println(new String(dst, 2, 2));
    System.out.println(buf.position());
    //reset() : 恢复到 mark 的位置
    buf.reset();
    System.out.println(buf.position());
    //判断缓冲区中是否还有剩余数据
    if(buf.hasRemaining()){
        //获取缓冲区中可以操作的数量
        System.out.println(buf.remaining());
    }
}
```

#### 3. 缓冲区状态变量

- `capacity`：表示 Buffer 最大数据容量，缓冲区容量不能为负，并且创建后不能更改
- `position`：当前已经读写的字节数，下一个要读取或写入的数据索引
- `limit`：还可以读写的字节数
- `标记(mark)与重置(reset)`：标记是一个索引，通过 Buffer 中的 mark() 方法指定 Buffer 中一个特定的position，之后可以通过调用 reset() 方法恢复到这个 position

>  标记、位置、限制、容量遵守以下不变式：0<=mark<=position<=limit<=capacity

状态变量的改变过程举例：

① 新建一个大小为 8 个字节的缓冲区，此时 position 为 0，而 limit = capacity = 8。capacity 变量不会改变

![](../../pics/io/io_8.png)

② 从输入通道中读取 5 个字节数据写入缓冲区中，此时 position 移动设置为 5，limit 保持不变

![](../../pics/io/io_9.png)

③ 在将缓冲区的数据写到输出通道之前，需要先调用` flip() `方法切换到读取数据模式，这个方法将 limit 设置为当前 position，并将 position 设置为 0

![](../../pics/io/io_10.png)

④ 调用`get()`方法从缓冲区中取 4 个字节到输出缓冲中，此时 position 设为 5

![](../../pics/io/io_11.png)

⑤ 最后需要调用 `clear() ` 方法来清空缓冲区，此时 position 和 limit 都被设置为最初位置。但是缓冲区中的数据依然存在，但数据处于“被遗忘”状态

![](../../pics/io/io_12.png)

```java
public void test1(){
    String str = "abcde";

    //1. 分配一个指定大小的缓冲区
    ByteBuffer buf = ByteBuffer.allocate(1024);

    System.out.println("-----------------allocate()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());

    //2. 利用 put() 存入数据到缓冲区中
    buf.put(str.getBytes());

    System.out.println("-----------------put()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());

    //3. 切换读取数据模式
    buf.flip();

    System.out.println("-----------------flip()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());

    //4. 利用 get() 读取缓冲区中的数据
    byte[] dst = new byte[buf.limit()];
    buf.get(dst);
    System.out.println(new String(dst, 0, dst.length));

    System.out.println("-----------------get()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());

    //5. rewind() : 可重复读，即切换读取数据模式
    buf.rewind();

    System.out.println("-----------------rewind()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());

    //6. clear() : 清空缓冲区. 但是缓冲区中的数据依然存在，但是处于“被遗忘”状态
    buf.clear();

    System.out.println("-----------------clear()----------------");
    System.out.println(buf.position());
    System.out.println(buf.limit());
    System.out.println(buf.capacity());
    //处于“被遗忘”状态的数据
    System.out.println((char)buf.get());
}
```

#### 4. 直接与非直接缓冲区

- **非直接缓冲区**：通过 allocate() 方法分配缓冲区，将缓冲区建立在 JVM 的内存中
- **直接缓冲区**：通过 allocateDirect() 方法分配直接缓冲区，将缓冲区建立在物理内存中，可以提高效率

![](../../pics/io/nio_13.png) 

![](../../pics/io/nio_14.png)

```java
public void test3(){
    //分配直接缓冲区
    ByteBuffer buf = ByteBuffer.allocateDirect(1024);
    System.out.println(buf.isDirect());//判断是否为直接缓冲区
}
```

### 2. 通道(Channel)

>  Channel 负责传输

- **通道**： 表示打开到 IO 设备(例如：文件、套接字)的连接，通道 Channel 是对原 I/O 包中的流的模拟，可以通过它读取和写入数据
- **作用**： 若需要使用 NIO 系统，需要获取用于连接 IO 设备的通道以及用于容纳数据的缓冲区，然后操作缓冲区，对数据进行处理

- **通道与流的不同**： 流只能在一个方向上移动(一个流必须是 InputStream 或者 OutputStream 的子类)，而通道是双向的，可以用于读、写或者同时用于读写

#### 1. 通道（Channel）简介

- 用于源节点与目标节点的连接
- 在 Java NIO 中负责缓冲区中数据的传输，Channel 本身不存储数据，因此需要配合缓冲区进行传输

#### 2. 通道的主要实现类

java.nio.channels.Channel 接口：
*   `FileChannel`： 用于读取、写入、映射和操作文件的通道
 *   `SocketChannel`： 通过 TCP 读写网络中的数据
 *   `ServerSocketChannel`： 可以监听新进来的 TCP 连接，对每一个新进来的连接都会创建一个SocketChannel
 *   `DatagramChannel`： 通过 UDP 读写网络中的数据通道

#### 3. 获取通道

- 方法： `getChannel()`

```java
//利用通道完成文件的复制（非直接缓冲区）
public void test1(){//10874-10953
    long start = System.currentTimeMillis();

    FileInputStream fis = null;
    FileOutputStream fos = null;
    
    FileChannel inChannel = null;
    FileChannel outChannel = null;
    try {
        fis = new FileInputStream("1.mkv");
        fos = new FileOutputStream("2.mkv");
				//① 获取通道
        inChannel = fis.getChannel();
        outChannel = fos.getChannel();
        //② 分配指定大小的缓冲区
        ByteBuffer buf = ByteBuffer.allocate(1024);
        //③ 将通道中的数据存入缓冲区中
        while(inChannel.read(buf) != -1){
            buf.flip(); //切换读取数据的模式
            //④ 将缓冲区中的数据写入通道中
            outChannel.write(buf);
            buf.clear(); //清空缓冲区
        }
    } catch (IOException e) {
        e.printStackTrace();
    } finally {
        if(outChannel != null){
            try {
                outChannel.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(inChannel != null){
            try {
                inChannel.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(fos != null){
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(fis != null){
            try {
                fis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    long end = System.currentTimeMillis();
    System.out.println("耗费时间为：" + (end - start));
}

//使用直接缓冲区完成文件的复制(内存映射文件)
public void test2() throws IOException{//2127-1902-1777
    long start = System.currentTimeMillis();

    FileChannel inChannel = FileChannel.open(Paths.get("1.mkv"), StandardOpenOption.READ);
    FileChannel outChannel = FileChannel.open(Paths.get("2.mkv"), StandardOpenOption.WRITE, 
                        												StandardOpenOption.READ, StandardOpenOption.CREATE);

    //内存映射文件
    MappedByteBuffer inMappedBuf = inChannel.map(MapMode.READ_ONLY, 0, inChannel.size());
    MappedByteBuffer outMappedBuf = outChannel.map(MapMode.READ_WRITE, 0, inChannel.size());

    //直接对缓冲区进行数据的读写操作
    byte[] dst = new byte[inMappedBuf.limit()];
    inMappedBuf.get(dst);
    outMappedBuf.put(dst);

    inChannel.close();
    outChannel.close();

    long end = System.currentTimeMillis();
    System.out.println("耗费时间为：" + (end - start));
}
```

#### 4. 通道之间的数据传输

 * `transferFrom()`：将数据从源通道传输到其他 Channel 中
 * `transferTo()`：将数据从源通道传输到其他 Channel 中

```java
//通道之间的数据传输(直接缓冲区)
public void test3() throws IOException{
    FileChannel inChannel = FileChannel.open(Paths.get("1.mkv"), StandardOpenOption.READ);
    FileChannel outChannel = FileChannel.open(Paths.get("2.mkv"), StandardOpenOption.WRITE, 
                         												StandardOpenOption.READ, StandardOpenOption.CREATE);

		inChannel.transferTo(0, inChannel.size(), outChannel);
    outChannel.transferFrom(inChannel, 0, inChannel.size());

    inChannel.close();
    outChannel.close();
}
```

#### 5. 分散(Scatter)与聚集(Gather)

 * **分散读取**：将通道的数据分散到多个缓冲区，按缓冲区顺序，从 Channel 中读取的数据依次将 Buffer 填满
 * **聚集写入**：将多个缓冲区的数据聚集到通道中，按缓冲区顺序，写入 position 和 limit 间的数据到 Channel

```java
//分散和聚集
public void test4() throws IOException{
    RandomAccessFile raf1 = new RandomAccessFile("1.txt", "rw");
    //1. 获取通道
    FileChannel channel1 = raf1.getChannel();
    //2. 分配指定大小的缓冲区
    ByteBuffer buf1 = ByteBuffer.allocate(100);
    ByteBuffer buf2 = ByteBuffer.allocate(1024);
    //3. 分散读取
    ByteBuffer[] bufs = {buf1, buf2};
    channel1.read(bufs);
    for (ByteBuffer byteBuffer : bufs) {
        byteBuffer.flip();
    }
    System.out.println(new String(bufs[0].array(), 0, bufs[0].limit()));
    System.out.println("-----------------");
    System.out.println(new String(bufs[1].array(), 0, bufs[1].limit()));
    //4. 聚集写入
    RandomAccessFile raf2 = new RandomAccessFile("2.txt", "rw");
    FileChannel channel2 = raf2.getChannel();
    channel2.write(bufs);
}
```

#### 6. 字符集：Charset

 * 编码：字符串 -> 字节数组
 * 解码：字节数组  -> 字符串

```java
//字符集
public void test5(){
    Map<String, Charset> map = Charset.availableCharsets();
    Set<Entry<String, Charset>> set = map.entrySet();
    for (Entry<String, Charset> entry : set) {
        System.out.println(entry.getKey() + "=" + entry.getValue());
    }
}

public void test6() throws IOException{
    Charset cs1 = Charset.forName("GBK");
    //获取编码器
    CharsetEncoder ce = cs1.newEncoder();
    //获取解码器
    CharsetDecoder cd = cs1.newDecoder();

    CharBuffer cBuf = CharBuffer.allocate(1024);
    cBuf.put("中国威武！");
    cBuf.flip();
    //编码
    ByteBuffer bBuf = ce.encode(cBuf);
    for (int i = 0; i < 10; i++) {
        System.out.println(bBuf.get());
    }
    //解码
    bBuf.flip();
    CharBuffer cBuf2 = cd.decode(bBuf);
    System.out.println(cBuf2.toString());

    System.out.println("------------------------------------------------------");

    //Charset cs2 = Charset.forName("UTF-8");//会显示乱码
    Charset cs2 = Charset.forName("GBK");
    bBuf.flip();
    CharBuffer cBuf3 = cs2.decode(bBuf);
    System.out.println(cBuf3.toString());
}
```

## 3. 文件 NIO 实例

以下展示了使用 NIO 快速复制文件的实例：

```java
public static void fastCopy(String src, String dist) throws IOException {
    /* 获得源文件的输入字节流 */
    FileInputStream fin = new FileInputStream(src);
    /* 获取输入字节流的文件通道 */
    FileChannel fcin = fin.getChannel();
    /* 获取目标文件的输出字节流 */
    FileOutputStream fout = new FileOutputStream(dist);
    /* 获取输出字节流的文件通道 */
    FileChannel fcout = fout.getChannel();
    /* 为缓冲区分配 1024 个字节 */
    ByteBuffer buffer = ByteBuffer.allocateDirect(1024);
    while (true) {
        /* 从输入通道中读取数据到缓冲区中 */
        int r = fcin.read(buffer);
        /* read() 返回 -1 表示 EOF */
        if (r == -1) {
            break;
        }
        /* 切换读写 */
        buffer.flip();
        /* 把缓冲区的内容写入输出文件中 */
        fcout.write(buffer);
        /* 清空缓冲区 */
        buffer.clear();
    }
}
```

## 4. 阻塞与非阻塞

- **传统 IO 流是阻塞式的**： 当一个线程调用 read() 或write() 时，该线程被阻塞，直到有数据被读取或写入
- **Java NIO 是非阻塞模式的**： 当线程从某通道进行读写数据时，若没有数据可用，该线程可以进行其他任务

#### 1. 使用 NIO 完成网络通信的三个核心

1. 通道(Channel)：负责连接，java.nio.channels.Channel 接口之 `SelectableChannel`：
   - SocketChannel
   - ServerSocketChannel
   - DatagramChannel
   - Pipe.SinkChannel
   - Pipe.SourceChannel


2. 缓冲区(Buffer)：负责数据的存取


3. 选择器(Selector)：是 SelectableChannel 的多路复用器，用于监控 SelectableChannel 的 IO 状况

#### 2. 阻塞 IO

```java
public class TestBlockingNIO {
    //客户端
    public void client() throws IOException{
        //1. 获取通道
        SocketChannel sChannel = SocketChannel.open(new InetSocketAddress("127.0.0.1", 9898));
        FileChannel inChannel = FileChannel.open(Paths.get("1.jpg"), StandardOpenOption.READ);
        //2. 分配指定大小的缓冲区
        ByteBuffer buf = ByteBuffer.allocate(1024);
        //3. 读取本地文件，并发送到服务端
        while(inChannel.read(buf) != -1){
            buf.flip();
            sChannel.write(buf);
            buf.clear();
        }
        //4. 关闭通道
        inChannel.close();
        sChannel.close();
    }

    //服务端
    public void server() throws IOException{
        //1. 获取通道
        ServerSocketChannel ssChannel = ServerSocketChannel.open();
        FileChannel outChannel = FileChannel.open(Paths.get("2.jpg"), 
                             												StandardOpenOption.WRITE, StandardOpenOption.CREATE);
        //2. 绑定连接
        ssChannel.bind(new InetSocketAddress(9898));
        //3. 获取客户端连接的通道
        SocketChannel sChannel = ssChannel.accept();
        //4. 分配指定大小的缓冲区
        ByteBuffer buf = ByteBuffer.allocate(1024);
        //5. 接收客户端的数据，并保存到本地
        while(sChannel.read(buf) != -1){
            buf.flip();
            outChannel.write(buf);
            buf.clear();
        }
        //6. 关闭通道
        sChannel.close();
        outChannel.close();
        ssChannel.close();
    }
}
```

```java
//接受反馈
public class TestBlockingNIO2 {
	//客户端
	public void client() throws IOException{
		SocketChannel sChannel = SocketChannel.open(new InetSocketAddress("127.0.0.1", 9898));
		FileChannel inChannel = FileChannel.open(Paths.get("1.jpg"), StandardOpenOption.READ);
		ByteBuffer buf = ByteBuffer.allocate(1024);
		while(inChannel.read(buf) != -1){
			buf.flip();
			sChannel.write(buf);
			buf.clear();
		}
      	//告诉服务端发送结束
		sChannel.shutdownOutput();
		//接收服务端的反馈
		int len = 0;
		while((len = sChannel.read(buf)) != -1){
			buf.flip();
			System.out.println(new String(buf.array(), 0, len));
			buf.clear();
		}
		inChannel.close();
		sChannel.close();
	}
	
	//服务端
	public void server() throws IOException{
		ServerSocketChannel ssChannel = ServerSocketChannel.open();
		FileChannel outChannel = FileChannel.open(Paths.get("2.jpg"), 
                             										StandardOpenOption.WRITE, StandardOpenOption.CREATE);
		ssChannel.bind(new InetSocketAddress(9898));
		SocketChannel sChannel = ssChannel.accept();
		ByteBuffer buf = ByteBuffer.allocate(1024);
		while(sChannel.read(buf) != -1){
			buf.flip();
			outChannel.write(buf);
			buf.clear();
		}
		//发送反馈给客户端
		buf.put("服务端接收数据成功".getBytes());
		buf.flip();
		sChannel.write(buf);
		
		sChannel.close();
		outChannel.close();
		ssChannel.close();
	}
}
```

#### 3. 非阻塞 IO

`SocketChannel`：是一个可以监听新进来的 `TCP` 连接的通道

```java
public class TestNonBlockingNIO {
	//客户端
	public void client() throws IOException{
		//1. 获取通道
		SocketChannel sChannel = SocketChannel.open(new InetSocketAddress("127.0.0.1", 9898));
		//2. 切换非阻塞模式
		sChannel.configureBlocking(false);
		//3. 分配指定大小的缓冲区
		ByteBuffer buf = ByteBuffer.allocate(1024);
		//4. 发送数据给服务端
		Scanner scan = new Scanner(System.in);
		while(scan.hasNext()){
			String str = scan.next();
			buf.put((new Date().toString() + "\n" + str).getBytes());
			buf.flip();
			sChannel.write(buf);
			buf.clear();
		}
		//5. 关闭通道
		sChannel.close();
	}

	//服务端
	public void server() throws IOException{
		//1. 获取通道
		ServerSocketChannel ssChannel = ServerSocketChannel.open();
		//2. 切换非阻塞模式
		ssChannel.configureBlocking(false);
		//3. 绑定连接
		ssChannel.bind(new InetSocketAddress(9898));
		//4. 获取选择器
		Selector selector = Selector.open();
		//5. 将通道注册到选择器上, 并且指定“监听接收事件”
		ssChannel.register(selector, SelectionKey.OP_ACCEPT);
		//6. 轮询式的获取选择器上已经“准备就绪”的事件
		while(selector.select() > 0){
			//7. 获取当前选择器中所有注册的“选择键(已就绪的监听事件)”
			Iterator<SelectionKey> it = selector.selectedKeys().iterator();
			while(it.hasNext()){
				//8. 获取准备“就绪”的事件
				SelectionKey sk = it.next();
				//9. 判断具体是什么事件准备就绪
				if(sk.isAcceptable()){
					//10. 若“接收就绪”，获取客户端连接
					SocketChannel sChannel = ssChannel.accept();
					//11. 切换非阻塞模式
					sChannel.configureBlocking(false);
					//12. 将该通道注册到选择器上
					sChannel.register(selector, SelectionKey.OP_READ);
				}else if(sk.isReadable()){
					//13. 获取当前选择器上“读就绪”状态的通道
					SocketChannel sChannel = (SocketChannel) sk.channel();
					//14. 读取数据
					ByteBuffer buf = ByteBuffer.allocate(1024);
					int len = 0;
					while((len = sChannel.read(buf)) > 0 ){
						buf.flip();
						System.out.println(new String(buf.array(), 0, len));
						buf.clear();
					}
				}
				//15. 取消选择键 SelectionKey
				it.remove();
			}
		}
	}
}
```

`DatagramChannel`：一个能收发 `UDP` 包的通道

```java
public class TestNonBlockingNIO2 {
	//发送
	public void send() throws IOException{
		DatagramChannel dc = DatagramChannel.open();
		dc.configureBlocking(false);
		ByteBuffer buf = ByteBuffer.allocate(1024);
		Scanner scan = new Scanner(System.in);
		while(scan.hasNext()){
			String str = scan.next();
			buf.put((new Date().toString() + ":\n" + str).getBytes());
			buf.flip();
			dc.send(buf, new InetSocketAddress("127.0.0.1", 9898));
			buf.clear();
		}
		dc.close();
	}
	//接收
	public void receive() throws IOException{
		DatagramChannel dc = DatagramChannel.open();
		dc.configureBlocking(false);
		dc.bind(new InetSocketAddress(9898));
		Selector selector = Selector.open();
		dc.register(selector, SelectionKey.OP_READ);
		while(selector.select() > 0){
			Iterator<SelectionKey> it = selector.selectedKeys().iterator();
			while(it.hasNext()){
				SelectionKey sk = it.next();
				if(sk.isReadable()){
					ByteBuffer buf = ByteBuffer.allocate(1024);
					dc.receive(buf);
					buf.flip();
					System.out.println(new String(buf.array(), 0, buf.limit()));
					buf.clear();
				}
			}
			it.remove();
		}
	}
}
```

#### 4. 管道(Pipe) 

- Java NIO 管道是两个线程间的单向数据连接
- Pipe有一个 `source` 通道和一个 `sink通道` ，数据会被写到 sink 通道，从 source 通道读取

```java
public class TestPipe {
	public void test() throws IOException{
		//1. 获取管道
		Pipe pipe = Pipe.open();
		ByteBuffer buf = ByteBuffer.allocate(1024);
		//2. 将缓冲区中的数据写入管道
		Pipe.SinkChannel sinkChannel = pipe.sink();
		buf.put("通过单向管道发送数据".getBytes());
		buf.flip();
		sinkChannel.write(buf);
		//3. 读取缓冲区中的数据
		Pipe.SourceChannel sourceChannel = pipe.source();
		buf.flip();
		int len = sourceChannel.read(buf);
		System.out.println(new String(buf.array(), 0, len));
		
		sourceChannel.close();
		sinkChannel.close();
	}
}
```

## 5. NIO.2 

- 增强了对文件处理和文件系统特性的支持

### 1. 自动资源管理

- **自动资源管理(Automatic Resource Management, ARM)**：
  - 以 try 语句的扩展版为基础，当 try 代码块结束时，**自动释放资源**
  - 当不再需要文件(或其他资源)时，可以防止无意中忘记释放它们
  - 需要关闭的资源，必须实现 AutoCloseable 接口或其子接口 Closeable

```java
//自动资源管理：自动关闭实现 AutoCloseable 接口的资源
public void test8(){
    try(
        FileChannel inChannel = FileChannel.open(Paths.get("1.jpg"), StandardOpenOption.READ);
        FileChannel outChannel = FileChannel.open(Paths.get("2.jpg"), 								
                                                  StandardOpenOption.WRITE, StandardOpenOption.CREATE)){
        		ByteBuffer buf = ByteBuffer.allocate(1024);
        		inChannel.read(buf);
    }catch(IOException e){

    }
}
```

### 2. Path 与Paths

- `Path`： 代表一个平台无关的平台路径，描述了目录结构中文件的位置

- Paths 提供的 get() 方法用来获取 Path 对象：
  - `Path get(String first, String … more)`： 用于将多个字符串串连成路径
- Path常用方法：
  - boolean endsWith(String path) : 判断是否以path 路径结束
  - boolean startsWith(String path) : 判断是否以path 路径开始
  - boolean isAbsolute() : 判断是否是绝对路径
  - Path getFileName() : 返回与调用Path 对象关联的文件名
  - Path getName(int idx) : 返回的指定索引位置idx 的路径名称
  - int getNameCount() : 返回Path 根目录后面元素的数量
  - Path getParent() ：返回Path对象包含整个路径，不包含Path 对象指定的文件路径
  - Path getRoot() ：返回调用Path 对象的根路径
  - Path resolve(Path p) :将相对路径解析为绝对路径
  - Path toAbsolutePath() : 作为绝对路径返回调用Path 对象
  - String toString() ：返回调用Path 对象的字符串表示形式

### 3. Files 类

- `Files`： 用于操作文件或目录的工具类

1. Files常用方法：
   - Path copy(Path src, Path dest, CopyOption … how) : 文件的复制
   - Path createDirectory(Path path, FileAttribute<?> … attr) : 创建一个目录
   - Path createFile(Path path, FileAttribute<?> … arr) : 创建一个文件
   - void delete(Path path) : 删除一个文件
   - Path move(Path src, Path dest, CopyOption…how) : 将src 移动到dest 位置
   - long size(Path path) : 返回path 指定文件的大小


2. Files常用方法：用于判断
   - boolean exists(Path path, LinkOption … opts) : 判断文件是否存在
   - boolean isDirectory(Path path, LinkOption … opts) : 判断是否是目录
   - boolean isExecutable(Path path) : 判断是否是可执行文件
   - boolean isHidden(Path path) : 判断是否是隐藏文件
   - boolean isReadable(Path path) : 判断文件是否可读
   - boolean isWritable(Path path) : 判断文件是否可写
   - boolean notExists(Path path, LinkOption … opts) : 判断文件是否不存在
   - public static \<A extends BasicFileAttributes> A readAttributes(Path path,Class\<A> type,LinkOption... options) : 获取与path 指定的文件相关联的属性。
3. Files常用方法：用于操作内容
   - SeekableByteChannel newByteChannel(Path path, OpenOption…how) : 获取与指定文件的连接，how 指定打开方式
   - DirectoryStream newDirectoryStream(Path path) : 打开path 指定的目录
   - InputStream newInputStream(Path path, OpenOption…how): 获取InputStream 对象
   - OutputStream newOutputStream(Path path, OpenOption…how) : 获取OutputStream 对象

## 6. 选择器

- **选择器(Selector)**： 是 SelectableChannle 对象的多路复用器，Selector 可以同时监控多个SelectableChannel 的 IO 状况，即利用 Selector 可使一个单独的线程管理多个Channel，Selector 是非阻塞IO 的核心

- **NIO 实现了 IO 多路复用中的 Reactor 模型**：

  - 一个线程 Thread 使用一个选择器 Selector 通过轮询的方式去监听多个通道 Channel 上的事件，从而让一个线程处理多个事件

  - 通过配置监听的通道 Channel 为非阻塞，则当 Channel 上的 IO 事件还未到达时，不会一直等待，而是继续轮询其它 Channel，找到 IO 事件已经到达的 Channel 执行

  - 因为创建和切换线程的开销很大，因此使用一个线程来处理多个事件，可以提高 IO 密集型应用的性能

  > 只有套接字 Channel 才能配置为非阻塞，而 FileChannel 不能，为 FileChannel 配置非阻塞也没有意义

![](../../pics/io/io_15.png)

### 1. 创建选择器

```java
Selector selector = Selector.open();
```

### 2. 将通道注册到选择器上

```java
ServerSocketChannel ssChannel = ServerSocketChannel.open();
ssChannel.configureBlocking(false);//通道必须配置为非阻塞模式
ssChannel.register(selector, SelectionKey.OP_ACCEPT);
```

将通道注册到选择器上时，需要指定注册的具体事件：

- SelectionKey.OP_CONNECT：连接
- SelectionKey.OP_ACCEPT：接收
- SelectionKey.OP_READ：读
- SelectionKey.OP_WRITE：写

```java
//SelectionKey 的定义
public static final int OP_READ = 1 << 0;
public static final int OP_WRITE = 1 << 2;
public static final int OP_CONNECT = 1 << 3;
public static final int OP_ACCEPT = 1 << 4;
```

可以看出每个事件可以被当成一个位域，从而组成事件集整数。例如：

```java
int interestSet = SelectionKey.OP_READ | SelectionKey.OP_WRITE;
```

- `SelectionKey`：表示 SelectableChannel 和 Selector 间的注册关系，每次向选择器注册通道时会选择一个事件(选择键)，选择键包含两个表示为整数值的操作集，操作集的每一位都表示该键的通道所支持的一类可选择操作

### 3. 监听事件

使用 select() 来监听到达的事件，它会一直阻塞到有至少一个事件到达

```java
int num = selector.select();
```

### 4. 获取到达的事件

```java
Set<SelectionKey> keys = selector.selectedKeys();
Iterator<SelectionKey> keyIterator = keys.iterator();
while (keyIterator.hasNext()) {
    SelectionKey key = keyIterator.next();
    if (key.isAcceptable()) {
        // ...
    } else if (key.isReadable()) {
        // ...
    }
    keyIterator.remove();
}
```

### 5. 事件循环

因为一次 select() 调用不能处理完所有事件，且服务器端可能需要一直监听事件，因此服务器端处理事件的代码一般会放在一个死循环内

```java
while (true) {
    int num = selector.select();
    Set<SelectionKey> keys = selector.selectedKeys();
    Iterator<SelectionKey> keyIterator = keys.iterator();
    while (keyIterator.hasNext()) {
        SelectionKey key = keyIterator.next();
        if (key.isAcceptable()) {
            // ...
        } else if (key.isReadable()) {
            // ...
        }
        keyIterator.remove();
    }
}
```

### 6. Selector 的常用方法

- `Set<SelectionKey> keys()`： 所有的 SelectionKey 集合，代表注册在该 Selector上的 Channel
- `selectedKeys()`：被选择的 SelectionKey 集合，返回此 Selector 的已选择键集
- `intselect()`：监控所有注册的 Channel，当有需要处理的 IO 操作时，该方法返回，并将对应的SelectionKey 加入被选择的 SelectionKey 集合中，该方法返回这些 Channel 的数量
- `int select(long timeout)`：可以设置超时时长的 select() 操作
- `intselectNow()`：执行一个立即返回的 select() 操作，该方法不会阻塞线程
- `Selectorwakeup()`：使一个还未返回的 select() 方法立即返回
- `void close()`：关闭该选择器

## 7. 套接字 NIO 实例

```java
public class NIOServer {
    public static void main(String[] args) throws IOException {
        Selector selector = Selector.open();
        ServerSocketChannel ssChannel = ServerSocketChannel.open();
        ssChannel.configureBlocking(false);
        ssChannel.register(selector, SelectionKey.OP_ACCEPT);
        ServerSocket serverSocket = ssChannel.socket();
        InetSocketAddress address = new InetSocketAddress("127.0.0.1", 8888);
        serverSocket.bind(address);
        while (true) {
            selector.select();
            Set<SelectionKey> keys = selector.selectedKeys();
            Iterator<SelectionKey> keyIterator = keys.iterator();
            while (keyIterator.hasNext()) {
                SelectionKey key = keyIterator.next();
                if (key.isAcceptable()) {
                    ServerSocketChannel ssChannel1 = (ServerSocketChannel) key.channel();
                    // 服务器会为每个新连接创建一个 SocketChannel
                    SocketChannel sChannel = ssChannel1.accept();
                    sChannel.configureBlocking(false);
                    // 这个新连接主要用于从客户端读取数据
                    sChannel.register(selector, SelectionKey.OP_READ);
                } else if (key.isReadable()) {
                    SocketChannel sChannel = (SocketChannel) key.channel();
                    System.out.println(readDataFromSocketChannel(sChannel));
                    sChannel.close();
                }
                keyIterator.remove();
            }
        }
    }

    private static String readDataFromSocketChannel(SocketChannel sChannel) 
        throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        StringBuilder data = new StringBuilder();
        while (true) {
            buffer.clear();
            int n = sChannel.read(buffer);
            if (n == -1) {
                break;
            }
            buffer.flip();
            int limit = buffer.limit();
            char[] dst = new char[limit];
            for (int i = 0; i < limit; i++) {
                dst[i] = (char) buffer.get(i);
            }
            data.append(dst);
            buffer.clear();
        }
        return data.toString();
    }
}
```

```java
public class NIOClient {
    public static void main(String[] args) throws IOException {
        Socket socket = new Socket("127.0.0.1", 8888);
        OutputStream out = socket.getOutputStream();
        String s = "hello world";
        out.write(s.getBytes());
        out.close();
    }
}
```

## 8. 内存映射文件

- **内存映射文件 I/O**： 是一种读/写文件数据的方法，比常规的基于流或者基于通道的 I/O 快得多

## 9. 对比

NIO 与普通 I/O 的区别主要有以下两点：

- NIO 是非阻塞的
- NIO 面向块，I/O 面向流
