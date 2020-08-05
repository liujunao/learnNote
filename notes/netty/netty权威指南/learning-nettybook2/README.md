# Netty 权威指南 （第2版）nettybook2

该项目是李林峰老师编写的netty权威指南（第二版）对应的源码。

源码原始地址（已失效）：http://vdisk.weibo.com/s/C9LV9iVqAFvqu

因为原始项目是用ant构建，而且还要下载导入。

所以本人将其项目进行简单的maven转换，并且提交到github上。这样同志们就可以直接import
-> git查看。

有关该书的更多信息可以关注李林峰老师的微博 @Nettying
以及查看其在ifeve网站上的文章：http://ifeve.com/author/linfeng/


[Gitee地址](https://gitee.com/baiyimi/learning-nettybook2)

[Github地址](https://github.com/baiyimi/learning-nettybook2)

---

### Netty版本说明


Netty当前的版本：

User guide for 4.x - RECOMMENDED VERSION

User guide for 3.x

User guide for 5.x - ABANDONED VERSION - NOT SUPPORTED

    4.x版本是当前官方推荐，4.x版本目前一直在维护中，值得称赞！    

    3.x版本是比较旧的版本，跟4.x版本相比变化比较大，特别是API。

    5.x是被舍弃的版本，官方不再支持！

    Netty 5.0以前是发布alpha版。听到你Netty 5.0不继续开发了，这个是相当大的吃惊，目前也有一部分书籍是基于Netty5来讲的，所以给那些初学者也是很郁闷的赶脚。

    为啥呢？看看GitHub上怎么作者怎么回复的吧？

---

The major change of using aForkJoinPool increases complexityand has not
demonstrated a clear performance benefit. Also keeping all the branches
in sync is quite some work without a real need for it as there is nothin
in current master which I think justifies a new major release.

Things that we should investigate to prepare for this change:
    Deprecate exceptionCaught in ChannelHandler, only expose it in ChannelInboundHandler Expose EventExecutorChooser from MultithreadEventExecutorGroup to allow the user more flexibility to choose next EventLoop Add another method to be able to send user events both ways in the pipeline. (#4378)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

主要意思：

使用ForkJoinPool增加了复杂性，并且没有显示出明显的性能优势。同时保持所有的分支同步是相当多的工作，没有必要。

详情请看github地址：

        https://github.com/netty/netty/issues/4466

    目前推荐使用的版本是Netty 4.0 or 4.1（我推荐） 。加油，Nettyer 。
