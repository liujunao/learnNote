# 一、基本使用

## 1. 数据写入与查询

```shell
# 登陆 mongod
mongo
# 显示当前所有数据库
show dbs
# 切换数据库，若无，mongod 会自动创建
use test
# 删除数据库
db.dropDatabase()
# 向数据表插入数据，若无，自动创建
# 注意：数据格式应为 JSON 格式
db.test_collections.insert({x:1})
# 查看当前所有数据表
show collections
# 插入多条数据，mongod 支持 js 语法
for(i=2;i<100;i++)db.test_collections.insert({x:i})
# 查询数据表的所有数据
db.test_collections.find()
# 查询数据表的指定数据，数据格式为 JSON
db.test_collections.find({x:1})
# 查询数据数量
db.test_collections.find().count()
# skip：跳过shuju；limit： 限制数据；sort： 对数据排序
db.test_collections.find().skip(3).limit(2).sort({x:1})
```

## 2. 数据更新

```shell
# 将 x=1 更新为 x=999
db.test_collections.update({x:1},{x:999})
# 只更新部分字段： $set
# 注意： 若不用 $set，将会覆盖掉全部内容
db.test_collections.insert({x:100,y:100,z:100}) # 插入测试数据
db.test_collections.update({z:100},{$set:{y:99}}) # 修改 y
db.test_collections.find({x:100}) # 检查结果是否正确
```

## 3. 更新不存在数据

```shell
# 增加 true 参数： 当更新数据不存在时自动插入
db.test_collections.update({y:100},{y:999},true)
db.test_collections.find({y:999}) # 检查结构是否正确
```

## 4. 更新多条数据

```shell
# 插入三条相同数据
db.test_collections.insert({c:1})
db.test_collections.insert({c:1})
db.test_collections.insert({c:1})
# 当多条数据相同时，默认只更新第一条数据： 更新 c=1 的数据
db.test_collections.update({c:1},{c:2})
# 检查结果
db.test_collections.find({c:1})
db.test_collections.find({c:2})
# 将相同数据全部更新： 第四个参数设为 true
db.test_collections.update({c:1},{$set:{c:2}},false,true)
# 检查结果
db.test_collections.find({c:1})
db.test_collections.find({c:2})
```

## 5. 数据删除

```shell
# 删除 c=2 的全部数据
# 注意： remove 操作必须传递参数，否则将报错
db.test_collections.remove({c:2})
# 检验
db.test_collections.find({c:2})
# 删除表
db.test_collections.drop()
# 检验
show tables
```

## 6. 创建索引

```shell
# 切换数据库并创建一个表
use test
db.test_collections.insert({x:1})
# 查询表中的索引
db.test_collections.getIndexes()
# 创建索引：x=1 为正向排序；x=-1 为逆向排序
db.test_collections.ensureIndex({x:1})
```

# 二、查询索引

## 1. ID 索引

- 自动创建

```shell
# 切换数据库并创建一个表
use test
db.test2.insert({d:1})
# 查询表中的索引
db.test2.getIndexes()
# 查询
db.test2.findOne()
```

## 2. 单键索引

- 最简单索引，但不会自动创建

```shell
# 创建单键索引
db.test2.ensureIndex({d:1})
# 使用索引查询
db.test2.find({d:1})
```

## 3. 多键索引

- 与单键索引创建形式相同，区别在于字段的值

```shell
# 对于插入的该数据，mongod 自动为其创建多键索引
db.test2.insert({d:[1,2,3,4]})
```

## 4. 复合索引

```shell
# 插入 {x:1,y:1,z:1} 记录
db.test2.insert({x:1,y:1,z:1})
# 创建 x,y 的复合索引
db.test2.ensureIndex({x:1,y:1})
# 使用 {x:1,y:1} 作为条件进行查询
db.test2.find({x:1,y:1})
```

## 5. 过期索引

- **过期索引**： 在一段时间后会过期，过期后相应数据会被自动删除；适合存储一段时间后会失效的数据
- **方式**： `db.collection.ensureIndex({time:1},{expireAfterSeconds:10})`

- **限制**： 

  - 存储在过期索引的值必须是指定的时间类型

    > - 必须是 ISODate 或 ISODate 数组
    > - 不能使用时间戳，否则不能被自动删除

  - 若指定了 ISODate 数组，则按照最小时间进行删除

  - 过期索引不能是复合索引

  - 删除时间不精确

    > - 删除过程由后台进程每 60s 跑一次
    > - 且删除过程也需要时间

```shell
# 创建过期索引： 30s 后删除
db.test2.ensureIndex({time:1},{expireAfterSeconds:30})
# 插入数据
db.test2.insert({time:new Date()})
db.test2.insert({time:1})
# 一段时间后检查结果
db.test2.find()
```

## 6. 删除索引

```shell
db.collection.dropIndex("index_name")
```

## 7. 索引属性

比较重要的属性：

- 名字： name 指定， `db.collection.ensureIndex({},{name:" "})`

- 唯一性： unique 指定， `db.collection.ensureIndex({},{unique: true/false})`

- 稀疏性： sparse 指定， `db.collection.ensureIndex({},{sparse: true/false})`

  > - 默认为 false，不稀疏
  > - 不稀疏时，会为不存在的字段创建索引

- 过期性： expireAfterSeconds 指定， `db.collection.ensureIndex({},{expireAfterSeconds: num})`

# 三、全文索引

## 1. 简介

- **全文索引**： 对字符串与字符串数组创建全文可搜索的索引
- **限制**： 
  - 每次查询只能指定一个 `$text` 查询
  - `$text` 查询不能出现在 `$nor` 查询中
  - 查询中若包含 `$text`，则 `hint` 不再起作用

- 创建方法：

  ```
  db.collection.ensureIndex({key:"text"})
  db.collection.ensureIndex({key_1:"text",key_2:"text"})
  db.collection.ensureIndex({"$**":"text"})
  ```

- 查询方法

  ```shell
  db.collection.find({$text:{$search:"xxx"}})
  db.collection.find({$text:{$search:"xx xx"}}) # 有一个包含便返回
  db.collection.find({$text:{$search:"aa bb -cc"}}) # 包含 aa bb 但不包含 cc
  db.collection.find({$text:{$search:"\"aa\" \"bb\" \"cc\""}}) # 同时包含 aa bb cc 时返回
  ```

## 2. 使用全文索引

```shell
# 创建全文索引
db.test2.ensureIndex({"article":"text"})
db.test2.insert({"article":"aa bb cc"})
db.test2.insert({"article":"aa bb hh tt"})
db.test2.insert({"article":"aa tt ff"})

# 查询全文索引
db.test2.find({$text:{$search:"aa"}})
db.test2.find({$text:{$search:"aa bb"}})
db.test2.find({$text:{$search:"\"aa\" \"bb\" \"cc\""}})
```

## 3. 相似度查询

```shell
# 查询
db.test2.find({$text:{$search:"aa bb"}},{score:{$meta:"textScore"}})
# 对查询结果进行排序
db.test2.find({$text:{$search:"aa bb"}},{score:{$meta:"textScore"}}).sort({score:{$meta:"textScore"}})
```

# 四、地理位置索引

## 1. 简介

- **地理位置索引**： 将一些点的位置存储在 MongoDB 中，创建索引后，可以按照位置来查找其他点
- **子分类**： 
  - **2d 索引**： 用于存储和查找==平面==上的点
  - **2dsphere 索引**： 用于存储和查找==球面==上的点

- **查找方式**： 
  - 查找距离某个点一定距离内的点
  - 查找包含在某区域内的点

## 2. 2d 索引

- **创建方式**： `db.collection.ensureIndex({"w":"2d"})`

- **位置表示方式**： 经纬度 [经度，维度]

- **取值范围**： 经度 [-180,180] ，纬度 [-90,90]

- 查询方式：

  - `$near` 查询： 查询距离某个点最近的点

  - `$geoWithin` 查询： 查询某个形状内的点

    > 形状表示：
    >
    > - `$box`： 矩形，使用 `{$box:[[<x1>,<y1>],[<x2>,<y2>]]}` 表示
    > - `$center`： 圆形，使用 `{$center:[[<x1>,<y1>],r]}` 表示
    > - `$polygon`： 多边形，使用 `{$polygon:[[<x1>,<y1>],[<x2>,<y2>],[<x3>,<y3>]...]}` 表示

  - `geoNear` 查询： 使用 `runCommand` 命令进行操作

    > ```shell
    > db.runCommand({
    >    geoNear: <collection>,
    >    near: [x,y],
    >    minDistance: (对 2d 索引无效),
    >    maxDistance: ,
    >    num:
    >    ...
    > })
    > ```

```shell
# 插入 2d 索引
db.location.ensureIndex({"w":"2d"})
# 插入测试数据
db.location.insert({w:[1,1]})
db.location.insert({w:[1,2]})
db.location.insert({w:[2,1]})
db.location.insert({w:[100,10]})
db.location.insert({w:[-100,80]})

# $near 查询
# 查询：会返回 100 个距离较近的点
db.location.find({w:{$near:[1,1]}})
# 限制距离，$near 不支持 $minDistance
db.location.find({w:{$near:[1,1],$maxDistance:10}})

# $geoWithin 查询
# 查询矩形范围内的点
db.location.find({w:{$geoWithin: {$box: [[0,0],[3,3]]}}})
# 查询圆内的点
db.location.find({w:{$geoWithin: {$center: [[0,0],5]}}})
# 查询多边形内的点
db.location.find({w:{$geoWithin: {$polygon:[[0,0],[0,1],[2,5],[6,1]]}}})

# geoNear 查询
db.runCommand({geoNear:"location",near:[1,2],maxDistance:10,num:1})
```

## 3. 2dsphere 索引

- **创建方式**： `db.collection.ensureIndex({"w":"2dsphere"})` 
- 位置表示方式： GeoJSON，格式： `{type: " ",coordinates:[<coordinates>]}`
- 查询方式： 与 2d 索引查询类似，支持 `$minDistance 和 maxDistance`

# 五、性能分析

## 1. 索引构建分析

- **索引好处**： 加快索引相关的查询
- **索引缺点**： 增加磁盘空间消耗，降低写入性能

评判方式：

- `mongostat` 工具
- `profile` 集合
- 日志
- `explain` 分析

## 2. `mongostat` 工具

- `mongostat`： 查看 mongodb 运行状态的程序
- 使用说明： `mongostat -h IP:port`

```shell
mongostat -h 127.0.0.1:27017 # mongodb 默认端口为 27017
```

## 3. `profile` 集合

```shell
# 查看 profile 设置
db.getProfilingStatus()
# 可使用 tab 键来自动补全来查看更多的设置
```

## 4. 日志



## 5. `explain` 分析

```shell
db.test2.find().explain()
```

# 五、安全

## 1. 简介

- **安全概览**：
  - **物理隔离**
  - **网络隔离**
  - **防火墙隔离** 
  - **用户名密码隔离**

## 2. 开启权限认证

- 在 `mongod.conf` 配置文件中设置： `auth=true`

- 开启后需要使用==用户名与密码==来登录(若没有用户名和密码则可以直接登录)

## 3. 创建用户

- **创建语法**： 

  ```shell
  createUser({
      user: "<name>",
      pwd: "<password>",
      customData: {<any information>},
      roles: [{role: "<role>",db: "<database>"}]
  })
  ```

- **数据库角色**： 内建类型(`read, readWrite, dbAdmin, dbOwner, userAdmin`)

- **集群角色**： `clusterAdmin, clusterManager ...`
- **备份角色**： `backup, restore ...`
- **其他特殊权限**： `DBAdminAnyDatabse ...`