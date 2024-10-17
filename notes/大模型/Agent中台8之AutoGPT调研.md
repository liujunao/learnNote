- github：https://github.com/Significant-Gravitas/AutoGPT
- 官网文档：https://docs.agpt.co/

# 一、简介

## 1、项目结构

- Autogpt 目录：一个由 LLMs 提供动力的半自主代理，可以为您执行任何任务
- Benchmark 目录：基准测试，允许自主、客观地评估性能
- forge 目录：为 Agent 应用准备的模板，详细教程 https://aiedge.medium.com/autogpt-forge-e3de53cc58ec
- frontend ：一个易于使用且开源的前端，适用于任何符合 Agent Protocol 标准的 Agent

## 2、Protocol

### 2.1 无序依赖协议

#### (1) `DirectiveProvider`

为 agent 提供约束、资源和最佳实践，即组装进 prompt

``` python
class DirectiveProvider(AgentComponent):
    def get_constraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])
```

#### (2) `CommandProvider`

提供可由代理执行的命令

```python
class CommandProvider(AgentComponent):
    def get_commands(self) -> Iterator[Command]:
        ...
```

### 2.2 有序依赖协议

#### (1) `MessageProvider`

生成将添加到 prompt 的消息

```python
class MessageProvider(AgentComponent):
    def get_messages(self) -> Iterator[ChatMessage]:
        ...
```

#### (2) `AfterParse`

解析响应后调用的协议

```python
class AfterParse(AgentComponent):
    def after_parse(self, response: ThoughtProcessOutput) -> None:
        ...
```

#### (3) `ExecutionFailure`

命令执行失败时调用的协议

```python
class ExecutionFailure(AgentComponent):
    @abstractmethod
    def execution_failure(self, error: Exception) -> None:
        ...
```

#### (4) `AfterExecute`

agent 成功执行命令后调用的协议

```python
class AfterExecute(AgentComponent):
    def after_execute(self, result: ActionResult) -> None:
        ...
```



## 3、Components

- 组件是 agent 的构建块，继承 `AgentComponent` 或实现一个或多个协议的类

- 组件可用于实现各种功能，例如向提示符提供消息、执行代码、与外部服务交互



## 4、Agents

> pipeline

`BaseAgent` 提供了任何代理正常工作所需的两种抽象方法： 

1. `propose_action` ：此方法负责根据代理的当前状态提出操作，返回 `ThoughtProcessOutput` 
2. `execute` ：此方法负责执行建议的操作，返回 `ActionResult` 

## 5、Command

命令是可以由代理调用的函数，可以具有代理将看到的参数和返回值



## 6、内置组件





# 二、agent protocol

## 1、openAPI 规范

官方文档：https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md

### 1.1 定义

- **openAI 文档**：定义或描述 API 或 API 元素的独立资源或复合资源

    > 必须包含至少一个 paths 字段、组件字段或 webhooks 字段

- **路径模版**：指使用模板表达式（由大括号 （{}） 分隔），以使用路径参数将 URL 路径的一部分标记为可替换

- **媒体类型**：分布在多个资源中

    ```shell
    #可能的媒体类型定义的一些示例：  
    text/plain; charset=utf-8
      application/json
      application/vnd.github+json
      application/vnd.github.v3+json
      application/vnd.github.v3.raw+json
      application/vnd.github.v3.text+json
      application/vnd.github.v3.html+json
      application/vnd.github.v3.full+json
      application/vnd.github.v3.diff
      application/vnd.github.v3.patch
    ```

- **HTTP 状态码**：用于指示已执行操作的状态



### 1.2 规范

- **版本**：使用 `major.minor.patch` 版本控制方案

- **格式**：符合 OpenAPI 规范的 OpenAPI 文档本身就是一个 JSON 对象，可以用 JSON 或 YAML 格式表示

- **文档结构**：

- **数据类型**：数据类型可以具有可选的修饰符属性，即 `format` 

    | [`type`](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#data-types) | [`format`](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#dataTypeFormat) |         Comments 评论          |
    | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------: |
    |                          `integer`                           |                           `int32`                            |         signed 32 bits         |
    |                          `integer`                           |                           `int64`                            |  signed 64 bits (a.k.a long)   |
    |                           `number`                           |                           `float`                            |                                |
    |                           `number`                           |                           `double`                           |                                |
    |                           `string`                           |                          `password`                          | A hint to UIs to obscure input |

- **RTF 格式**：必须至少支持 [CommonMark 0.27](https://spec.commonmark.org/0.27/) 中描述的 Markdown 语法

- **URIs 中的相对引用**：相对引用（包括 `Reference Objects` 、 `PathItem Object` `$ref` 字段、 `Link Object` `operationRef` 字段和 `Example Object` `externalValue` 字段中的引用）根据RFC3986使用引用文档作为基本 URI 进行解析

- **URLs 中的相对引用**：使用“作为基本 URL” `Server Object` 中定义的 URL 来解析相对引用

- **schema**：在以下说明中，如果某个字段不是显式必填字段或用 MUST 或 SHALL 描述的字段，则可以将其视为 OPTIONAL

    - **OpenAI 对象**：
    - **info 对象**：提供有关 API 的元数据
    - **Contact 对象**：公开的 API 联系信息
    - **License 对象**：公开的 API 的许可证信息
    - **Server 对象**：表示服务器的对象
    - **Server Variable 对象**：表示用于服务器 URL 模板替换的服务器变量的对象
    - **Components 对象**：拥有一组可重复使用的对象，用于 OAS 的不同方面
    - **Paths 对象**：保存各个端点及其操作的相对路径，该路径会附加到 `Server Object` 中的 URL 中，以构建完整的 URL
    - **Path Item 对象**：描述单个路径上可用的操作
    - **Operation 对象**：描述路径上的单个 API 操作
    - **External Documentation 对象**：
    - **Parameter 对象**：
    - **Request Body 对象**：
    - **Media Type 对象**：
    - **Encoding 对象**：
    - **Response 对象**：
    - **Callback 对象**：
    - **Example 对象**：
    - **Link 对象**：
    - **Header 对象**：
    - **Tag 对象**：
    - **Reference 对象**：
    - **Schema 对象**：
    - **Discriminator 对象**：
    - **XML 对象**：
    - **Security Scheme 对象**：
    - **OAuth Flows 对象**：
    - **Security Requirement 对象**：

- 规范扩展
- 安全过滤



### 1.3 swagger 工具实现

文档：https://swagger.io/docs/specification/about/



## 2、agent protocol

官方网站：https://agentprotocol.ai/



