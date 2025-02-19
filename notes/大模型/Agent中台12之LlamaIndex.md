# 零、前言总结

## 1、核心组件

`Node`：基础 Object 对象，用于存储包括文档在内的基本信息，Document 就是继承类

其核心字段如下：

- `text/img_source`：存储不同用途的基本信息
- `metadata`：存储扩展信息，包括非 emb 信息也存在里面(感觉是当初的设计问题)
- `emb`：存储 vector 信息

## 2、pipeline

核心功能：将 `TransformComponent` 的继承类按顺序执行，包括：split、emb、store 等逻辑



## 3、split/nodeParser

- 核心类`NodeParser`：
- 核心函数`get_nodes_from_documents`：
    - `_parse_nodes`：切分 document，并组装新的 Node 对象
    - `_postprocess_parsed_nodes`：构建 Node 之间的 relationships，比如：图中的边(关系)

## 4、transform(转换)

- extractor：提炼 node 中的核心信息，如 title，结果放入 metadata 中
- emb：直接调用 embedding 模型即可
- graph：调用模型计算点和边，结果放入 metadata 中

## 5、store

核心类 `StorageContext`

核心函数 `from_defaults`

核心存储：docstore、vector_store、graph_store 等

## 6、索引

核心类 `BaseIndex`

核心函数：

- `from_documents`：执行 transformComponent 节点(split/extractor/embed/store等步骤)
- `as_retriever`
- `as_query_engine`：由 as_retriever 和 llm 组成
- `as_chat_engine`：由 llm 和 as_retriever/as_query_engine/EngineTool 组成

## 7、检索

```python
retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")
```

## 8、postprocessor(rerank)

```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

nodes = [
    NodeWithScore(node=Node(text="text1"), score=0.7),
    NodeWithScore(node=Node(text="text2"), score=0.8),
]

# similarity postprocessor: filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)

# cohere rerank: rerank nodes given query using trained model
reranker = CohereRerank(api_key="<COHERE_API_KEY>", top_n=2)
reranker.postprocess_nodes(nodes, query_str="<user_query>")

#NodeWithScore
class NodeWithScore(BaseComponent):
    node: SerializeAsAny[BaseNode]
    score: Optional[float] = None
```

## 9、结果合成(response_synthesizer)

核心类 `BaseSynthesizer`

核心函数 `synthesize`

- 获取 response
- response 合并，如：graph
- 转换为对应的返回对象

```python
response_synthesizer = get_response_synthesizer(
    response_mode="refine", #指定 response_synthesizer
    service_context=service_context, #定义 llm 相关配置
    text_qa_template=text_qa_template, #匹配的 prompt
    refine_template=refine_template, #匹配的 prompt
    use_async=False,
    streaming=False,
)

# synchronous
response = response_synthesizer.synthesize(
    "query string",
    nodes=[NodeWithScore(node=Node(text="text"), score=1.0), ...],
    additional_source_nodes=[
        NodeWithScore(node=Node(text="text"), score=1.0),
        ...,
    ],
)
```

## 10、查询引擎(query engine)

```python
#方法一
RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs,) #llm是为了组装 response_synthesizer
#方法二
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What did the author do growing up?")
```

查询引擎的组成：

- `retriever`      
- `response_synthesizer`
- `node_postprocessors`

## 11、agent

**核心类**：

- `AgentRunner`：是存储状态(包括对话内存)、创建和维护任务、通过每个任务运行步骤、提供面向用户的高级接口
- `AgentWorker`：控制任务的逐步执行，即**给定 input step，agentWorker 负责生成下一步**。
    - 可以用参数初始化，并对从任务/任务步骤对象传递的状态作用，但不能固有地存储自己的状态
    - 外部 `AgentRunner` 负责调用 `AgentWorker` 并收集/汇总结果

**辅助类**：Task 为一个具体的 LLM 任务，而 step 则是包含一个或多个 Task 的具体执行步骤

- `Task` ：高级任务，接入用户查询 +沿其他信息（例如内存）

- `TaskStep` ：表示一个 step。将其作为输入给 `AgentWorker` ，并返回 `TaskStepOutput` 

    > 完成一个 `Task` 可能涉及多个 `TaskStep`

- `TaskStepOutput`：从给定的 step 执行中输出，输出是否完成任务

agent 类型：

- `ReActAgent`：while(true) 循环执行 task，直到获取到的 step_output 结果显示结束为止
- `FunctionCallingAgent`：通过 LLM 获取到要执行的 tools，然后依次执行
- `PlanAgent`：通过 LLM 获取到要执行的 plan，并执行 plan 中得到的 task，拿到最后的结果

## 12、workflow





## 13、评估

**响应评估**：

- **`Correctness`(正确性)**：生成的答案是否与给定查询的参考答案相匹配(需要标签)
    - 根据现有的 response 和 reference，调用 LLM 来进行相似性评估
    - 然后，获取结果的 score 和 reason
- **`Semantic Similarity`(语义相似性)**：是否在语义上与参考答案相似(需要标签)
    - 将 response 和 reference 都转换为 emb
    - 然后，计算二者的相似度
- **`Faithfulness`(忠诚)**：评估答案是否忠实于检索到的上下文(换句话说，是否有幻觉)
    - 给定一批 contexts，并转换为 index
    - 然后，用 response 进行检索，判断 response 是否属于这批 contexts，即幻觉
- **`Context Relevancy`(上下文相关性)**：检索到上下文是否与查询有关
    - 给定一批 contexts，并转换为 index
    - 然后，用 response 进行检索，并用 LLM 进行相关性评估，获取结果的 score 和 reason
- **`Answer Relevancy`(答案相关性)**：生成的答案是否与查询相关
    - 根据现有的 response 和 query，调用 LLM 来进行相似性评估
    - 然后，获取结果的 score 和 reason
- **`Guideline Adherence`(指南依从性)**：预测的答案是否遵守特定准则
    - 使用 LLM 计算两个 response 与 reference 的相关性，得到  score 和 reason
    - 然后，进行对比

**检索评估**：

- **`Dataset generation`(数据集生成)**：给定一个非结构化的文本语料库，合成生成（问题，上下文）对

- **`Retrieval Evaluation`(检索评估)**：给定检索器和一组问题，使用排名指标评估结果的结果

    - 核心逻辑：
        - 先根据 query 检索，得到 retrieved_text
        - 然后根据指定的 metric 分别计算 expected_texts 和 retrieved_texts 的评估结果

    - metric 类型：`hit_rate、mrr、precision、recall、ap、ndcg、cohere_rerank_relevancy`

```python
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)
```

# 一、核心类/组件

## 1、Documents / Nodes(shema类)

### 1.1 BaseNode

```python
class BaseNode(BaseComponent):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Embedding of the node."
    )

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering

    """
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )
    relationships: Dict[
        Annotated[NodeRelationship, EnumNameSerializer],
        RelatedNodeType,
    ] = Field(
        default_factory=dict,
        description="A mapping of relationships to other node information.",
    )
```

### 1.2 Node 与 TextNode

```python
class Node(BaseNode):
    text_resource: MediaResource | None = Field(
        default=None, description="Text content of the node."
    )
    image_resource: MediaResource | None = Field(
        default=None, description="Image content of the node."
    )
    audio_resource: MediaResource | None = Field(
        default=None, description="Audio content of the node."
    )
    video_resource: MediaResource | None = Field(
        default=None, description="Video content of the node."
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text_resource is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    
class TextNode(BaseNode):
    text: str = Field(default="", description="Text content of the node.")
    mimetype: str = Field(
        default="text/plain", description="MIME type of the node content."
    )
    start_char_idx: Optional[int] = Field(
        default=None, description="Start char index of the node."
    )
    end_char_idx: Optional[int] = Field(
        default=None, description="End char index of the node."
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )

#补充
DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"
```

### 1.3 Document

- **Document 可以理解为 load 时获取到的完整文本，比如一本书的信息**
- **Node 可以理解理解为进行 chunk 与 parse 后得到部分文本，比如：书的一个章节**

```python
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    excluded_llm_metadata_keys=["file_name"],
    metadata_seperator="::",
    metadata_template="{key}=>{value}",
    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
)

print(
    "The LLM sees this: \n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)
print(
    "The Embedding model sees this: \n",
    document.get_content(metadata_mode=MetadataMode.EMBED),
)
```

## 2、pipeline(ingestion包)

### 2.1 用法

```python
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[Document.example()])
```

### 2.2 核心字段

```python
class IngestionPipeline(BaseModel):
    """
    An ingestion pipeline that can be applied to data.
    
    Examples:
        ```python
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.openai import OpenAIEmbedding

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                OpenAIEmbedding(),
            ],
        )

        nodes = pipeline.run(documents=documents)
    """
    name: str = Field(
        default=DEFAULT_PIPELINE_NAME,
        description="Unique name of the ingestion pipeline",
    )
    project_name: str = Field(
        default=DEFAULT_PROJECT_NAME, description="Unique name of the project"
    )

    transformations: List[TransformComponent] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    readers: Optional[List[ReaderConfig]] = Field(
        description="Reader to use to read the data"
    )
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )
    cache: IngestionCache = Field(
        default_factory=IngestionCache,
        description="Cache to use to store the data",
    )
    docstore: Optional[BaseDocumentStore] = Field(
        default=None,
        description="Document store to use for de-duping with a vector store.",
    )
    docstore_strategy: DocstoreStrategy = Field(
        default=DocstoreStrategy.UPSERTS, description="Document de-dup strategy."
    )
    disable_cache: bool = Field(default=False, description="Disable the cache")
```
### 2.3 TransformComponent 

> 后面的 splitter、index(embedding/graph)、storage 等都是继承该类

```python
class TransformComponent(BaseComponent, DispatcherSpanMixin):
    """Base class for transform components."""
    @abstractmethod
    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Transform nodes."""

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Async transform nodes."""
        return self.__call__(nodes, **kwargs)
```

### 2.4 run 函数

#### (1) run

```python
@dispatcher.span
def run(
    self,
    show_progress: bool = False,
    documents: Optional[List[Document]] = None,
    nodes: Optional[Sequence[BaseNode]] = None,
    cache_collection: Optional[str] = None,
    in_place: bool = True,
    store_doc_text: bool = True,
    num_workers: Optional[int] = None,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Run a series of transformations on a set of nodes.

    If a vector store is provided, nodes with embeddings will be added to the vector store.
    If a vector store + docstore are provided, the docstore will be used to de-duplicate documents.

    Args:
        show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
        documents(Optional[List[Document]],optional): Set of documents to be transformed. Defaults to None.
        nodes (Optional[Sequence[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
        cache_collection (Optional[str], optional): Cache for transformations. Defaults to None.
        in_place (bool, optional): Whether transformations creates a new list for transformed nodes or 
        	modifies the array passed to `run_transformations`. Defaults to True.
        num_workers (Optional[int], optional): The number of parallel processes to use.
            If set to None, then sequential compute is used. Defaults to None.

    Returns:
        Sequence[BaseNode]: The set of transformed Nodes/Documents
    """
    input_nodes = self._prepare_inputs(documents, nodes)

    # check if we need to dedup
    if self.docstore is not None and self.vector_store is not None:
        if self.docstore_strategy in (
            DocstoreStrategy.UPSERTS,
            DocstoreStrategy.UPSERTS_AND_DELETE,
        ):
            nodes_to_run = self._handle_upserts(
                input_nodes, store_doc_text=store_doc_text
            )
        elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
            nodes_to_run = self._handle_duplicates(
                input_nodes, store_doc_text=store_doc_text
            )
        else:
            raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
    elif self.docstore is not None and self.vector_store is None:
        if self.docstore_strategy == DocstoreStrategy.UPSERTS:
            print(
                "Docstore strategy set to upserts, but no vector store. "
                "Switching to duplicates_only strategy."
            )
            self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
        elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            print(
                "Docstore strategy set to upserts and delete, but no vector store. "
                "Switching to duplicates_only strategy."
            )
            self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
        nodes_to_run = self._handle_duplicates(
            input_nodes, store_doc_text=store_doc_text
        )

    else:
        nodes_to_run = input_nodes

    if num_workers and num_workers > 1:
        num_cpus = multiprocessing.cpu_count()
        if num_workers > num_cpus:
            warnings.warn(
                "Specified num_workers exceed number of CPUs in the system. "
                "Setting `num_workers` down to the maximum CPU count."
            )
            num_workers = num_cpus

        with multiprocessing.get_context("spawn").Pool(num_workers) as p:
            node_batches = self._node_batcher(
                num_batches=num_workers, nodes=nodes_to_run
            )
            nodes_parallel = p.starmap(
                run_transformations,
                zip(
                    node_batches,
                    repeat(self.transformations),
                    repeat(in_place),
                    repeat(self.cache if not self.disable_cache else None),
                    repeat(cache_collection),
                ),
            )
            nodes = reduce(lambda x, y: x + y, nodes_parallel, [])  # type: ignore
    else:
        nodes = run_transformations(
            nodes_to_run,
            self.transformations,
            show_progress=show_progress,
            cache=self.cache if not self.disable_cache else None,
            cache_collection=cache_collection,
            in_place=in_place,
            **kwargs,
        )

    nodes = nodes or []

    if self.vector_store is not None:
        nodes_with_embeddings = [n for n in nodes if n.embedding is not None]
        if nodes_with_embeddings:
            self.vector_store.add(nodes_with_embeddings)

    return nodes
```

#### (2) run_transformations

```python
def run_transformations(
    nodes: Sequence[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        if cache is not None:
            hash = get_transformation_hash(nodes, transform)
            cached_nodes = cache.get(hash, collection=cache_collection)
            if cached_nodes is not None:
                nodes = cached_nodes
            else:
                nodes = transform(nodes, **kwargs) #触发调用相应继承类中的 __call__ 函数
                cache.put(hash, nodes, collection=cache_collection)
        else:
            nodes = transform(nodes, **kwargs)

    return nodes
```

# 二、Splitter/nodeParser(node_parser包)

> documents 转换为 node

## 1、NodeParser

```python
class NodeParser(TransformComponent, ABC):
    """Base interface for node parser."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    id_func: Optional[IdFuncCallable] = Field(
        default=None,
        description="Function to generate node IDs.",
    )
```

## 2、get_nodes_from_documents

```python
def get_nodes_from_documents(
    self,
    documents: Sequence[Document],
    show_progress: bool = False,
    **kwargs: Any,
) -> List[BaseNode]:
    """Parse documents into nodes.

    Args:
        documents (Sequence[Document]): documents to parse
        show_progress (bool): whether to show progress bar

    """
    doc_id_to_document = {doc.id_: doc for doc in documents}

    with self.callback_manager.event(
        CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
    ) as event:
        nodes = self._parse_nodes(documents, show_progress=show_progress, **kwargs) #doc 解析
        nodes = self._postprocess_parsed_nodes(nodes, doc_id_to_document) #node连接(relationships)

        event.on_end({EventPayload.NODES: nodes})

    return nodes
```

## 3、doc 解析(_parse_nodes)

### 3.1 TextSplitter 对比 JSONNodeParser

```python
class TextSplitter(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits = self.split_text(node.get_content()) #切分

            all_nodes.extend(
                build_nodes_from_splits(splits, node, id_func=self.id_func) #结果组装
            )

        return all_nodes

#---------------------------------------------------
class JSONNodeParser(NodeParser):
    """JSON node parser.

    Splits a document into Nodes using custom JSON splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Handle invalid JSON input here
            return []

        json_nodes = []
        if isinstance(data, dict):
            lines = [*self._depth_first_yield(data, 0, [])]
            json_nodes.extend(
                build_nodes_from_splits(["\n".join(lines)], node, id_func=self.id_func)
            )
        elif isinstance(data, list):
            for json_object in data:
                lines = [*self._depth_first_yield(json_object, 0, [])]
                json_nodes.extend(
                    build_nodes_from_splits(
                        ["\n".join(lines)], node, id_func=self.id_func
                    )
                )
        else:
            raise ValueError("JSON is invalid")

        return json_nodes
    
    def _depth_first_yield(
        self, json_data: Dict, levels_back: int, path: List[str]
    ) -> Generator[str, None, None]:
        """Do depth first yield of all of the leaf nodes of a JSON.

        Combines keys in the JSON tree using spaces.

        If levels_back is set to 0, prints all levels.

        """
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from self._depth_first_yield(value, levels_back, new_path)
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from self._depth_first_yield(value, levels_back, path)
        else:
            new_path = path[-levels_back:]
            new_path.append(str(json_data))
            yield " ".join(new_path)
```

### 3.2 build_nodes_from_splits

```python
def build_nodes_from_splits(
    text_splits: List[str],
    document: BaseNode,
    ref_doc: Optional[BaseNode] = None,
    id_func: Optional[IdFuncCallable] = None,
) -> List[TextNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document
    id_func = id_func or default_id_func
    nodes: List[TextNode] = []
    """Calling as_related_node_info() on a document recomputes the hash for the whole text and metadata"""
    """It is not that bad, when creating relationships between the nodes, but is terrible when adding a relationship"""
    """between the node and a document, hence we create the relationship only once here and pass it to the nodes"""
    relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                id_=id_func(i, document),
                text=text_chunk,
                embedding=document.embedding,
                image=document.image,
                image_path=document.image_path,
                image_url=document.image_url,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_separator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships=relationships,
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document):
            node = TextNode(
                id_=id_func(i, document),
                text=text_chunk,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_separator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships=relationships,
            )
            nodes.append(node)
        elif isinstance(document, TextNode):
            node = TextNode(
                id_=id_func(i, document),
                text=text_chunk,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships=relationships,
            )
            nodes.append(node)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")

    return nodes
```

## 4、relationships(_postprocess_parsed_nodes)

```python
def _postprocess_parsed_nodes(
    self, nodes: List[BaseNode], parent_doc_map: Dict[str, Document]
) -> List[BaseNode]:
    for i, node in enumerate(nodes):
        parent_doc = parent_doc_map.get(node.ref_doc_id or "", None)
        parent_node = node.source_node

        if parent_doc is not None:
            if parent_doc.source_node is not None:
                node.relationships.update(
                    {
                        NodeRelationship.SOURCE: parent_doc.source_node,
                    }
                )
            start_char_idx = parent_doc.text.find(
                node.get_content(metadata_mode=MetadataMode.NONE)
            )

            # update start/end char idx
            if start_char_idx >= 0 and isinstance(node, TextNode):
                node.start_char_idx = start_char_idx
                node.end_char_idx = start_char_idx + len(
                    node.get_content(metadata_mode=MetadataMode.NONE)
                )

            # update metadata
            if self.include_metadata:
                # Merge parent_doc.metadata into nodes.metadata, giving preference to node's values
                node.metadata = {**parent_doc.metadata, **node.metadata}

        if parent_node is not None:
            if self.include_metadata:
                parent_metadata = parent_node.metadata

                combined_metadata = {**parent_metadata, **node.metadata}

                # Merge parent_node.metadata into nodes.metadata, giving preference to node's values
                node.metadata.update(combined_metadata)

        if self.include_prev_next_rel:
            # establish prev/next relationships if nodes share the same source_node
            if (
                i > 0
                and node.source_node
                and nodes[i - 1].source_node
                and nodes[i - 1].source_node.node_id == node.source_node.node_id  # type: ignore
            ):
                node.relationships[NodeRelationship.PREVIOUS] = nodes[
                    i - 1
                ].as_related_node_info()
            if (
                i < len(nodes) - 1
                and node.source_node
                and nodes[i + 1].source_node
                and nodes[i + 1].source_node.node_id == node.source_node.node_id  # type: ignore
            ):
                node.relationships[NodeRelationship.NEXT] = nodes[
                    i + 1
                ].as_related_node_info()

    return nodes

class NodeRelationship(str, Enum):
    """Node relationships used in `BaseNode` class.

    Attributes:
        SOURCE: The node is the source document.
        PREVIOUS: The node is the previous node in the document.
        NEXT: The node is the next node in the document.
        PARENT: The node is the parent node in the document.
        CHILD: The node is a child node in the document.

    """
    SOURCE = auto()
    PREVIOUS = auto()
    NEXT = auto()
    PARENT = auto()
    CHILD = auto()
```

# 三、转换

## 1、常规 Metadata 提取(extractors包)

> 以 TitleExtractor 为例

### 1.1 TitleExtractor 定义

```python
class TitleExtractor(BaseExtractor):
    """Title extractor. Useful for long documents. Extracts `document_title` metadata field.

    Args:
        llm (Optional[LLM]): LLM
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into a document-level title
    """
```

### 1.2 aextract

```python
async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
    nodes_by_doc_id = self.separate_nodes_by_ref_id(nodes)
    titles_by_doc_id = await self.extract_titles(nodes_by_doc_id)
    return [{"document_title": titles_by_doc_id[node.ref_doc_id]} for node in nodes]

def separate_nodes_by_ref_id(self, nodes: Sequence[BaseNode]) -> Dict:
    separated_items: Dict[Optional[str], List[BaseNode]] = {}
    for node in nodes:
        key = node.ref_doc_id
        if key not in separated_items:
            separated_items[key] = []
        if len(separated_items[key]) < self.nodes:
            separated_items[key].append(node)

    return separated_items
```

### 1.3 extract_titles

```python
 async def extract_titles(self, nodes_by_doc_id: Dict) -> Dict:
    titles_by_doc_id = {}
    for key, nodes in nodes_by_doc_id.items():
        title_candidates = await self.get_title_candidates(nodes)
        combined_titles = ", ".join(title_candidates)
        titles_by_doc_id[key] = await self.llm.apredict(
            PromptTemplate(template=self.combine_template),
            context_str=combined_titles,
        )
    return titles_by_doc_id

async def get_title_candidates(self, nodes: List[BaseNode]) -> List[str]:
    title_jobs = [
        self.llm.apredict(
            PromptTemplate(template=self.node_template),
            context_str=cast(TextNode, node).text,
        )
        for node in nodes
    ]
    return await run_jobs(
        title_jobs, show_progress=self.show_progress, workers=self.num_workers
    )
    
DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """

DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """
```

## 2、embedding(base/embedding包)

> 向量信息存储在 embedding 字段中

### 2.1 核心字段

```python
class BaseEmbedding(TransformComponent, DispatcherSpanMixin):
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )
```

### 2.2 `__call__` 函数

```python
def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
    embeddings = self.get_text_embedding_batch(
        [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
        **kwargs,
    )

    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    return nodes
```

### 2.3 `_get_text_embedding` 函数

```python
#最后结果放在 node.embedding 字段中
@abstractmethod
def _get_text_embedding(self, text: str) -> Embedding:
    """
    Embed the input text synchronously.

    Subclasses should implement this method. Reference get_text_embedding's
    docstring for more information.
    """
```

## 3、Property Graph

> 图信息存储在 metadata 字段中

### 3.1 SimpleLLMPathExtractor

#### (1) 用法

```python
prompt = (
    "Some text is provided below. Given the text, extract up to "
    "{max_paths_per_chunk} "
    "knowledge triples in the form of `subject,predicate,object` on each line. Avoid stopwords.\n"
)

def parse_fn(response_str: str) -> List[Tuple[str, str, str]]:
    lines = response_str.split("\n")
    triples = [line.split(",") for line in lines]
    return triples

kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=prompt,
    parse_fn=parse_fn,
)
```

#### (2) 核心字段

```python
class SimpleLLMPathExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
```

#### (3) 核心函数`_aextract`

```python
async def _aextract(self, node: BaseNode) -> BaseNode:
    """Extract triples from a node."""
    text = node.get_content(metadata_mode=MetadataMode.LLM)
    try:
        llm_response = await self.llm.apredict(
            self.extract_prompt,
            text=text,
            max_knowledge_triplets=self.max_paths_per_chunk,
        )
        triples = self.parse_fn(llm_response)
    except ValueError:
        triples = []

    existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
    existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

    metadata = node.metadata.copy()
    for subj, rel, obj in triples:
        subj_node = EntityNode(name=subj, properties=metadata)
        obj_node = EntityNode(name=obj, properties=metadata)
        rel_node = Relation(
            label=rel,
            source_id=subj_node.id,
            target_id=obj_node.id,
            properties=metadata,
        )

        existing_nodes.extend([subj_node, obj_node])
        existing_relations.append(rel_node)

    #最后结果放在 node.metadata 字段中
    node.metadata[KG_NODES_KEY] = existing_nodes
    node.metadata[KG_RELATIONS_KEY] = existing_relations

    return node
```

# 四、存储(storage包)

## 1、StorageContext

### 1.1 用法

```python
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext

# create storage context using default stores
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)
```

### 1.2 from_defaults

```python
@classmethod
def from_defaults(
    cls,
    docstore: Optional[BaseDocumentStore] = None,
    index_store: Optional[BaseIndexStore] = None,
    vector_store: Optional[BasePydanticVectorStore] = None,
    image_store: Optional[BasePydanticVectorStore] = None,
    vector_stores: Optional[Dict[str, BasePydanticVectorStore]] = None,
    graph_store: Optional[GraphStore] = None,
    property_graph_store: Optional[PropertyGraphStore] = None,
    persist_dir: Optional[str] = None,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> "StorageContext":
    """Create a StorageContext from defaults.
    Args:
        docstore (Optional[BaseDocumentStore]): document store
        index_store (Optional[BaseIndexStore]): index store
        vector_store (Optional[BasePydanticVectorStore]): vector store
        graph_store (Optional[GraphStore]): graph store
        image_store (Optional[BasePydanticVectorStore]): image store
    """
    if persist_dir is None:
        docstore = docstore or SimpleDocumentStore()
        index_store = index_store or SimpleIndexStore()
        graph_store = graph_store or SimpleGraphStore()
        image_store = image_store or SimpleVectorStore()

        if vector_store:
            vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
        else:
            vector_stores = vector_stores or {
                DEFAULT_VECTOR_STORE: SimpleVectorStore()
            }
        if image_store:
            # append image store to vector stores
            vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store
    else:
        docstore = docstore or SimpleDocumentStore.from_persist_dir(
            persist_dir, fs=fs
        )
        index_store = index_store or SimpleIndexStore.from_persist_dir(
            persist_dir, fs=fs
        )
        graph_store = graph_store or SimpleGraphStore.from_persist_dir(
            persist_dir, fs=fs
        )

        try:
            property_graph_store = (
                property_graph_store
                or SimplePropertyGraphStore.from_persist_dir(persist_dir, fs=fs)
            )
        except FileNotFoundError:
            property_graph_store = None

        if vector_store:
            vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
        elif vector_stores:
            vector_stores = vector_stores
        else:
            vector_stores = SimpleVectorStore.from_namespaced_persist_dir(
                persist_dir, fs=fs
            )
        if image_store:
            # append image store to vector stores
            vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store  # type: ignore

    return cls(
        docstore=docstore,
        index_store=index_store,
        vector_stores=vector_stores,  # type: ignore
        graph_store=graph_store,
        property_graph_store=property_graph_store,
    )
```

# 五、索引对象(indices包)

## 1、BaseIndex

### 1.1 from_documents

处理 Document/Node 数据，包括：split/extractor/embed/store等步骤

```python
def from_documents(
    cls: Type[IndexType],
    documents: Sequence[Document],
    storage_context: Optional[StorageContext] = None,
    show_progress: bool = False,
    callback_manager: Optional[CallbackManager] = None,
    transformations: Optional[List[TransformComponent]] = None,
    **kwargs: Any,
) -> IndexType:
    """Create index from documents.
    Args:
        documents (Optional[Sequence[BaseDocument]]): List of documents to build the index from.
    """
    storage_context = storage_context or StorageContext.from_defaults()
    docstore = storage_context.docstore
    callback_manager = callback_manager or Settings.callback_manager
    transformations = transformations or Settings.transformations

    with callback_manager.as_trace("index_construction"):
        for doc in documents:
            docstore.set_document_hash(doc.get_doc_id(), doc.hash)

        #执行 transformComponent 节点(split/extractor/embed/store等步骤)
        nodes = run_transformations(
            documents,  # type: ignore
            transformations,
            show_progress=show_progress,
            **kwargs,
        )

        #返回对应的继承类
        return cls(
            nodes=nodes,
            storage_context=storage_context,
            callback_manager=callback_manager,
            show_progress=show_progress,
            transformations=transformations,
            **kwargs,
        )
```

### 1.2 as_retriever

```python
@abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        ...
```

### 1.3 as_query_engine

```python
def as_query_engine(
    self, llm: Optional[LLMType] = None, **kwargs: Any
) -> BaseQueryEngine:
    """Convert the index to a query engine.

    Calls `index.as_retriever(**kwargs)` to get the retriever and then wraps it in a
    `RetrieverQueryEngine.from_args(retriever, **kwrags)` call.
    """
    # NOTE: lazy import
    from llama_index.core.query_engine.retriever_query_engine import (
        RetrieverQueryEngine,
    )

    #获取检索器(BaseRetriever)
    retriever = self.as_retriever(**kwargs)
    llm = (
        resolve_llm(llm, callback_manager=self._callback_manager)
        if llm
        else Settings.llm
    )

    #返回对应的 QueryEngine
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        **kwargs,
    )
```

### 1.4 as_chat_engine

```python
def as_chat_engine(
    self,
    chat_mode: ChatMode = ChatMode.BEST,
    llm: Optional[LLMType] = None,
    **kwargs: Any,
) -> BaseChatEngine:
    """Convert the index to a chat engine.

    Calls `index.as_query_engine(llm=llm, **kwargs)` to get the query engine and then
    wraps it in a chat engine based on the chat mode.

    Chat modes:
     - ChatMode.BEST(default): Chat engine that uses an agent(react or openai) with a query engine tool
     - ChatMode.CONTEXT`: Chat engine that uses a retriever to get context
     - ChatMode.CONDENSE_QUESTION: Chat engine that condenses questions
     - ChatMode.CONDENSE_PLUS_CONTEXT: Chat engine that condenses questions and uses a retriever to get context
     - ChatMode.SIMPLE: Simple chat engine that uses the LLM directly
     - ChatMode.REACT: Chat engine that uses a react agent with a query engine tool
     - ChatMode.OPENAI: Chat engine that uses an openai agent with a query engine tool
    """
    llm = (
        resolve_llm(llm, callback_manager=self._callback_manager)
        if llm
        else Settings.llm
    )

    query_engine = self.as_query_engine(llm=llm, **kwargs)

    # resolve chat mode
    if chat_mode in [ChatMode.REACT, ChatMode.OPENAI, ChatMode.BEST]:
        # use an agent with query engine tool in these chat modes
        # NOTE: lazy import
        from llama_index.core.agent import AgentRunner
        from llama_index.core.tools.query_engine import QueryEngineTool

        # convert query engine to tool
        query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)

        return AgentRunner.from_llm(
            tools=[query_engine_tool],
            llm=llm,
            **kwargs,
        )

    if chat_mode == ChatMode.CONDENSE_QUESTION:
        # NOTE: lazy import
        from llama_index.core.chat_engine import CondenseQuestionChatEngine

        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            llm=llm,
            **kwargs,
        )
    elif chat_mode == ChatMode.CONTEXT:
        from llama_index.core.chat_engine import ContextChatEngine

        return ContextChatEngine.from_defaults(
            retriever=self.as_retriever(**kwargs),
            llm=llm,
            **kwargs,
        )

    elif chat_mode == ChatMode.CONDENSE_PLUS_CONTEXT:
        from llama_index.core.chat_engine import CondensePlusContextChatEngine

        return CondensePlusContextChatEngine.from_defaults(
            retriever=self.as_retriever(**kwargs),
            llm=llm,
            **kwargs,
        )

    elif chat_mode == ChatMode.SIMPLE:
        from llama_index.core.chat_engine import SimpleChatEngine

        return SimpleChatEngine.from_defaults(
            llm=llm,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown chat mode: {chat_mode}")
```

## 2、增删改查

### 2.1 insert

#### (1) 案例

```python
from llama_index.core import SummaryIndex, Document

index = SummaryIndex([])
text_chunks = ["text_chunk_1", "text_chunk_2", "text_chunk_3"]

doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text=text, id_=f"doc_id_{i}")
    doc_chunks.append(doc)

# insert
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)
```

#### (2) 核心代码

```python
def insert(self, document: Document, **insert_kwargs: Any) -> None:
    """Insert a document."""
    with self._callback_manager.as_trace("insert"):
        nodes = run_transformations(
            [document],
            self._transformations,
            show_progress=self._show_progress,
            **insert_kwargs,
        )

        self.insert_nodes(nodes, **insert_kwargs)
        self.docstore.set_document_hash(document.get_doc_id(), document.hash)
        
def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
    """Insert nodes."""
    for node in nodes:
        if isinstance(node, IndexNode):
            try:
                node.dict()
            except ValueError:
                self._object_map[node.index_id] = node.obj
                node.obj = None

    with self._callback_manager.as_trace("insert_nodes"):
        self.docstore.add_documents(nodes, allow_update=True) #doc
        self._insert(nodes, **insert_kwargs) #emb/graph 等数据
        self._storage_context.index_store.add_index_struct(self._index_struct) #索引(emb不需要)
```

### 2.2 delete

#### (1) 案例

```python
index.delete_ref_doc("doc_id_0", delete_from_docstore=True)
```

#### (2) 核心代码

```python
def delete_ref_doc(
    self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
) -> None:
    """Delete a document and it's nodes by using ref_doc_id."""
    ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
    if ref_doc_info is None:
        logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
        return

    self.delete_nodes(
        ref_doc_info.node_ids,
        delete_from_docstore=False,
        **delete_kwargs,
    )

    if delete_from_docstore:
        self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
        
def delete_nodes(
    self,
    node_ids: List[str],
    delete_from_docstore: bool = False,
    **delete_kwargs: Any,
) -> None:
    for node_id in node_ids:
        self._delete_node(node_id, **delete_kwargs)
        if delete_from_docstore:
            self.docstore.delete_document(node_id, raise_error=False)
    self._storage_context.index_store.add_index_struct(self._index_struct)
```

### 2.3 update

#### (1) 案例

```python
# NOTE: the document has a `doc_id` specified
doc_chunks[0].text = "Brand new document text"
index.update_ref_doc(doc_chunks[0])
```

#### (2) 核心代码

```python
def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
    """Update a document and it's corresponding nodes.
    This is equivalent to deleting the document and then inserting it again.
    Args:
        document (Union[BaseDocument, BaseIndex]): document to update
        insert_kwargs (Dict): kwargs to pass to insert
        delete_kwargs (Dict): kwargs to pass to delete
    """
    with self._callback_manager.as_trace("update"):
        self.delete_ref_doc(
            document.get_doc_id(),
            delete_from_docstore=True,
            **update_kwargs.pop("delete_kwargs", {}),
        )
        self.insert(document, **update_kwargs.pop("insert_kwargs", {}))
```

### 2.4 refresh

#### (1) 案例

```python
# modify first document, with the same doc_id
doc_chunks[0] = Document(text="Super new document text", id_="doc_id_0")

# add a new document
doc_chunks.append(
    Document(
        text="This isn't in the index yet, but it will be soon!",
        id_="doc_id_3",
    )
)

# refresh the index
refreshed_docs = index.refresh_ref_docs(doc_chunks)

# refreshed_docs[0] and refreshed_docs[-1] should be true
```

#### (2) 核心代码

```python
def refresh_ref_docs(
    self, documents: Sequence[Document], **update_kwargs: Any
) -> List[bool]:
    """Refresh an index with documents that have changed.
    This allows users to save LLM and Embedding model calls, while only
    updating documents that have any changes in text or metadata. It
    will also insert any documents that previously were not stored.
    """
    with self._callback_manager.as_trace("refresh"):
        refreshed_documents = [False] * len(documents)
        for i, document in enumerate(documents):
            existing_doc_hash = self._docstore.get_document_hash(
                document.get_doc_id()
            )
            if existing_doc_hash is None:
                self.insert(document, **update_kwargs.pop("insert_kwargs", {}))
                refreshed_documents[i] = True
            elif existing_doc_hash != document.hash:
                self.update_ref_doc(
                    document, **update_kwargs.pop("update_kwargs", {})
                )
                refreshed_documents[i] = True
        return refreshed_documents
```

## 3、VectorStoreIndex

### 3.1 基本用法

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader("../../examples/data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
```

## 4、PropertyGraphIndex

### 4.1 基本用法

```python
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)
```

# 六、检索

## 1、BaseIndex.as_retriever 用法

```python
retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")
```

## 2、BaseRetriever.retrieve

```python
@dispatcher.span
def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
    """Retrieve nodes given query.
    Args:
        str_or_query_bundle (QueryType): Either a query string or a QueryBundle object.
    """
    self._check_callback_manager()
    dispatcher.event(
        RetrievalStartEvent(
            str_or_query_bundle=str_or_query_bundle,
        )
    )
    if isinstance(str_or_query_bundle, str):
        query_bundle = QueryBundle(str_or_query_bundle)
    else:
        query_bundle = str_or_query_bundle
    with self.callback_manager.as_trace("query"):
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as retrieve_event:
            nodes = self._retrieve(query_bundle) #检索
            nodes = self._handle_recursive_retrieval(query_bundle, nodes)
            retrieve_event.on_end(
                payload={EventPayload.NODES: nodes},
            )
    dispatcher.event(
        RetrievalEndEvent(
            str_or_query_bundle=str_or_query_bundle,
            nodes=nodes,
        )
    )
    return nodes
```

## 3、`_retrieve` 函数

### 3.1 VectorIndexRetriever

#### (1) `_retrieve`

```python
@dispatcher.span
def _retrieve(
    self,
    query_bundle: QueryBundle,
) -> List[NodeWithScore]:
    if self._vector_store.is_embedding_query:
        if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
            query_bundle.embedding = (
                self._embed_model.get_agg_embedding_from_queries(query_bundle.embedding_strs)
            )
    return self._get_nodes_with_embeddings(query_bundle)
```

#### (2) BaseEmbedding.get_agg_embedding_from_queries

```python
def get_agg_embedding_from_queries(
    self,
    queries: List[str],
    agg_fn: Optional[Callable[..., Embedding]] = None,
) -> Embedding:
    """Get aggregated embedding from multiple queries."""
    query_embeddings = [self.get_query_embedding(query) for query in queries]
    agg_fn = agg_fn or mean_agg
    return agg_fn(query_embeddings)

@dispatcher.span
def get_query_embedding(self, query: str) -> Embedding:
    """
    Embed the input query.

    When embedding a query, depending on the model, a special instruction
    can be prepended to the raw query string. For example, "Represent the
    question for retrieving supporting documents: ". If you're curious,
    other examples of predefined instructions can be found in
    embeddings/huggingface_utils.py.
    """
    model_dict = self.to_dict()
    model_dict.pop("api_key", None)
    dispatcher.event(EmbeddingStartEvent(model_dict=model_dict,))
    with self.callback_manager.event(
        CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
    ) as event:
        query_embedding = self._get_query_embedding(query)
        event.on_end(payload={EventPayload.CHUNKS: [query], EventPayload.EMBEDDINGS: [query_embedding],]},)
    dispatcher.event(EmbeddingEndEvent(chunks=[query],embeddings=[query_embedding],))
    return query_embedding

@abstractmethod
def _get_query_embedding(self, query: str) -> Embedding:
    """
    Embed the input query synchronously.

    Subclasses should implement this method. Reference get_query_embedding's
    docstring for more information.
    """
```

#### (3) VectorIndexRetriever._get_nodes_with_embeddings

```python
def _get_nodes_with_embeddings(
    self, query_bundle_with_embeddings: QueryBundle
) -> List[NodeWithScore]:
    query = self._build_vector_store_query(query_bundle_with_embeddings)
    query_result = self._vector_store.query(query, **self._kwargs)
    return self._build_node_list_from_query_result(query_result)
```

#### (4) VectorStoreQuery

```python
@dataclass
class VectorStoreQuery:
    """Vector store query."""
    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 1
    doc_ids: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    query_str: Optional[str] = None
    output_fields: Optional[List[str]] = None
    embedding_field: Optional[str] = None

    mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT

    # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    alpha: Optional[float] = None

    # metadata filters
    filters: Optional[MetadataFilters] = None

    # only for mmr
    mmr_threshold: Optional[float] = None

    # NOTE: currently only used by postgres hybrid search
    sparse_top_k: Optional[int] = None
    # NOTE: return top k results from hybrid search. similarity_top_k is used for dense search top k
    hybrid_top_k: Optional[int] = None
```

#### (5) SimpleVectorStore.query

```python
def query(
    self,
    query: VectorStoreQuery,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    # Prevent metadata filtering on stores that were persisted without metadata.
    if (
        query.filters is not None
        and self.data.embedding_dict
        and not self.data.metadata_dict
    ):
        raise ValueError(
            "Cannot filter stores that were persisted without metadata. "
            "Please rebuild the store with metadata to enable filtering."
        )
    # Prefilter nodes based on the query filter and node ID restrictions.
    query_filter_fn = _build_metadata_filter_fn(
        lambda node_id: self.data.metadata_dict[node_id], query.filters
    )

    if query.node_ids is not None:
        available_ids = set(query.node_ids)

        def node_filter_fn(node_id: str) -> bool:
            return node_id in available_ids

    else:

        def node_filter_fn(node_id: str) -> bool:
            return True

    node_ids = []
    embeddings = []
    # TODO: consolidate with get_query_text_embedding_similarities
    for node_id, embedding in self.data.embedding_dict.items():
        if node_filter_fn(node_id) and query_filter_fn(node_id):
            node_ids.append(node_id)
            embeddings.append(embedding)

    query_embedding = cast(List[float], query.query_embedding)

    if query.mode in LEARNER_MODES:
        top_similarities, top_ids = get_top_k_embeddings_learner(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
        )
    elif query.mode == MMR_MODE:
        mmr_threshold = kwargs.get("mmr_threshold", None)
        top_similarities, top_ids = get_top_k_mmr_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
            mmr_threshold=mmr_threshold,
        )
    elif query.mode == VectorStoreQueryMode.DEFAULT:
        top_similarities, top_ids = get_top_k_embeddings(
            query_embedding,
            embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=node_ids,
        )
    else:
        raise ValueError(f"Invalid query mode: {query.mode}")

    return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
```

### 3.2 PGRetriever

#### (1) `_retrieve`

```python
def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    results = []
    if self.use_async:
        return asyncio_run(self._aretrieve(query_bundle))

    for sub_retriever in tqdm(self.sub_retrievers, disable=not self.show_progress):
        results.extend(sub_retriever.retrieve(query_bundle)) #sub_retriever 为 BasePGRetriever

    return self._deduplicate(results)
```

#### (2) BasePGRetriever._retrieve

```python
def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    nodes = self.retrieve_from_graph(query_bundle)
    if self.include_text and nodes:
        nodes = self.add_source_text(nodes)
    return nodes
```

# 七、postprocessor(rerank)

## 1、BaseNodePostprocessor

### 1.1 postprocess_nodes

```python
def postprocess_nodes(
    self,
    nodes: List[NodeWithScore],
    query_bundle: Optional[QueryBundle] = None,
    query_str: Optional[str] = None,
) -> List[NodeWithScore]:
    """Postprocess nodes."""
    if query_str is not None and query_bundle is not None:
        raise ValueError("Cannot specify both query_str and query_bundle")
    elif query_str is not None:
        query_bundle = QueryBundle(query_str)
    else:
        pass
    return self._postprocess_nodes(nodes, query_bundle)
```

### 1.2 `_postprocess_nodes`

```python
@abstractmethod
def _postprocess_nodes(
    self,
    nodes: List[NodeWithScore],
    query_bundle: Optional[QueryBundle] = None,
) -> List[NodeWithScore]:
    """Postprocess nodes."""
```

## 2、使用案例

```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

nodes = [
    NodeWithScore(node=Node(text="text1"), score=0.7),
    NodeWithScore(node=Node(text="text2"), score=0.8),
]

# similarity postprocessor: filter nodes below 0.75 similarity score
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
filtered_nodes = processor.postprocess_nodes(nodes)

# cohere rerank: rerank nodes given query using trained model
reranker = CohereRerank(api_key="<COHERE_API_KEY>", top_n=2)
reranker.postprocess_nodes(nodes, query_str="<user_query>")

#NodeWithScore
class NodeWithScore(BaseComponent):
    node: SerializeAsAny[BaseNode]
    score: Optional[float] = None
```

### 2.1 SimilarityPostprocessor

```python
def _postprocess_nodes(
    self,
    nodes: List[NodeWithScore],
    query_bundle: Optional[QueryBundle] = None,
) -> List[NodeWithScore]:
    """Postprocess nodes."""
    sim_cutoff_exists = self.similarity_cutoff is not None

    new_nodes = []
    for node in nodes:
        should_use_node = True
        if sim_cutoff_exists:
            similarity = node.score
            if similarity is None:
                should_use_node = False
            elif cast(float, similarity) < cast(float, self.similarity_cutoff):
                should_use_node = False

        if should_use_node:
            new_nodes.append(node)

    return new_nodes
```

### 2.2 CohereRerank

```python
def _postprocess_nodes(
    self,
    nodes: List[NodeWithScore],
    query_bundle: Optional[QueryBundle] = None,
) -> List[NodeWithScore]:
    dispatcher.event(
        ReRankStartEvent(query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model)
    )

    if query_bundle is None:
        raise ValueError("Missing query bundle in extra info.")
    if len(nodes) == 0:
        return []

    with self.callback_manager.event(
        CBEventType.RERANKING,
        payload={
            EventPayload.NODES: nodes,
            EventPayload.MODEL_NAME: self.model,
            EventPayload.QUERY_STR: query_bundle.query_str,
            EventPayload.TOP_K: self.top_n,
        },
    ) as event:
        texts = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED)
            for node in nodes
        ]
        #调用模型计算分数
        results = self._client.rerank(
            model=self.model,
            top_n=self.top_n,
            query=query_bundle.query_str,
            documents=texts,
        )

        new_nodes = []
        for result in results.results:
            new_node_with_score = NodeWithScore(
                node=nodes[result.index].node, score=result.relevance_score
            )
            new_nodes.append(new_node_with_score)
        event.on_end(payload={EventPayload.NODES: new_nodes})

    dispatcher.event(ReRankEndEvent(nodes=new_nodes))
    return new_nodes
```

# 八、结果合成(response_synthesizer)

## 1、BaseSynthesizer

### 1.1 synthesize

```python
@dispatcher.span
def synthesize(
    self,
    query: QueryTextType,
    nodes: List[NodeWithScore],
    additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    **response_kwargs: Any,
) -> RESPONSE_TYPE:
    dispatcher.event(SynthesizeStartEvent(query=query,))
    if len(nodes) == 0:
        if self._streaming:
            empty_response_stream = StreamingResponse(response_gen=empty_response_generator())
            dispatcher.event(SynthesizeEndEvent(query=query, response=empty_response_stream,))
            return empty_response_stream
        else:
            empty_response = Response("Empty Response")
            dispatcher.event(SynthesizeEndEvent(query=query,response=empty_response,))
            return empty_response

    if isinstance(query, str):
        query = QueryBundle(query_str=query)

    with self._callback_manager.event(
        CBEventType.SYNTHESIZE,
        payload={EventPayload.QUERY_STR: query.query_str},
    ) as event:
        #合并 response
        response_str = self.get_response(
            query_str=query.query_str,
            text_chunks=[
                n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
            ],
            **response_kwargs,
        )

        additional_source_nodes = additional_source_nodes or []
        source_nodes = list(nodes) + list(additional_source_nodes)
        #将 response 转换为对应的类型：PydanticResponse、Response、StreamingResponse、AsyncStreamingResponse等
        response = self._prepare_response_output(response_str, source_nodes)
        event.on_end(payload={EventPayload.RESPONSE: response})

    dispatcher.event(SynthesizeEndEvent(query=query,response=response,))
    return response
```

### 1.2 get_response

```python
@abstractmethod
def get_response(
    self,
    query_str: str,
    text_chunks: Sequence[str],
    **response_kwargs: Any,
) -> RESPONSE_TEXT_TYPE:
    """Get response."""
    ...
```

## 2、使用案例

```python
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    response_mode="refine", #指定 response_synthesizer
    service_context=service_context, #定义 llm 相关配置
    text_qa_template=text_qa_template, #匹配的 prompt
    refine_template=refine_template, #匹配的 prompt
    use_async=False,
    streaming=False,
)

# synchronous
response = response_synthesizer.synthesize(
    "query string",
    nodes=[NodeWithScore(node=Node(text="text"), score=1.0), ...],
    additional_source_nodes=[
        NodeWithScore(node=Node(text="text"), score=1.0),
        ...,
    ],
)

# asynchronous
response = await response_synthesizer.asynthesize(
    "query string",
    nodes=[NodeWithScore(node=Node(text="text"), score=1.0), ...],
    additional_source_nodes=[
        NodeWithScore(node=Node(text="text"), score=1.0),
        ...,
    ],
)
```

# 九、查询引擎(query engine)

## 1、用法

### 1.1 高集成用法

```python
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    verbose=True,
)

#BaseIndex.as_query_engine
def as_query_engine(
    self, llm: Optional[LLMType] = None, **kwargs: Any
) -> BaseQueryEngine:
    # NOTE: lazy import
    from llama_index.core.query_engine.retriever_query_engine import (
        RetrieverQueryEngine,
    )

    #获取检索器
    retriever = self.as_retriever(**kwargs)
    llm = (
        resolve_llm(llm, callback_manager=self._callback_manager)
        if llm else Settings.llm
    )
    #构建检索引擎
    return RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs,)
```

### 1.2 低集成用法

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What did the author do growing up?")
print(response)
```

## 2、BaseQueryEngine 与 RetrieverQueryEngine

### 2.1 BaseQueryEngine

```python
@dispatcher.span
def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
    dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
    with self.callback_manager.as_trace("query"):
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        query_result = self._query(str_or_query_bundle) #调用查询函数
    dispatcher.event(
        QueryEndEvent(query=str_or_query_bundle, response=query_result)
    )
    return query_result
```

### 2.2 RetrieverQueryEngine

#### (1) from_args

```python
@classmethod
def from_args(
    cls,
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    response_synthesizer: Optional[BaseSynthesizer] = None,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    callback_manager: Optional[CallbackManager] = None,
    # response synthesizer args
    response_mode: ResponseMode = ResponseMode.COMPACT,
    text_qa_template: Optional[BasePromptTemplate] = None,
    refine_template: Optional[BasePromptTemplate] = None,
    summary_template: Optional[BasePromptTemplate] = None,
    simple_template: Optional[BasePromptTemplate] = None,
    output_cls: Optional[Type[BaseModel]] = None,
    use_async: bool = False,
    streaming: bool = False,
    **kwargs: Any,
) -> "RetrieverQueryEngine":
    """Initialize a RetrieverQueryEngine object.".

    Args:
        retriever (BaseRetriever): A retriever object.
        llm (Optional[LLM]): An instance of an LLM.
        response_synthesizer (Optional[BaseSynthesizer]): An instance of a response synthesizer.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of node postprocessors.
        callback_manager (Optional[CallbackManager]): A callback manager.
        response_mode (ResponseMode): A ResponseMode object.
        text_qa_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
        refine_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
        summary_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
        simple_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
        output_cls (Optional[Type[BaseModel]]): The pydantic model to pass to the response synthesizer.
        use_async (bool): Whether to use async.
        streaming (bool): Whether to use streaming.
    """
    llm = llm or Settings.llm

    response_synthesizer = response_synthesizer or get_response_synthesizer(
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        summary_template=summary_template,
        simple_template=simple_template,
        response_mode=response_mode,
        output_cls=output_cls,
        use_async=use_async,
        streaming=streaming,
    )

    callback_manager = callback_manager or Settings.callback_manager

    return cls(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        callback_manager=callback_manager,
        node_postprocessors=node_postprocessors,
    )
```

#### (2) `_query`

```python
@dispatcher.span
def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
    with self.callback_manager.event(
        CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
    ) as query_event:
        #检索
        nodes = self.retrieve(query_bundle)
        #结果合成
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )
        query_event.on_end(payload={EventPayload.RESPONSE: response})

    return response
```

# 十、路由(router/selectors)

## 1、用法

### 1.1 独立用法

```python
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector

# choices as a list of tool metadata
choices = [
    ToolMetadata(description="description for choice 1", name="choice_1"),
    ToolMetadata(description="description for choice 2", name="choice_2"),
]

# choices as a list of strings
choices = [
    "choice 1 - description for choice 1",
    "choice 2: description for choice 2",
]

selector = LLMSingleSelector.from_defaults()
selector_result = selector.select(
    choices, query="What's revenue growth for IBM in 2007?"
)
print(selector_result.selections)
```

### 1.2 query engine 用法

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.selectors.pydantic_selectors import Pydantic
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, SummaryIndex

# define query engines
...

# initialize tools
list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

# initialize router query engine (single selection, pydantic)
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
query_engine.query("<query>")
```

### 1.3 retriver 用法

```python
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import RetrieverTool

# define indices
...

# define retrievers
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()

# initialize tools
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On (using entities mentioned in query)",
)

# define retriever
retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)
```

### 1.4 ToolMetadata

```python
@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema
    return_direct: bool = False
```

### 1.5 SelectorResult

```python
SelectorResult = MultiSelection

class MultiSelection(BaseModel):
    selections: List[SingleSelection]
    
class SingleSelection(BaseModel):
    index: int
    reason: str
```

## 2、BaseSelector

```python
def select(
    self, choices: Sequence[MetadataType], query: QueryType
) -> SelectorResult:
    metadatas = [_wrap_choice(choice) for choice in choices]
    query_bundle = _wrap_query(query)
    return self._select(choices=metadatas, query=query_bundle)
```

## 3、LLMSingleSelector

### 3.1 from_defaults

```python
@classmethod
def from_defaults(
    cls,
    llm: Optional[LLM] = None,
    prompt_template_str: Optional[str] = None,
    output_parser: Optional[BaseOutputParser] = None,
) -> "LLMSingleSelector":
    # optionally initialize defaults
    llm = llm or Settings.llm
    prompt_template_str = prompt_template_str or DEFAULT_SINGLE_SELECT_PROMPT_TMPL
    output_parser = output_parser or SelectionOutputParser()

    # construct prompt
    prompt = SingleSelectPrompt(
        template=prompt_template_str,
        output_parser=output_parser,
        prompt_type=PromptType.SINGLE_SELECT,
    )
    return cls(llm, prompt)
```

### 3.2 `_select`

```python
def _select(
    self, choices: Sequence[ToolMetadata], query: QueryBundle
) -> SelectorResult:
    # prepare input
    choices_text = _build_choices_text(choices)

    # predict
    prediction = self._llm.predict(
        prompt=self._prompt,
        num_choices=len(choices),
        context_list=choices_text,
        query_str=query.query_str,
    )

    # parse output
    assert self._prompt.output_parser is not None
    parse = self._prompt.output_parser.parse(prediction)
    return _structured_output_to_selector_result(parse)
```

# 十一、agent

## 1、BaseAgent

### 1.1 BaseAgent

```python
class BaseAgent(BaseChatEngine, BaseQueryEngine):
    
    @trace_method("query")
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(
            response=str(agent_response), source_nodes=agent_response.source_nodes
        )
        
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")
```

### 1.2 BaseAgentRunner

```python
class BaseAgentRunner(BaseAgent):
    @abstractmethod
    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""

    @abstractmethod
    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """Delete task.
        NOTE: this will not delete any previous executions from memory.
        """

    @abstractmethod
    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""

    @abstractmethod
    def get_completed_tasks(self, **kwargs: Any) -> List[Task]:
        """Get completed tasks."""

    @abstractmethod
    def get_task_output(self, task_id: str, **kwargs: Any) -> TaskStepOutput:
        """Get task output."""

    @abstractmethod
    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""

    @abstractmethod
    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""

    @abstractmethod
    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""

    def get_completed_step(
        self, task_id: str, step_id: str, **kwargs: Any
    ) -> TaskStepOutput:
        """Get completed step."""
        # call get_completed_steps, and then find the right task
        completed_steps = self.get_completed_steps(task_id, **kwargs)
        for step_output in completed_steps:
            if step_output.task_step.step_id == step_id:
                return step_output
        raise ValueError(f"Could not find step_id: {step_id}")

    @abstractmethod
    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""

    @abstractmethod
    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""

    @abstractmethod
    def finalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""

    @abstractmethod
    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")
```

### 1.3 BaseAgentWorker

```python
class BaseAgentWorker(PromptMixin, DispatcherSpanMixin):
    @abstractmethod
    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""

    @abstractmethod
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        
    @abstractmethod
    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
```

## 2、AgentRunner 与AgentWorker

### 2.1 解释说明

- `AgentRunner`：是存储状态(包括对话内存)、创建和维护任务、通过每个任务运行步骤、提供面向用户的高级接口
- `AgentWorker`：控制任务的逐步执行，即**给定 input step，agentWorker 负责生成下一步**。
    - 可以用参数初始化，并对从任务/任务步骤对象传递的状态作用，但不能固有地存储自己的状态
    - 外部 `AgentRunner` 负责调用 `AgentWorker` 并收集/汇总结果

### 2.2 辅助类

- `Task` ：高级任务，接入用户查询 +沿其他信息（例如内存）

- `TaskStep` ：表示一个 step。将其作为输入给 `AgentWorker` ，并返回 `TaskStepOutput` 

    > 完成一个 `Task` 可能涉及多个 `TaskStep`

- `TaskStepOutput`：从给定的 step 执行中输出，输出是否完成任务

#### (1) Task

```python
class Task(BaseModel):
    """Agent Task.
    Represents a "run" of an agent given a user input.
    """
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Task ID"
    )
    input: str = Field(..., description="User input")
    memory: SerializeAsAny[BaseMemory] = Field(
        ...,
        description=("Conversational Memory. Maintains state before execution of this task."),
    )
    extra_state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional user-specified state for a given task. "
            "Can be modified throughout the execution of a task."
        ),
    )
```

#### (2) TaskStep

```python
class TaskStep(BaseModel):
    """Agent task step.
    Represents a single input step within the execution run ("Task") of an agent given a user input.
    The output is returned as a `TaskStepOutput`.
    """
    task_id: str = Field(..., description="Task ID")
    step_id: str = Field(..., description="Step ID")
    input: Optional[str] = Field(default=None, description="User input")
    step_state: Dict[str, Any] = Field(
        default_factory=dict, description="Additional state for a given step."
    )
    next_steps: Dict[str, "TaskStep"] = Field(
        default_factory=dict, description="Next steps to be executed."
    )
    prev_steps: Dict[str, "TaskStep"] = Field(
        default_factory=dict,
        description="Previous steps that were dependencies for this step.",
    )
    is_ready: bool = Field(
        default=True, description="Is this step ready to be executed?"
    )
```

#### (3) TaskStepOutput

```python
class TaskStepOutput(BaseModel):
    """Agent task step output."""
    output: Any = Field(..., description="Task step output")
    task_step: TaskStep = Field(..., description="Task step input")
    next_steps: List[TaskStep] = Field(..., description="Next steps to be executed.")
    is_last: bool = Field(default=False, description="Is this the last step?")
```

#### (4) TaskState

```python
class TaskState(BaseModel):
    task: Task = Field(..., description="Task.")
    step_queue: Deque[TaskStep] = Field(
        default_factory=deque, description="Task step queue."
    )
    completed_steps: List[TaskStepOutput] = Field(
        default_factory=list, description="Completed step outputs."
    )
```

#### (5) AgentState

```python
class AgentState(BaseModel):
    task_dict: Dict[str, TaskState] = Field(
        default_factory=dict, description="Task dictionary."
    )
```

### 2.3 使用案例

```python
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker

# construct OpenAIAgent from tools
openai_step_engine = OpenAIAgentWorker.from_tools(tools, llm=llm, verbose=True)
agent = AgentRunner(openai_step_engine)

# create task
task = agent.create_task("What is (121 * 3) + 42?")

# execute step
step_output = agent.run_step(task)

# if step_output is done, finalize response
if step_output.is_last:
    response = agent.finalize_response(task.task_id)

# list tasks
task.list_tasks()

# get completed steps
task.get_completed_steps(task.task_id)

print(str(response))
```

#### (1) AgentRunner.create_task

```python
def create_task(self, input: str, **kwargs: Any) -> Task:
    if not self.init_task_state_kwargs:
        extra_state = kwargs.pop("extra_state", {})
    else:
        if "extra_state" in kwargs:
            raise ValueError("Cannot specify both `extra_state` and `init_task_state_kwargs`")
        else:
            extra_state = self.init_task_state_kwargs

    callback_manager = kwargs.pop("callback_manager", self.callback_manager)
    task = Task(
        input=input,
        memory=self.memory,
        extra_state=extra_state,
        callback_manager=callback_manager,
        **kwargs,
    )
    # get initial step from task, and put it in the step queue
    initial_step = self.agent_worker.initialize_step(task)
    task_state = TaskState(
        task=task,
        step_queue=deque([initial_step]),
    )
    # add it to state
    self.state.task_dict[task.task_id] = task_state

    return task
```

#### (2) OpenAIAgentWorker.initialize_step

```python
def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
    sources: List[ToolOutput] = []
    new_memory = ChatMemoryBuffer.from_defaults() # temporary memory for new messages
    # initialize task state
    task_state = {
        "sources": sources,
        "n_function_calls": 0,
        "new_memory": new_memory,
    }
    task.extra_state.update(task_state)

    return TaskStep(
        task_id=task.task_id,
        step_id=str(uuid.uuid4()),
        input=task.input,
    )
```

#### (3) AgentRunner.run_step

```python
@dispatcher.span
def _run_step(
    self,
    task_id: str,
    step: Optional[TaskStep] = None,
    input: Optional[str] = None,
    mode: ChatResponseMode = ChatResponseMode.WAIT,
    **kwargs: Any,
) -> TaskStepOutput:
    #获取任务和 step
    task = self.state.get_task(task_id)
    step_queue = self.state.get_step_queue(task_id)
    step = step or step_queue.popleft()
    if input is not None:
        step.input = input

    dispatcher.event(AgentRunStepStartEvent(task_id=task_id, step=step, input=input))

    #执行
    if mode == ChatResponseMode.WAIT:
        cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
    elif mode == ChatResponseMode.STREAM:
        cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    #结果预处理
    # append cur_step_output next steps to queue
    next_steps = cur_step_output.next_steps
    step_queue.extend(next_steps)
    # add cur_step_output to completed steps
    completed_steps = self.state.get_completed_steps(task_id)
    completed_steps.append(cur_step_output)

    dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
    return cur_step_output
```

#### (4) OpenAIAgentWorker.run_step

```python
    def _run_step(
        self,
        step: TaskStep,
        task: Task,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        tool_choice: Union[str, dict] = "auto",
    ) -> TaskStepOutput:
        if step.input is not None:
            add_user_step_to_memory(step, task.extra_state["new_memory"], verbose=self._verbose)
        #获取 tool
        tools = self.get_tools(task.input)
        openai_tools = [tool.metadata.to_openai_tool() for tool in tools]
        #获取 llm 参数：msg、tool等
        llm_chat_kwargs = self._get_llm_chat_kwargs(task, openai_tools, tool_choice)
        #模型调用
        agent_chat_response = self._get_agent_response(task, mode=mode, **llm_chat_kwargs)
        #结果获取(next tool/task)
        latest_tool_calls = self.get_latest_tool_calls(task) or []
        latest_tool_outputs: List[ToolOutput] = []
        #tool 数量判断，超过指定数量直接结束
        if not self._should_continue(latest_tool_calls, task.extra_state["n_function_calls"]):
            is_done = True
            new_steps = []
        else:
            is_done = False
            # tool 执行
            for tool_call in latest_tool_calls:
                # Some validation
                if not isinstance(tool_call, get_args(OpenAIToolCall)):
                    raise ValueError("Invalid tool_call object")
                if tool_call.type != "function":
                    raise ValueError("Invalid tool type. Unsupported by OpenAI")
                
                return_direct = self._call_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    latest_tool_outputs,
                )
                task.extra_state["sources"].append(latest_tool_outputs[-1])

                # change function call to the default value, if a custom function was given
                # as an argument (none and auto are predefined by OpenAI)
                if tool_choice not in ("auto", "none"):
                    tool_choice = "auto"
                task.extra_state["n_function_calls"] += 1

                if return_direct and len(latest_tool_calls) == 1:
                    is_done = True
                    response_str = latest_tool_outputs[-1].content
                    chat_response = ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str)
                    )
                    agent_chat_response = self._process_message(task, chat_response)
                    agent_chat_response.is_dummy_stream = (mode == ChatResponseMode.STREAM)
                    break

            #上面的判断条件没过时，step 重新包装返回
            new_steps = (
                [
                    step.get_next_step(step_id=str(uuid.uuid4()), input=None,)
                ]
                if not is_done
                else []
            )

        # Attach all tool outputs from this step as sources
        agent_chat_response.sources = latest_tool_outputs

        return TaskStepOutput(
            output=agent_chat_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )
```

#### (5) AgentRunner.finalize_response

```python
@dispatcher.span
def finalize_response(
    self,
    task_id: str,
    step_output: Optional[TaskStepOutput] = None,
) -> AGENT_CHAT_RESPONSE_TYPE:
    if step_output is None:
        step_output = self.state.get_completed_steps(task_id)[-1]
    if not step_output.is_last:
        raise ValueError("finalize_response can only be called on the last step output")

    if not isinstance( step_output.output, (AgentChatResponse, StreamingAgentChatResponse),):
        raise ValueError("When `is_last` is True, cur_step_output.output must be "
                        f"AGENT_CHAT_RESPONSE_TYPE: {step_output.output}")

    # finalize task(消息存储/重置等操作)
    self.agent_worker.finalize_task(self.state.get_task(task_id))

    if self.delete_task_on_finish:
        self.delete_task(task_id)

    # Attach all sources generated across all steps
    step_output.output.sources = self.get_task(task_id).extra_state.get("sources", [])
    step_output.output.set_source_nodes()

    return cast(AGENT_CHAT_RESPONSE_TYPE, step_output.output)
```

## 3、ReActAgent(AgentRunner)

### 3.1 案例

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
agent.chat("What is 2123 * 215123")
```

### 3.2 chat

```python
@dispatcher.span
def _chat(
    self,
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
    tool_choice: Union[str, dict] = "auto",
    mode: ChatResponseMode = ChatResponseMode.WAIT,
) -> AGENT_CHAT_RESPONSE_TYPE:
    if chat_history is not None:
        self.memory.set(chat_history)
    task = self.create_task(message) #创建任务

    result_output = None
    dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
    #循环运行任务
    while True:
        # pass step queue in as argument, assume step executor is stateless
        cur_step_output = self._run_step(task.task_id, mode=mode, tool_choice=tool_choice)

        if cur_step_output.is_last:
            result_output = cur_step_output
            break

        # ensure tool_choice does not cause endless loops
        tool_choice = "auto"

    result = self.finalize_response(
        task.task_id,
        result_output,
    )
    dispatcher.event(AgentChatWithStepEndEvent(response=result))
    return result
```

## 4、FunctionCallingAgent

### 4.1 FunctionCallingAgentWorker.run_step

```python
@trace_method("run_step")
def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
    if step.input is not None:
        add_user_step_to_memory(step, task.extra_state["new_memory"], verbose=self._verbose)
    # 获取 tools
    tools = self.get_tools(task.input)

    # get response and tool call (if exists)
    response = self._llm.chat_with_tools(
        tools=tools,
        user_msg=None,
        chat_history=self.get_all_messages(task),
        verbose=self._verbose,
        allow_parallel_tool_calls=self.allow_parallel_tool_calls,
    )

    #获取其中的 tools
    tool_calls = self._llm.get_tool_calls_from_response(
        response, error_on_no_tool_call=False
    )
    tool_outputs: List[ToolOutput] = []
    if not self.allow_parallel_tool_calls and len(tool_calls) > 1:
        raise ValueError(
            "Parallel tool calls not supported for synchronous function calling agent"
        )

    # call all tools, gather responses
    task.extra_state["new_memory"].put(response.message)
    if (
        len(tool_calls) == 0
        or task.extra_state["n_function_calls"] >= self._max_function_calls
    ):
        # we are done
        is_done = True
        new_steps = []
    else:
        is_done = False
        for i, tool_call in enumerate(tool_calls):
            # tool 执行
            return_direct = self._call_function(
                tools,
                tool_call,
                task.extra_state["new_memory"],
                tool_outputs,
                verbose=self._verbose,
            )
            task.extra_state["sources"].append(tool_outputs[-1])
            task.extra_state["n_function_calls"] += 1

            # check if any of the tools return directly -- only works if there is one tool call
            if i == 0 and return_direct:
                is_done = True
                response = task.extra_state["sources"][-1].content
                break

        # put tool output in sources and memory
        new_steps = (
            [
                step.get_next_step(step_id=str(uuid.uuid4()), input=None,)
            ]
            if not is_done
            else []
        )

    # get response string
    # return_direct can change the response type
    try:
        response_str = str(response.message.content)
    except AttributeError:
        response_str = str(response)

    agent_response = AgentChatResponse(response=response_str, sources=tool_outputs)

    return TaskStepOutput(
        output=agent_response,
        task_step=step,
        is_last=is_done,
        next_steps=new_steps,
    )
```

### 4.2 FunctionCallingLLM.chat_with_tools

```python
def chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    **kwargs: Any,
) -> ChatResponse:
    """Chat with function calling."""
    chat_kwargs = self._prepare_chat_with_tools(
        tools,
        user_msg=user_msg,
        chat_history=chat_history,
        verbose=verbose,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        **kwargs,
    )
    response = self.chat(**chat_kwargs)
    return self._validate_chat_with_tools_response(
        response,
        tools,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        **kwargs,
    )
```

#### (1) `_prepare_chat_with_tools`

```python
#class OpenAI(FunctionCallingLLM)
def _prepare_chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    tool_choice: Union[str, dict] = "auto",
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Predict and call the tool."""
    tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

    # if strict is passed in, use, else default to the class-level attribute, else default to True`
    if strict is not None:
        strict = strict
    else:
        strict = self.strict

    if self.metadata.is_function_calling_model:
        for tool_spec in tool_specs:
            if tool_spec["type"] == "function":
                tool_spec["function"]["strict"] = strict
                # in current openai 1.40.0 it is always false.
                tool_spec["function"]["parameters"]["additionalProperties"] = False

    if isinstance(user_msg, str):
        user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

    messages = chat_history or []
    if user_msg:
        messages.append(user_msg)

    return {
        "messages": messages,
        "tools": tool_specs or None,
        "tool_choice": resolve_tool_choice(tool_choice) if tool_specs else None,
        **kwargs,
    }
```

#### (2) `chat`

```python
#class OpenAI(FunctionCallingLLM)
@llm_chat_callback()
def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
    if self._use_chat_completions(kwargs):
        chat_fn = self._chat
    else:
        chat_fn = completion_to_chat_decorator(self._complete)
    return chat_fn(messages, **kwargs)

@llm_retry_decorator
def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
    client = self._get_client()
    message_dicts = to_openai_message_dicts(messages, model=self.model,)

    if self.reuse_client:
        response = client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **self._get_model_kwargs(**kwargs),
        )
    else:
        with client:
            response = client.chat.completions.create(
                messages=message_dicts,
                stream=False,
                **self._get_model_kwargs(**kwargs),
            )

    openai_message = response.choices[0].message
    message = from_openai_message(openai_message, modalities=self.modalities or ["text"])
    openai_token_logprobs = response.choices[0].logprobs
    logprobs = None
    if openai_token_logprobs and openai_token_logprobs.content:
        logprobs = from_openai_token_logprobs(openai_token_logprobs.content)

    return ChatResponse(
        message=message,
        raw=response,
        logprobs=logprobs,
        additional_kwargs=self._get_response_token_counts(response),
    )
```

## 5、planAgent

### 5.1 案例

```python
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
)

worker = FunctionCallingAgentWorker.from_tools(tools, llm=llm)
agent = StructuredPlannerAgent(worker)
```

### 5.2 BasePlanningAgentRunner.chat

```python
@dispatcher.span
def _chat(
    self,
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
    tool_choice: Union[str, dict] = "auto",
    mode: ChatResponseMode = ChatResponseMode.WAIT,
) -> AGENT_CHAT_RESPONSE_TYPE:
    if chat_history is not None:
        self.memory.set(chat_history)

    # 使用 llm 获取需要执行的 plan
    plan_id = self.create_plan(message)

    results = []
    dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
    while True:
        #根据 plan 获取 task 队列
        next_task_ids = self.get_next_tasks(plan_id)
        if len(next_task_ids) == 0:
            break

        #执行 task
        jobs = [
            self.arun_task(sub_task_id, mode=mode, tool_choice=tool_choice)
            for sub_task_id in next_task_ids
        ]
        results = asyncio_run(run_jobs(jobs, workers=len(jobs)))

        for sub_task_id in next_task_ids:
            self.mark_task_complete(plan_id, sub_task_id)

        # refine the plan
        self.refine_plan(message, plan_id)

    dispatcher.event(AgentChatWithStepEndEvent(response=results[-1] if len(results) > 0 else None))
    return results[-1]
```

### 5.3 StructuredPlannerAgent.create_plan

```python
def create_plan(self, input: str, **kwargs: Any) -> str:
    """Create plan. Returns the plan_id."""
    tools = self.get_tools(input)
    tools_str = ""
    for tool in tools:
        tools_str += ((tool.metadata.name or "") + ": " + tool.metadata.description + "\n")

    try:
        plan = self.llm.structured_predict(
            Plan,
            self.initial_plan_prompt,
            tools_str=tools_str,
            task=input,
        )
    except (ValueError, ValidationError):
        if self.verbose:
            print("No complex plan predicted. Defaulting to a single task plan.")
        plan = Plan(
            sub_tasks=[SubTask(name="default", input=input, expected_output="", dependencies=[])]
        )

    #plan 存储
    plan_id = str(uuid.uuid4())
    self.state.plan_dict[plan_id] = plan

    #创建 task 队列
    for sub_task in plan.sub_tasks:
        self.create_task(sub_task.input, task_id=sub_task.name)
    return plan_id

DEFAULT_INITIAL_PLAN_PROMPT = """\
Think step-by-step. Given a task and a set of tools, create a comprehensive, end-to-end plan to accomplish the task.
Keep in mind not every task needs to be decomposed into multiple sub-tasks if it is simple enough.
The plan should end with a sub-task that can achieve the overall task.

The tools available are:
{tools_str}

Overall Task: {task}
"""
```

## 6、自定义 Agent

### 6.1 案例

```python
## This is an example showing a trivial function that multiplies an input number by 2 each time.
## Pass this into an agent
def multiply_agent_fn(state: dict) -> Tuple[Dict[str, Any], bool]:
    """Mock agent input function."""
    if "max_count" not in state:
        raise ValueError("max_count must be specified.")

    # __output__ is a special key indicating the final output of the agent
    # __task__ is a special key representing the Task object passed by the agent to the function.
    # `task.input` is the input string passed
    if "__output__" not in state:
        state["__output__"] = int(state["__task__"].input)
        state["count"] = 0
    else:
        state["__output__"] = state["__output__"] * 2
        state["count"] += 1

    is_done = state["count"] >= state["max_count"]

    # the output of this function should be a tuple of the state variable and is_done
    return state, is_done

from llama_index.core.agent import FnAgentWorker

agent = FnAgentWorker(fn=multiply_agent_fn, initial_state={"max_count": 5}).as_agent()
agent.query("5")
```

## 7、代理功能引入

### 7.1 引入 retriver 能力

```python
# define an "object" index over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.agent.openai import OpenAIAgent

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=2), verbose=True
)
```

### 7.2 context retriver 能力

```python
from llama_index.core import Document
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent


# toy index - stores a list of Abbreviations
texts = [
    "Abbreviation: X = Revenue",
    "Abbreviation: YZ = Risk Factors",
    "Abbreviation: Z = Costs",
]
docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs)

# add context agent
context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    query_engine_tools,
    context_index.as_retriever(similarity_top_k=1),
    verbose=True,
)
response = context_agent.chat("What is the YZ of March 2022?")
```

### 7.3 query planning 能力

```python
# define query plan tool
from llama_index.core.tools import QueryPlanTool
from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    service_context=service_context
)
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=[query_tool_sept, query_tool_june, query_tool_march],
    response_synthesizer=response_synthesizer,
)

# initialize agent
agent = OpenAIAgent.from_tools(
    [query_plan_tool],
    max_function_calls=10,
    llm=OpenAI(temperature=0, model="gpt-4-0613"),
    verbose=True,
)

# should output a query plan to call march, june, and september tools
response = agent.query(
    "Analyze Uber revenue growth in March, June, and September"
)
```

## 8、定义 Tool

### 8.1 查询引擎 Tool

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# NOTE: lyft_index and uber_index are both SimpleVectorIndex instances
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        return_direct=False,
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        return_direct=False,
    ),
]

# initialize ReAct agent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
```

### 8.2 使用其他代理作为工具

```python
from llama_index.core.tools import QueryEngineTool

query_engine_tools = [
    QueryEngineTool(
        query_engine=sql_agent,
        metadata=ToolMetadata(
            name="sql_agent", description="Agent that can execute SQL queries."
        ),
    ),
    QueryEngineTool(
        query_engine=gmail_agent,
        metadata=ToolMetadata(
            name="gmail_agent",
            description="Tool that can send emails on Gmail.",
        ),
    ),
]

outer_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
```

# 十二、workflow











# 十三、评估

## 1、响应评估

### 1.1 简述

- **`Correctness`(正确性)**：生成的答案是否与给定查询的参考答案相匹配(需要标签)
- **`Semantic Similarity`(语义相似性)**：是否在语义上与参考答案相似(需要标签)
- **`Faithfulness`(忠诚)**：评估答案是否忠实于检索到的上下文(换句话说，是否有幻觉)
- **`Context Relevancy`(上下文相关性)**：检索到上下文是否与查询有关
- **`Answer Relevancy`(答案相关性)**：生成的答案是否与查询相关
- **`Guideline Adherence`(指南依从性)**：预测的答案是否遵守特定准则

### 1.2 BaseEvaluator 与 EvaluationResult

#### (1) BaseEvaluator

```python
class BaseEvaluator(PromptMixin):
    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string, retrieved contexts, and generated response string.
        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        return asyncio_run(
            self.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
                **kwargs,
            )
        )
```

#### (2) EvaluationResult

```python
class EvaluationResult(BaseModel):
    query: Optional[str] = Field(default=None, description="Query string")
    contexts: Optional[Sequence[str]] = Field(default=None, description="Context strings")
    response: Optional[str] = Field(default=None, description="Response string")
```

### 1.3 Correctness

> 生成的答案是否与给定查询的参考答案相匹配(需要标签)

```python
class CorrectnessEvaluator(BaseEvaluator):
    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference or "(NO REFERENCE ANSWER SUPPLIED)",
        )

        # Use the parser function
        score, reasoning = self.parser_function(eval_response) #参看 default_parser

        return EvaluationResult(
            query=query,
            response=response,
            passing=score >= self._score_threshold if score is not None else None,
            score=score,
            feedback=reasoning,
        )

#计算 score, reasoning
def default_parser(eval_response: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns: Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    if not eval_response.strip():
        return None, "No response"

    score_str, reasoning_str = eval_response.split("\n", 1)
    try:
        score = float(score_str)
    except ValueError:
        score = None

    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning
```

### 1.4 Semantic Similarity

> 是否在语义上与参考答案相似(需要标签)

```python
class SemanticSimilarityEvaluator(BaseEvaluator):
    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        response_embedding = await self._embed_model.aget_text_embedding(response)
        reference_embedding = await self._embed_model.aget_text_embedding(reference)

        similarity_score = self._similarity_fn(response_embedding, reference_embedding)
        passing = similarity_score >= self._similarity_threshold
        return EvaluationResult(
            score=similarity_score,
            passing=passing,
            feedback=f"Similarity score: {similarity_score}",
        )
```

### 1.5 Faithfulness

> 评估答案是否忠实于检索到的上下文(换句话说，是否有幻觉)

#### (1) 源码

```python
class FaithfulnessEvaluator(BaseEvaluator):
    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        #构建文档索引
        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs)
        query_engine = index.as_query_engine(
            llm=self._llm,
            text_qa_template=self._eval_template,
            refine_template=self._refine_template,
        )
        #查询 response 的相似文档
        response_obj = await query_engine.aquery(response)
        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=raw_response_txt,
        )
```

#### (2) 使用案例

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

# build index
...

# define evaluator
evaluator = FaithfulnessEvaluator(llm=llm)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)

#方式一：
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))

#方式二：单独评估每个源环境
response_str = response.response
for source_node in response.source_nodes:
    eval_result = evaluator.evaluate(
        response=response_str, contexts=[source_node.get_content()]
    )
    print(str(eval_result.passing))
```

### 1.6 Context Relevancy

> 检索到上下文是否与查询有关

#### (1) 源码

```python
class ContextRelevancyEvaluator(BaseEvaluator):
    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        #构建文档索引
        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs)
        query_engine = index.as_query_engine(
            llm=self._llm,
            text_qa_template=self._eval_template,
            refine_template=self._refine_template,
        )
        #查询 query 相似文档
        response_obj = await query_engine.aquery(query)
        raw_response_txt = str(response_obj)

        #计算文档相似得分
        score, reasoning = self.parser_function(raw_response_txt)

        invalid_result, invalid_reason = False, None
        if score is None and reasoning is None:
            if self._raise_error:
                raise ValueError("The response is invalid")
            invalid_result = True
            invalid_reason = "Unable to parse the output string."

        if score:
            score /= self.score_threshold

        return EvaluationResult(
            query=query,
            contexts=contexts,
            score=score,
            feedback=raw_response_txt,
            invalid_result=invalid_result,
            invalid_reason=invalid_reason,
        )
        
#计算文档相似得分        
def _default_parser_function(output_str: str) -> Tuple[Optional[float], Optional[str]]:
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)([\d.]+)"

    # Using regex to find all matches
    result = re.search(pattern, output_str)

    # Check if any match is found
    if result:
        # Assuming there's only one match in the text, extract feedback and response
        feedback, score = result.groups()
        score = float(score) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None
```

#### (2) 使用案例

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import RelevancyEvaluator

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

# build index
...

# define evaluator
evaluator = RelevancyEvaluator(llm=llm)

# query index
query_engine = vector_index.as_query_engine()
query = "What battles took place in New York City in the American Revolution?"
response = query_engine.query(query)

#方式一：
eval_result = evaluator.evaluate_response(query=query, response=response)
print(str(eval_result))

#方式二：在特定的源节点上评估
response_str = response.response
for source_node in response.source_nodes:
    eval_result = evaluator.evaluate(
        query=query,
        response=response_str,
        contexts=[source_node.get_content()],
    )
    print(str(eval_result.passing)
```

### 1.7 Answer Relevancy

> 生成的答案是否与查询相关

```python
class AnswerRelevancyEvaluator(BaseEvaluator):
        async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        #llm 进行评估
        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            response=response,
        )

        #计算文档得分
        score, reasoning = self.parser_function(eval_response)

        invalid_result, invalid_reason = False, None
        if score is None and reasoning is None:
            if self._raise_error:
                raise ValueError("The response is invalid")
            invalid_result = True
            invalid_reason = "Unable to parse the output string."

        if score:
            score /= self.score_threshold

        return EvaluationResult(
            query=query,
            response=response,
            score=score,
            feedback=eval_response,
            invalid_result=invalid_result,
            invalid_reason=invalid_reason,
        )

#计算文档得分
def _default_parser_function(output_str: str) -> Tuple[Optional[float], Optional[str]]:
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)(\d)"
    result = re.search(pattern, output_str)

    if result:
        # Assuming there's only one match in the text, extract feedback and response
        feedback, score = result.groups()
        score = float(score) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None
```

### 1.8 Guideline Adherence

> 预测的答案是否遵守特定准则

```python
class GuidelineEvaluator(BaseEvaluator):
        async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        #llm 评估
        eval_response = await self._llm.apredict(
            self._eval_template,
            query=query,
            response=response,
            guidelines=self._guidelines,
        )
        #结果转换
        eval_data = self._output_parser.parse(eval_response)
        eval_data = cast(EvaluationData, eval_data)

        return EvaluationResult(
            query=query,
            response=response,
            passing=eval_data.passing,
            score=1.0 if eval_data.passing else 0.0,
            feedback=eval_data.feedback,
        )
```

### 1.9 PairwiseComparisonEvaluator

```python
async def aevaluate(
    self,
    query: Optional[str] = None,
    response: Optional[str] = None,
    contexts: Optional[Sequence[str]] = None,
    second_response: Optional[str] = None,
    reference: Optional[str] = None,
    sleep_time_in_seconds: int = 0,
    **kwargs: Any,
) -> EvaluationResult:
    #计算评估结果(score + reason)
    eval_result = await self._get_eval_result(
        query, response, second_response, reference
    )
    if self._enforce_consensus and not eval_result.invalid_result:
        #计算第二个评估结果(score + reason)
        flipped_eval_result = await self._get_eval_result(
            query, second_response, response, reference
        )
        #对比
        if not flipped_eval_result.invalid_result:
            resolved_eval_result = await self._resolve_results(
                eval_result, flipped_eval_result
            )
        else:
            resolved_eval_result = EvaluationResult(
                query=eval_result.query,
                response=eval_result.response,
                feedback=flipped_eval_result.response,
                invalid_result=True,
                invalid_reason="Output cannot be parsed.",
            )
    else:
        resolved_eval_result = eval_result

    return resolved_eval_result
```

## 2、检索评估(Retrieval Evaluation)

### 2.1 简述

- **`Dataset generation`(数据集生成)**：给定一个非结构化的文本语料库，合成生成（问题，上下文）对
- **`Retrieval Evaluation`(检索评估)**：给定检索器和一组问题，使用排名指标评估结果的结果

### 2.2 案例

```python
from llama_index.core.evaluation import RetrieverEvaluator

# define retriever somewhere (e.g. from index)
# retriever = index.as_retriever(similarity_top_k=2)
retriever = ...

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)
```

### 2.3 基础类

#### (1) BaseRetrievalEvaluator

```python
class BaseRetrievalEvaluator(BaseModel):
    metrics: List[BaseRetrievalMetric] = Field(..., description="List of metrics to evaluate")

#函数一：创建 evaluator
@classmethod
def from_metric_names(
    cls, metric_names: List[str], **kwargs: Any
) -> "BaseRetrievalEvaluator":
    """Create evaluator from metric names."""
    metric_types = resolve_metrics(metric_names)
    return cls(metrics=[metric() for metric in metric_types], **kwargs)
    
def resolve_metrics(metrics: List[str]) -> List[Type[BaseRetrievalMetric]]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")
    return [METRIC_REGISTRY[metric] for metric in metrics]

METRIC_REGISTRY: Dict[str, Type[BaseRetrievalMetric]] = {
    "hit_rate": HitRate,
    "mrr": MRR,
    "precision": Precision,
    "recall": Recall,
    "ap": AveragePrecision,
    "ndcg": NDCG,
    "cohere_rerank_relevancy": CohereRerankRelevancyMetric,
}

#函数二：评估
async def aevaluate(
    self,
    query: str,
    expected_ids: List[str],
    expected_texts: Optional[List[str]] = None,
    mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
    **kwargs: Any,
) -> RetrievalEvalResult:
    """Run evaluation with query string, retrieved contexts, and generated response string."""
    #检索文档 -- 需要继承实现
    retrieved_ids, retrieved_texts = await self._aget_retrieved_ids_and_texts(
        query, mode
    )
    
    #计算检索文档与期望文档指标
    metric_dict = {}
    for metric in self.metrics:
        eval_result = metric.compute(
            query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
        )
        metric_dict[metric.metric_name] = eval_result

    return RetrievalEvalResult(
        query=query,
        expected_ids=expected_ids,
        expected_texts=expected_texts,
        retrieved_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        mode=mode,
        metric_dict=metric_dict,
    )
```

#### (2) BaseRetrievalMetric

```python
class BaseRetrievalMetric(BaseModel, ABC):
    @abstractmethod
    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric.

        Args:
            query (Optional[str]): Query string
            expected_ids (Optional[List[str]]): Expected ids
            retrieved_ids (Optional[List[str]]): Retrieved ids
            **kwargs: Additional keyword arguments
        """
        
class RetrievalMetricResult(BaseModel):
    """Metric result.
    Attributes:
        score (float): Score for the metric
        metadata (Dict[str, Any]): Metadata for the metric result
    """
    score: float = Field(..., description="Score for the metric")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata for the metric result"
    )
```

#### (3) RetrievalEvalResult

```python
class RetrievalEvalResult(BaseModel):
    """Retrieval eval result.
    Attributes:
        query (str): Query string
        expected_ids (List[str]): Expected ids
        retrieved_ids (List[str]): Retrieved ids
        metric_dict (Dict[str, BaseRetrievalMetric]): Metric dictionary for the evaluation
    """
    query: str = Field(..., description="Query string")
    expected_ids: List[str] = Field(..., description="Expected ids")
    expected_texts: Optional[List[str]] = Field(
        default=None,
        description="Expected texts associated with nodes provided in `expected_ids`",
    )
    retrieved_ids: List[str] = Field(..., description="Retrieved ids")
    retrieved_texts: List[str] = Field(..., description="Retrieved texts")
    mode: "RetrievalEvalMode" = Field(
        default=RetrievalEvalMode.TEXT, description="text or image"
    )
    metric_dict: Dict[str, RetrievalMetricResult] = Field(
        ..., description="Metric dictionary for the evaluation"
    )
```

### 2.4 RetrieverEvaluator

```python
async def _aget_retrieved_ids_and_texts(
    self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
) -> Tuple[List[str], List[str]]:
    #检索
    retrieved_nodes = await self.retriever.aretrieve(query)

    #结果后处理
    if self.node_postprocessors:
        for node_postprocessor in self.node_postprocessors:
            retrieved_nodes = node_postprocessor.postprocess_nodes(retrieved_nodes, query_str=query)

    return (
        [node.node.node_id for node in retrieved_nodes],
        [node.node.text for node in retrieved_nodes],
    )
```

### 2.5 MultiModalRetrieverEvaluator

```python
async def _aget_retrieved_ids_and_texts(
    self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
) -> Tuple[List[str], List[str]]:
    #检索
    retrieved_nodes = await self.retriever.aretrieve(query)
    image_nodes: List[ImageNode] = []
    text_nodes: List[TextNode] = []

    #结果后处理
    if self.node_postprocessors:
        for node_postprocessor in self.node_postprocessors:
            retrieved_nodes = node_postprocessor.postprocess_nodes(retrieved_nodes, query_str=query)

    for scored_node in retrieved_nodes:
        node = scored_node.node
        if isinstance(node, ImageNode):
            image_nodes.append(node)
        if node.text:
            text_nodes.append(node)

    if mode == "text":
        return (
            [node.node_id for node in text_nodes],
            [node.text for node in text_nodes],
        )
    elif mode == "image":
        return (
            [node.node_id for node in image_nodes],
            [node.text for node in image_nodes],
        )
    else:
        raise ValueError("Unsupported mode.")
```

## 3、Evaluation 扩展能力

### 3.1 Batch Evaluation

#### (1) 使用案例

```python
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=8,
)

eval_results = await runner.aevaluate_queries(
    vector_index.as_query_engine(), queries=questions
)
```

## 4、额外能力--自动生成 question

### 4.1 使用案例

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

# build documents
documents = SimpleDirectoryReader("./data").load_data()

# define generator, generate questions
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=llm,
    num_questions_per_chunk=10,  # set the number of questions per nodes
)

rag_dataset = dataset_generator.generate_questions_from_nodes()
questions = [e.query for e in rag_dataset.examples]
```

### 4.2 `_agenerate_dataset`

#### (1) DatasetGenerator

```python
async def _agenerate_dataset(
    self,
    nodes: List[BaseNode],
    num: int | None = None,
    generate_response: bool = False,
) -> QueryResponseDataset:
    query_tasks: List[Coroutine] = []
    queries: Dict[str, str] = {}
    responses_dict: Dict[str, str] = {}

    if self._show_progress:
        from tqdm.asyncio import tqdm_asyncio

        async_module = tqdm_asyncio
    else:
        async_module = asyncio

    summary_indices: List[SummaryIndex] = []
    for node in nodes:
        if num is not None and len(query_tasks) >= num:
            break
        index = SummaryIndex.from_documents(
            [
                Document(
                    text=node.get_content(metadata_mode=self._metadata_mode),
                    metadata=node.metadata,  # type: ignore
                )
            ],
            callback_manager=self.callback_manager,
        )

        query_engine = index.as_query_engine(
            llm=self.llm,
            text_qa_template=self.text_question_template,
            use_async=True,
        )
        task = query_engine.aquery(
            self.question_gen_query,
        )
        query_tasks.append(task)
        summary_indices.append(index)

    responses = await async_module.gather(*query_tasks)
    for idx, response in enumerate(responses):
        result = str(response).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        cleaned_questions = [
            question for question in cleaned_questions if len(question) > 0
        ]
        cur_queries = {
            str(uuid.uuid4()): question for question in cleaned_questions
        }
        queries.update(cur_queries)

        if generate_response:
            index = summary_indices[idx]
            qr_tasks = []
            cur_query_items = list(cur_queries.items())
            cur_query_keys = [query_id for query_id, _ in cur_query_items]
            for query_id, query in cur_query_items:
                qa_query_engine = index.as_query_engine(
                    llm=self.llm,
                    text_qa_template=self.text_qa_template,
                )
                qr_task = qa_query_engine.aquery(query)
                qr_tasks.append(qr_task)
            qr_responses = await async_module.gather(*qr_tasks)
            for query_id, qa_response in zip(cur_query_keys, qr_responses):
                responses_dict[query_id] = str(qa_response)
        else:
            pass

    query_ids = list(queries.keys())
    if num is not None:
        query_ids = query_ids[:num]
        # truncate queries, responses to the subset of query ids
        queries = {query_id: queries[query_id] for query_id in query_ids}
        if generate_response:
            responses_dict = {
                query_id: responses_dict[query_id] for query_id in query_ids
            }

    return QueryResponseDataset(queries=queries, responses=responses_dict)
```

#### (2) RagDatasetGenerator

```python
async def _agenerate_dataset(
    self,
    nodes: List[BaseNode],
    labelled: bool = False,
) -> LabelledRagDataset:
    """Node question generator."""
    query_tasks = []
    examples: List[LabelledRagDataExample] = []
    summary_indices: List[SummaryIndex] = []
    for node in nodes:
        index = SummaryIndex.from_documents(
            [
                Document(
                    text=node.get_content(metadata_mode=self._metadata_mode),
                    metadata=node.metadata,  # type: ignore
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    relationships=node.relationships,
                )
            ],
        )

        query_engine = index.as_query_engine(
            llm=self._llm,
            text_qa_template=self.text_question_template,
            use_async=True,
        )
        task = query_engine.aquery(
            self.question_gen_query,
        )
        query_tasks.append(task)
        summary_indices.append(index)

    responses = await run_jobs(query_tasks, self._show_progress, self._workers)
    for idx, response in enumerate(responses):
        result = str(response).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        cleaned_questions = [
            question for question in cleaned_questions if len(question) > 0
        ][: self.num_questions_per_chunk]

        num_questions_generated = len(cleaned_questions)
        if num_questions_generated < self.num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({self.num_questions_per_chunk})."
            )

        index = summary_indices[idx]
        reference_context = nodes[idx].get_content(metadata_mode=MetadataMode.NONE)
        model_name = self._llm.metadata.model_name
        created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
        if labelled:
            index = summary_indices[idx]
            qr_tasks = []
            for query in cleaned_questions:
                # build summary index off of node (i.e. context)
                qa_query_engine = index.as_query_engine(
                    llm=self._llm,
                    text_qa_template=self.text_qa_template,
                )
                qr_task = qa_query_engine.aquery(query)
                qr_tasks.append(qr_task)
            answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                qr_tasks, self._show_progress, self._workers
            )
            for question, answer_response in zip(
                cleaned_questions, answer_responses
            ):
                example = LabelledRagDataExample(
                    query=question,
                    reference_answer=str(answer_response),
                    reference_contexts=[reference_context],
                    reference_answer_by=created_by,
                    query_by=created_by,
                )
                examples.append(example)
        else:
            for query in cleaned_questions:
                example = LabelledRagDataExample(
                    query=query,
                    reference_answer="",
                    reference_contexts=[reference_context],
                    reference_answer_by=None,
                    query_by=created_by,
                )
                examples.append(example)

    # split train/test
    return LabelledRagDataset(examples=examples)
```