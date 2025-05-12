## 1、实现

```python
# -*-coding: utf-8 -*-
import os
os.system("ifconfig")


import math
import time
import uuid
import json
import logging

import torch
import argparse
from enum import Enum
from typing import Optional, List, AsyncGenerator
from pydantic import ValidationError

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
import openai.types.chat.chat_completion as chat_types
from openai.types.completion_usage import CompletionUsage
import openai.types.chat.chat_completion_message_tool_call as chat_message_tool_call_types
import openai.types.chat.chat_completion_chunk as chat_chunk_types
import openai.types.chat.completion_create_params as create_types
from openai.types.chat import ChatCompletionToolChoiceOptionParam
import xgrammar as xgr
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


def format_sse(data):
    return f"data: {json.dumps(data.model_dump(), ensure_ascii=False)}\n\n"


app = FastAPI()
engine = None

global tokenizer
global full_vocab_size
global grammar_compiler
global engine_args

TOOL_CALL_TOKEN = "<tool_call>"
TOOL_CALL_END_TOKEN = "</tool_call>"
THINK_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"


class State(Enum):
    CONTENT = 0
    THINKING = 1
    TOOL_CALL = 2
    SELECT = 3  # 并行工具调用，决定是否还有下一个工具
    END = 4



def build_json_schema(tools, tool_choice: ChatCompletionToolChoiceOptionParam):
    if isinstance(tool_choice, dict):
        fname = tool_choice["function"]["name"]
        for f in tools:
            if f["function"]["name"] == fname:
                param_schema = f["function"]["parameters"]
                break
        else:
            raise ValueError(f"Tool {fname} not found")

        defs = None
        if "$defs" in param_schema:
            defs = param_schema.pop("$defs")

        result = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": fname},
                "arguments": param_schema,
            },
            "required": ["name", "arguments"],
        }

        if defs is not None:
            result["$defs"] = defs

        return result
    elif tool_choice in ("auto", "required"):
        one_of_items = []
        for f in tools:
            param_schema = f["function"]["parameters"]

            defs = None
            if "$defs" in param_schema:
                defs = param_schema.pop("$defs")

            item = {
                "properties": {
                    "name": {"type": "string", "const": f["function"]["name"]},
                    "arguments": param_schema,
                },
                "required": ["name", "arguments"],
            }

            if defs is not None:
                item["$defs"] = defs
            one_of_items.append(item)

        return {
            "oneOf": one_of_items
        }
    elif tool_choice == "none":
        return None


class VLLMLogitsProcessor(object):
    def __init__(self, req: create_types.CompletionCreateParams,
                 tokenizer: PreTrainedTokenizer,
                 ):
        self.state = None
        self.mask: Optional[torch.Tensor] = None

        self.req = req

        if not req.get("tools"):
            self.tool_choice = "none"
        else:
            self.tool_choice = "auto"
        if tc := req.get("tool_choice"):
            self.tool_choice = tc

        # if req.get("reasoning_effort") and "Qwen3" in engine_args.model:
        #     self.enable_thinking = True
        # else:
        #     self.enable_thinking = False
        self.enable_thinking = True

        schema = build_json_schema(req.get("tools"), self.tool_choice)
        if schema is not None:
            json_grammar = xgr.Grammar.from_json_schema(schema)
            extra_grammar = xgr.Grammar.from_ebnf('root ::= [\\n]')
            added_grammar = grammar_compiler.compile_grammar(xgr.Grammar.concat(extra_grammar, json_grammar, extra_grammar))
            self.grammar_matcher = xgr.GrammarMatcher(added_grammar, terminate_without_stop_token=True)
        else:
            self.grammar_matcher = None

        self.token_bitmask = xgr.allocate_token_bitmask(1, full_vocab_size)
        self.first_time_param = True
        self.id_cache = []

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        # print("=" * 120)
        # print(f"gen_ids: {input_ids}")
        # print(f"gen_text: {tokenizer.decode(input_ids)}")

        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            self.mask = torch.full_like(scores, -math.inf)

        if self.state is None:
            if self.enable_thinking:
                self.state = State.THINKING
                self.id_cache.append(tokenizer.convert_tokens_to_ids(THINK_TOKEN))
                # print("-> THINKING")
            else:
                self.state = State.CONTENT
                # print("-> CONTENT")
        elif self.state == State.THINKING:
            if input_ids[-1] == tokenizer.convert_tokens_to_ids(THINK_END_TOKEN):
                self.state = State.CONTENT
                self.id_cache.append(tokenizer.encode("\n\n")[0])
                # print("-> CONTENT")
        elif self.state == State.CONTENT:
            if input_ids[-1] == tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN):
                self.state = State.TOOL_CALL
                # print("-> TOOL_CALL")
                self.grammar_matcher.reset()
                self.first_time_param = True
        elif self.state == State.TOOL_CALL:
            if input_ids[-1] == tokenizer.convert_tokens_to_ids(TOOL_CALL_END_TOKEN):
                self.state = State.SELECT
                # print("-> SELECT")
        elif self.state == State.SELECT:
            if input_ids[-1] == tokenizer.eos_token_id:
                self.state = State.END
                # print("-> END")
            elif input_ids[-1] == tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN):
                self.state = State.TOOL_CALL
                self.grammar_matcher.reset()
                self.first_time_param = True
                # print("-> TOOL_CALL")
        else:  # self.state == State.END
            pass

        if self.id_cache:
            self.mask[self.id_cache.pop(0)] = 0
            return scores + self.mask

        # 状态具体逻辑
        if self.state == State.THINKING:
            self.mask.fill_(0.)
            self.mask[tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN)] = -math.inf
        elif self.state == State.CONTENT:
            if self.tool_choice == "none":
                self.mask.fill_(0.)
                self.mask[tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN)] = -math.inf
            elif self.tool_choice == "required" or isinstance(self.tool_choice, dict):
                self.mask[tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN)] = 0
            else:
                self.mask.fill_(0.)
        elif self.state == State.TOOL_CALL:
            if self.first_time_param:
                self.first_time_param = False
            else:
                if not self.grammar_matcher.is_terminated():
                    assert self.grammar_matcher.accept_token(input_ids[-1])
            if not self.grammar_matcher.is_terminated():
                xgr.reset_token_bitmask(self.token_bitmask)
                self.grammar_matcher.fill_next_token_bitmask(self.token_bitmask)
                scores_with_batch = scores.unsqueeze(0)
                xgr.apply_token_bitmask_inplace(scores_with_batch, self.token_bitmask.to(scores.device))
                return scores_with_batch.squeeze(0)
            else:
                self.mask[tokenizer.convert_tokens_to_ids(TOOL_CALL_END_TOKEN)] = 0
        elif self.state == State.SELECT:
            if input_ids[-1] == tokenizer.encode("\n")[0]:  # 第二次
                self.mask[tokenizer.convert_tokens_to_ids(TOOL_CALL_TOKEN)] = 0
            else: #
                if self.req.get("parallel_tool_calls", True):
                    # tokenizer.convert_tokens_to_ids("\n") 会是 None，和encode的结果不一样，不知道为啥
                    self.mask[[tokenizer.eos_token_id, tokenizer.encode("\n")[0]]] = 0
                else:
                    self.mask[tokenizer.eos_token_id] = 0
        elif self.state == State.END:
            self.mask[tokenizer.eos_token_id] = 0
        return scores + self.mask


async def stream_results(results_generator, stream: bool) -> AsyncGenerator[bytes, None]:
    gen_text = ""
    gen_token_ids = []
    rid = str(uuid.uuid4())
    model = "vllm"
    result = {"content": "", "tool_calls": []}
    tool_call_cache = []
    state = State.CONTENT

    state_change = {
        State.CONTENT: {
            TOOL_CALL_TOKEN: State.TOOL_CALL,
        },
        State.TOOL_CALL: {
            TOOL_CALL_END_TOKEN: State.SELECT,
        },
        State.SELECT: {
            TOOL_CALL_TOKEN: State.TOOL_CALL,
            tokenizer.eos_token: State.END,
        }
    }

    async for request_output in results_generator:
        chunk = request_output.outputs[0].text[len(gen_text):]
        token_id = request_output.outputs[0].token_ids[len(gen_token_ids):][0]
        gen_text = request_output.outputs[0].text
        gen_token_ids = request_output.outputs[0].token_ids

        if tokenizer.convert_ids_to_tokens(token_id) == TOOL_CALL_END_TOKEN:
            tool_call = tokenizer.decode(tool_call_cache)
            tool_call = json.loads(tool_call)
            tool_call["arguments"] = json.dumps(tool_call["arguments"], ensure_ascii=False)
            result["tool_calls"].append(tool_call)
            tool_call_cache = []

        if next_state := state_change[state].get(tokenizer.convert_ids_to_tokens(token_id)):
            state = next_state
            continue

        if state == State.CONTENT:
            result["content"] += chunk
            if stream and chunk:
                yield format_sse(chat_chunk_types.ChatCompletionChunk(
                    id=rid,
                    choices=[chat_chunk_types.Choice(
                        index=0,
                        delta=chat_chunk_types.ChoiceDelta(content=chunk),
                    )],
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk"
                ))
        elif state == State.TOOL_CALL:
            tool_call_cache.append(token_id)

    if state == State.END:
        if stream:
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    finish_reason="tool_calls",
                    delta=chat_chunk_types.ChoiceDelta(
                        tool_calls=[chat_chunk_types.ChoiceDeltaToolCall(
                            index=i,
                            id=random_uuid(),
                            type="function",
                            function=chat_chunk_types.ChoiceDeltaToolCallFunction(
                                name=r["name"], arguments=r["arguments"]
                            ),
                        ) for i, r in enumerate(result["tool_calls"])]
                    )
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))
        else:
            yield chat_types.ChatCompletion(
                id=rid,
                choices=[chat_types.Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=chat_types.ChatCompletionMessage(
                        content=result["content"] if result["content"] else None,
                        tool_calls=[chat_message_tool_call_types.ChatCompletionMessageToolCall(
                            id=random_uuid(),
                            type="function",
                            function=chat_message_tool_call_types.Function(
                                name=r["name"], arguments=r["arguments"]
                            )
                        ) for r in result["tool_calls"]],
                        role="assistant"
                    )
                )],
                created=int(time.time()),
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),  # 为了bfcl不报错
                model=model,
                object="chat.completion"
            )
    else: # state == State.CONTENT
        if stream:
            yield format_sse(chat_chunk_types.ChatCompletionChunk(
                id=rid,
                choices=[chat_chunk_types.Choice(
                    index=0,
                    finish_reason="stop",
                    delta=chat_chunk_types.ChoiceDelta(),
                )],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk"
            ))
        else:
            yield chat_types.ChatCompletion(
                id=rid,
                choices=[chat_types.Choice(
                    finish_reason="stop",
                    index=0,
                    message=chat_types.ChatCompletionMessage(
                        content=result["content"],
                        role="assistant"
                    )
                )],
                created=int(time.time()),
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model=model,
                object="chat.completion"
            )


@app.post("/chat/completions")
async def chat_completion(request: Request) -> Response:
    data = await request.json()
    data.setdefault("stream", False)

    try:
        req: create_types.CompletionCreateParams = data
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # qwen3 模板问题
    for m in req["messages"]:
        if m["role"] == "assistant" and m["content"] is None:
            m["content"] = ""

    # if req.get("reasoning_effort") and "Qwen3" in engine_args.model:
    #     token_ids = tokenizer.apply_chat_template(req["messages"], tools=req.get("tools"), tokenize=True, add_generation_prompt=True, enable_thinking=True)
    # else:
    #     token_ids = tokenizer.apply_chat_template(req["messages"], tools=req.get("tools"), tokenize=True, add_generation_prompt=True, enable_thinking=False)
    token_ids = tokenizer.apply_chat_template(req["messages"], tools=req.get("tools"), tokenize=True, add_generation_prompt=True, enable_thinking=True)

    sampling_params = SamplingParams(
        max_tokens=req.get("max_tokens", 1024),
        temperature=req.get("temperature", 0.0),
    )
    processor = VLLMLogitsProcessor(req, tokenizer)
    sampling_params.logits_processors = [processor]
    request_id = random_uuid()

    try:
        results_generator = engine.generate(
            prompt={"prompt_token_ids": token_ids},
            sampling_params=sampling_params,
            request_id=request_id
        )
        generator = stream_results(results_generator, req["stream"])
        if req["stream"]:
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            final_output = None
            async for request_output in generator:
                if await request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await engine.abort(request_id)
                    return Response(status_code=499)
                final_output = request_output

            assert final_output is not None
            return JSONResponse(final_output.model_dump())
    except Exception as e:
        logging.exception(e)
        return JSONResponse({"error": "unknown error"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
    config = AutoConfig.from_pretrained(engine_args.model)
    full_vocab_size = config.vocab_size

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=5
    )
```

## 2、部署启动

```shell
VLLM_USE_V1=0 uv run qwen_enforcer.py --model /data/.modelcache/common-crawl-data/model-repo/Qwen/Qwen2.5-7B-Instruct/acbd96531cda22292a3ceaa67e984955d3965282 --tensor-parallel-size 1 --max-model-len 32768
```

## 3、案例

```python
import openai

client = openai.OpenAI(
    base_url="http://xxxx",
    api_key="11",
)



from pydantic import BaseModel, Field
import json

class ReturnJsonParams(BaseModel):
    age: int = Field(..., description="age")
    name: str = Field(..., description="name")
    hobbies: list[str] = Field(..., description="hobbies")
    

class GetWeatherParams(BaseModel):
    city: str = Field(..., description="city")
```

### 3.1 案例一

```python
messages = [{"role": "user", "content": """获取北京和上海的天气"""}]

#  正常 tool call
completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.01,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "返回提取的信息",
                "parameters": ReturnJsonParams.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": GetWeatherParams.model_json_schema()
            }
        },
    ],
    reasoning_effort=None,
    tool_choice="auto",
    max_tokens=512,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": null,
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": [
        {
            "id": "1cc2a2c8d32c4ba1b13380c83c15442e",
            "function": {
                "arguments": "{\"city\": \"北京\"}",
                "name": "get_weather"
            },
            "type": "function"
        },
        {
            "id": "1ee54f61cd7f46c7bcce7d616450ed4a",
            "function": {
                "arguments": "{\"city\": \"上海\"}",
                "name": "get_weather"
            },
            "type": "function"
        }
    ],
    "annotations": null
}
```

### 3.2 案例二

```python
messages = [{"role": "user", "content": """获取北京和上海的天气"""}]

#  禁止并行
completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.01,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "返回提取的信息",
                "parameters": ReturnJsonParams.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": GetWeatherParams.model_json_schema()
            }
        },
    ],
    reasoning_effort=None,
    tool_choice="auto",
    max_tokens=512,
    parallel_tool_calls=False,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": null,
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": [
        {
            "id": "3f300dda0d914844af171e2d54a378a0",
            "function": {
                "arguments": "{\"city\": \"北京\"}",
                "name": "get_weather"
            },
            "type": "function"
        }
    ],
    "annotations": null
}
```

### 3.3 案例三

```python
messages = [{"role": "user", "content": """获取北京和上海的天气"""}]

#  with think
completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.01,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "返回提取的信息",
                "parameters": ReturnJsonParams.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": GetWeatherParams.model_json_schema()
            }
        },
    ],
    reasoning_effort="high",
    tool_choice="auto",
    max_tokens=512,
    # parallel_tool_calls=False,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": "<think>\n好的，用户需要获取北京和上海的天气信息。首先，我需要检查可用的工具。提供的工具中有get_weather函数，可以获取指定城市的天气。由于用户同时询问了两个城市，我需要分别调用这个函数两次，每次传入不同的城市参数。首先调用北京，然后调用上海。确保每次调用的参数正确，城市名称准确无误。这样就能得到两个城市的天气数据，然后返回给用户。\n</think>\n\n",
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": [
        {
            "id": "46fc8074002a41bb8fed1d6607821753",
            "function": {
                "arguments": "{\"city\": \"北京\"}",
                "name": "get_weather"
            },
            "type": "function"
        },
        {
            "id": "828107a3493f45b292b9bd49d50ec3e2",
            "function": {
                "arguments": "{\"city\": \"上海\"}",
                "name": "get_weather"
            },
            "type": "function"
        }
    ],
    "annotations": null
}
```

### 3.4 案例四

```python
messages = [{"role": "user", "content": """获取北京和上海的天气"""}]

#  禁止函数调用
completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.5,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "返回提取的信息",
                "parameters": ReturnJsonParams.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": GetWeatherParams.model_json_schema()
            }
        },
    ],
    reasoning_effort=None,
    tool_choice="none",
    max_tokens=512,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": "北京和上海的天气信息如下：\n\n**北京的天气：**\n- 天气状况：晴\n- 温度：25°C\n\n**上海的天气：**\n- 天气状况：多云\n- 温度：28°C",
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": null,
    "annotations": null
}
```

### 3.5 案例五

```python
messages = [{"role": "user", "content": """获取北京和上海的天气"""}]

#  强制调用函数
completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.01,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "返回提取的信息",
                "parameters": ReturnJsonParams.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": GetWeatherParams.model_json_schema()
            }
        },
    ],
    reasoning_effort=None,
    tool_choice={"type": "function", "function": {"name": "return_json"}},
    max_tokens=512,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": null,
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": [
        {
            "id": "8ddaf503dd7942bcb97975cf07ce99cb",
            "function": {
                "arguments": "{\"age\": 25, \"name\": \"Alice\", \"hobbies\": [\"reading\", \"hiking\"]}",
                "name": "return_json"
            },
            "type": "function"
        }
    ],
    "annotations": null
}
```

### 3.6 案例六

```python
from typing import Annotated
from pydantic import StringConstraints


messages = [{"role": "user", "content": """列举Boss直聘的优点"""}]


class ReturnAdvantages(BaseModel):
    advantages: list[Annotated[str, StringConstraints(max_length=6)]] = Field(
        description="优点",
        min_items=3,
        max_items=3,
    )
    
# class ReturnAdvantages(BaseModel):
#     advantages: list[str] = Field(
#         description="优点",
#     )
    

completion = client.chat.completions.create(
    model="xx",
    messages=messages,
    temperature=0.01,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "return_advantages",
                "description": "返回优点",
                "parameters": ReturnAdvantages.model_json_schema()
            }
        },
    ],
    reasoning_effort=None,
    tool_choice={"type": "function", "function": {"name": "return_advantages"}},
    max_tokens=512,
)

print(json.dumps(completion.choices[0].message.model_dump(), indent=4, ensure_ascii=False))
```

输出：

```json
{
    "content": null,
    "refusal": null,
    "role": "assistant",
    "audio": null,
    "function_call": null,
    "tool_calls": [
        {
            "id": "1560b6f7e2b546839ddc454c2c97eb3e",
            "function": {
                "arguments": "{\"advantages\": [\"高效沟通\", \"职位丰富\", \"操作便捷\"]}",
                "name": "return_advantages"
            },
            "type": "function"
        }
    ],
    "annotations": null
}
```

