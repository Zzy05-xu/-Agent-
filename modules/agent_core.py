"""
Agent 核心模块
基于 LangChain create_react_agent 构建 ReAct 范式智能体�?集成 6 个专业求职工具，支持对话历史，全链路异常降级�?
架构设计�?1. 使用 langchain.agents.create_react_agent 构建标准 ReAct Agent
2. 自定义中文系统提示词，明确角色与行为规范
3. AgentExecutor 封装，配置最大迭代次数与解析错误处理
4. 三层异常捕获：API超时 / 调用失败 / Token超限 �?自动降级为直接LLM回答
"""
from typing import Any, Dict, Generator, List, Optional, Tuple
from typing import Any, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage

from config.logger import get_logger
from config.settings import get_llm
from modules.tools import ALL_TOOLS

# ── ReAct 系统提示�?─────────────────────────────────────────
# 设计思路�?# - 明确 Agent 角色定位�?专业求职辅导助手"
# - 要求先规划再执行，提升回答质�?# - 定义工具调用规范，避免无意义循环
# - 中文语境，适配国内实习求职场景

REACT_SYSTEM_PROMPT = """你是一位专业求职辅导助手，专注于帮助大学生准备实习求职�?
你的核心能力�?1. 检索匹配的实习岗位信息
2. 解析和评估简历质�?3. �?STAR 法则优化简历经�?4. 生成模拟面试�?5. 管理投递进度记�?
⚠️ 行为规范（非常重要）�?- 每次回答前，先分析用户需求，规划需要调用哪些工�?- 工具调用完成后，基于工具返回结果给出清晰、有条理的回�?- 如果工具返回错误或空结果，友好告知用户并给出替代建议
- 不要编造信息，答案必须基于工具返回的实际数�?- 回答使用 Markdown 格式，结构清晰，善用标题、列表和emoji

可用工具�?{tools}

工具名称列表：{tool_names}

请严格按以下 ReAct 格式回复。每轮必须包�?Thought + Action + Action Input 三部分，缺一不可�?
【正确格式示例�?Question: 帮我找数据分析的实习岗位
Thought: 用户需要数据分析方向的实习岗位信息，我应该使用 job_search 工具来检�?Action: job_search
Action Input: 数据分析实习 互联�?Observation: [工具返回的结果]
Thought: 我已经获取到了相关的岗位信息，可以整理回答给用户�?Final Answer: 为你找到了以下数据分析实习岗�?..

【错误示例（会导致解析失败）�?- 只写 Thought 不写 Action �?绝对禁止
- Thought 中包含换�?�?必须在同一行写�?- Action Input 为空 �?必须填写具体搜索关键�?
Question: 用户的问�?Thought: 思考当前需要做什么（一行写完，不要换行�?Action: 工具名称（必须是 [{tool_names}] 之一�?Action Input: 工具输入参数（必须填写具体内容，不能为空�?Observation: 工具返回的结�?... (可重复多�?
Thought: 我已经有足够的信息来回答�?Final Answer: 针对用户问题的最终回�?
注意：Thought必须在同一行写完，不能有换行符�?
开始！"""

# ── Prompt 模板 ──────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
    REACT_SYSTEM_PROMPT
    + "\n\n## 对话历史\n{chat_history}\n\n"
    + "## 当前问题\n{input}\n\n"
    + "{agent_scratchpad}"
)


# ── Agent 初始�?─────────────────────────────────────────────


# ── Agent 实例缓存（模块级，避免每次请求重建） ──
_agent_cache = {}

def get_or_create_agent(max_iterations: int = 5, temperature: float = 0.2) -> AgentExecutor:
    """获取或创建 Agent 实例（按参数缓存复用）。"""
    key = (max_iterations, temperature)
    if key not in _agent_cache:
        _agent_cache[key] = create_agent(max_iterations, temperature)
    return _agent_cache[key]

def clear_agent_cache() -> None:
    """清空 Agent 缓存（切换 API 配置时调用）。"""
    _agent_cache.clear()

def create_agent(max_iterations: int = 5, temperature: float = 0.2) -> AgentExecutor:
    """
    创建并返回配置好�?AgentExecutor 实例（带 ConversationBufferMemory）�?    
    配置要点�?    - max_iterations=5：最�?�?Thought-Action-Observation 循环
    - handle_parsing_errors=True：LLM 输出格式异常时自动重�?    - verbose=False：生产环境关闭详细日志，Streamlit 中自行展�?    - return_intermediate_steps=True：返回思考链和工具调用日�?    - memory: 将多轮对话历史注�?prompt �?chat_history 变量
    
    Args:
        max_iterations: 最大迭代次数，防止无限循环
        temperature: LLM 温度参数
    
    Returns:
        配置好的 AgentExecutor 实例
    """
    llm = get_llm(temperature=temperature, max_tokens=2048)

    # ReAct prompt 已包�?{input} �?{agent_scratchpad} 变量
    # 新增 {chat_history} 变量用于注入多轮对话上下文
    react_agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=REACT_PROMPT,
    )

    executor = AgentExecutor(
        agent=react_agent,
        tools=ALL_TOOLS,
        max_iterations=max_iterations,
        stop=["Observation:", "Final Answer:"],
        handle_parsing_errors=True,
        verbose=False,
        return_intermediate_steps=True,
        max_execution_time=120,
    )

    return executor


# ── Agent 调用入口 ───────────────────────────────────────────

def run_agent(
    query: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> dict:
    """
    Agent 调用入口，支持对话历史，全链路异常降级�?    
    多轮对话实现�?    - chat_history �?LangChain BaseMessage 列表
    - 会被注入�?ReAct prompt �?{chat_history} 变量
    - Agent 可以看到之前的对话上下文，实现连贯的多轮交互
    
    返回结构�?    {
        "output": "最终回答文�?,
        "intermediate_steps": [(action, observation), ...],  # 思考链+工具调用
        "has_error": False,   # 是否触发降级
        "error_msg": "",      # 降级时的错误信息
    }
    
    Args:
        query: 用户输入的查询文�?        chat_history: 可选的对话历史消息列表（LangChain 格式�?    
    Returns:
        包含回答、思考链、错误标志的字典
    """
    result = {
        "output": "",
        "intermediate_steps": [],
        "has_error": False,
        "error_msg": "",
    }

    try:
        executor = get_or_create_agent()

        invoke_args = {"input": query}
        if chat_history:
            invoke_args["chat_history"] = chat_history

        raw_result = executor.invoke(invoke_args)

        result["output"] = raw_result.get("output", "Agent did not return a valid response.")
        result["intermediate_steps"] = raw_result.get("intermediate_steps", [])

    except Exception as e:
        logger = get_logger(__name__)
        error_str = str(e)
        error_type = type(e).__name__
        logger.error(f"Agent 调用失败 ({error_type}): {error_str}", exc_info=True)

        # ── 三层异常分类降级 ──
        if "timeout" in error_str.lower() or "timed out" in error_str.lower():
            # �?层：API 超时
            result["has_error"] = True
            result["error_msg"] = f"API 调用超时: {error_str}"
            result["output"] = _fallback_direct_answer(
                query, "请求超时，已切换为基础模式回答。\n\n"
            )

        elif "token" in error_str.lower() or "context_length" in error_str.lower():
            # �?层：Token 超限
            result["has_error"] = True
            result["error_msg"] = f"Token 超限: {error_str}"
            result["output"] = _fallback_direct_answer(
                query, "上下文过长，已截断并切换为基础模式。\n\n"
            )

        else:
            # �?层：其他调用失败
            result["has_error"] = True
            result["error_msg"] = f"Agent 调用失败 ({error_type}): {error_str}"
            result["output"] = _fallback_direct_answer(
                query, f"工具调用遇到问题（{error_type}），已切换为基础对话模式。\n\n"
            )

    return result

def run_agent_streaming(
    query: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Agent 流式调用入口，使用 threading + queue 安全运行异步流（兼容 Windows）。
    yield 事件: tool_start / tool_end / token / done / error
    """
    logger = get_logger(__name__)
    full_output = ""

    try:
        executor = get_or_create_agent()
        invoke_args = {"input": query}
        if chat_history:
            invoke_args["chat_history"] = chat_history

        import threading
        import queue as _q
        result_queue = _q.Queue()

        def _async_runner():
            import asyncio
            async def _stream():
                try:
                    async for event in executor.astream_events(invoke_args, version="v2"):
                        kind = event.get("event", "")
                        if kind == "on_tool_start":
                            result_queue.put({"type": "tool_start", "tool": event.get("name", "unknown"), "input": str(event.get("data", {}).get("input", ""))[:500]})
                        elif kind == "on_tool_end":
                            result_queue.put({"type": "tool_end", "tool": event.get("name", "unknown"), "output": str(event.get("data", {}).get("output", ""))[:1000]})
                        elif kind == "on_chat_model_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                result_queue.put({"type": "token", "content": chunk.content})
                    result_queue.put({"type": "__done__"})
                except Exception as exc:
                    result_queue.put({"type": "__error__", "error": str(exc)})
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_stream())
                loop.close()
            except Exception as exc:
                result_queue.put({"type": "__error__", "error": str(exc)})

        t = threading.Thread(target=_async_runner, daemon=True)
        t.start()
        import time as _t
        deadline = _t.time() + 120
        while _t.time() < deadline:
            try:
                item = result_queue.get(timeout=0.1)
            except:
                if not t.is_alive():
                    break
                continue
            tpe = item.get("type", "")
            if tpe == "__done__":
                break
            if tpe == "__error__":
                logger.warning("stream error: " + item.get("error", ""))
                break
            if tpe == "token":
                full_output += item["content"]
            yield item
        t.join(timeout=5)

        if not full_output:
            logger.warning("no stream text, falling back to sync")
            result = run_agent(query, chat_history)
            full_output = result["output"]
            # Character-level streaming for Chinese text compatibility
            for i in range(0, len(full_output), 3):
                yield {"type": "token", "content": full_output[i:i+3]}
            yield {"type": "done", "output": full_output, "has_error": result.get("has_error", False)}
            return

        yield {"type": "done", "output": full_output, "has_error": False}

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        logger.error(f"streaming failed ({error_type}): {error_str}", exc_info=True)
        fallback = _fallback_direct_answer(query, f"工具调用遇到问题（{error_type}），已切换为基础对话模式。\n\n")
        yield {"type": "error", "error_msg": f"Agent 调用失败 ({error_type}): {error_str}", "output": fallback}





def _fallback_direct_answer(query: str, prefix: str = "") -> str:
    """
    降级兜底方案：直接用 LLM 回答 + 功能引导�?    
    �?Agent 工具链调用失败时（API超时/Token超限/网络错误），
    不中断服务，而是用纯 LLM 给出建议并引导用户使用侧边栏功能�?    
    Args:
        query: 用户原始查询
        prefix: 降级提示前缀
    
    Returns:
        降级后的回答文本
    """
    try:
        logger = get_logger(__name__)
        logger.warning(f"触发降级兜底回答，原始查�.") 
        llm = get_llm(temperature=0.3, max_tokens=1024)
        fallback_prompt = (
            f"用户的问题是：「{query}」\n\n"
            "你是一位实习求职助手。由于技术原因，部分自动化工具暂时不可用�."
            "请基于你的知识直接回答用户问题，并在末尾引导用户使用侧边栏的手动功能入口�."
            "回答简洁有用，控制�?00字以内�."
        )
        response = llm.invoke(fallback_prompt)
        direct_answer = response.content if hasattr(response, "content") else str(response)
    except Exception:
        direct_answer = (
            "抱歉，当前服务出现异常，请您稍后重试。\n\n"
            "💡 您可以尝试：\n"
            "- 在侧边栏检�?API 配置是否正确\n"
            "- 使用「简历优化」标签页上传简历\n"
            "- 使用「投递管理」标签页手动记录投递进�."
        )

    guide = (
        "\n\n💡 **快速功能入�?*（侧边栏/标签页均可使用）：\n"
        "- 📋 岗位检�?�?在「智能对话助手」中输入岗位关键词\n"
        "- 📄 简历解�?�?在「简历优化」页上传 PDF\n"
        "- 📊 匹配评分 �?在「简历优化」页输入 JD + 简历\n"
        "- �?简历优�?�?在「简历优化」页使用 STAR 改写\n"
        "- 🎤 面试�?�?在「模拟面试」页选择目标岗位\n"
        "- 📌 投递管�?�?在「投递管理」页管理进度"
    )

    return prefix + direct_answer + guide
