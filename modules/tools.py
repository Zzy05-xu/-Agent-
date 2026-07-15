"""
Agent 工具函数模块
基于 LangChain Tool 类规范封装 6 个专业求职辅助工具。
所有工具输入为字符串，输出为格式化字符串，完全适配 ReAct Agent 调用。

工具清单：
1. job_search_tool      - 岗位检索（RAG检索JD知识库）
2. resume_parse_tool    - 简历解析（PDF文本提取+结构化）
3. resume_match_tool    - 简历匹配评分（LLM评估）
4. resume_optimize_tool - 简历优化（STAR法则改写）
5. interview_question_tool - 面试题生成（RAG+LLM）
6. application_tracker_tool - 投递进度管理（CSV增删改查）
"""
import json
import os
import re
import time, threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain.tools import Tool
from pypdf import PdfReader

from config.logger import get_logger
from config.settings import get_llm, APPLICATIONS_CSV
from modules.rag_knowledge import search_knowledge


# ================================================================
# TTL cache (reduce duplicate API calls)
# ================================================================

_ttl_cache = {}
_CACHE_TTL = 300

def _cache_key(prefix: str, text: str) -> str:
    return f"{prefix}:{text[:200]}"

def _cache_get(key: str):
    if key in _ttl_cache:
        val, ts = _ttl_cache[key]
        if __import__("time").time() - ts < _CACHE_TTL:
            return val
        del _ttl_cache[key]
    return None

def _cache_set(key: str, value):
    _ttl_cache[key] = (value, __import__("time").time())
    if len(_ttl_cache) > 200:
        old = sorted(_ttl_cache, key=lambda k: _ttl_cache[k][1])[:100]
        for k in old:
            _ttl_cache.pop(k, None)

_csv_lock = __import__("threading").Lock()
# ═══════════════════════════════════════════════════════════════
# 工具 1：岗位检索
# ═══════════════════════════════════════════════════════════════

def _job_search(query: str) -> str:
    """
    从 JD 知识库中检索匹配的实习岗位。
    输入: 关键词或岗位要求描述
    输出: 去重后的 Top 匹配 JD 信息
    """
    ck = _cache_key("js", query)
    ch = _cache_get(ck)
    if ch is not None:
        return ch

    from modules.rag_knowledge import load_vector_store
    from config.settings import VECTOR_STORE_DIR

    jd_store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
    
    try:
        vector_store = load_vector_store(jd_store_path)
    except (FileNotFoundError, RuntimeError) as e:
        return (
            f"⚠️ 知识库未就绪：{e}\n\n"
            "📌 请先在侧边栏「知识库管理」中点击「构建知识库」，"
            "将 JD 和面经文档索引到向量库中。"
        )

    results = search_knowledge(vector_store, query, top_k=3)

    if not results or "检索异常" in results[0].get("content", ""):
        return "❌ 未找到匹配的岗位信息，请尝试调整关键词后重试。"

    seen_sources = set()
    seen_contents = set()
    unique_results = []
    for item in results:
        src = item["source"]
        src_file = src.split("/")[-1] if "/" in src else src
        ck2 = item["content"][:80].strip()
        if src_file not in seen_sources and ck2 not in seen_contents:
            seen_sources.add(src_file)
            seen_contents.add(ck2)
            unique_results.append(item)
    results = unique_results

    if not results:
        return "❌ 未找到匹配的岗位信息，请尝试调整关键词后重试。"

    output_parts = [f"📋 **岗位检索结果（Top {len(results)}）**\n"]
    for i, item in enumerate(results, 1):
        output_parts.append(
            f"---\n"
            f"**#{i}** | 📄 来源: {item['source']} | 相似度距离: {item['score']}\n\n"
            f"{item['content']}\n"
        )

    result = "\n".join(output_parts)
    _cache_set(ck, result)
    return result


job_search_tool = Tool(
    name="job_search",
    func=_job_search,
    description=(
        "从实习岗位知识库中检索匹配的岗位信息。"
        "当你需要帮用户搜索特定方向（如数据分析、后端开发）的实习岗位时，调用此工具。"
        "输入应为关键词或岗位方向描述，如'数据分析实习 互联网'。"
        "输出将包含公司名、岗位职责、任职要求、薪资范围等关键信息。"
    ),
)

# ═══════════════════════════════════════════════════════════════
# 工具 2：简历解析
# ═══════════════════════════════════════════════════════════════

# ── LLM 简历结构化提取 Prompt ──
RESUME_PARSE_PROMPT = """你是一位专业的简历解析专家。请从以下简历原文中提取关键信息，并以严格 JSON 格式返回。

{resume_text}

请返回如下 JSON 结构（只输出 JSON，不要其他文字）：
{{
    "name": "姓名（如未找到填入空字符串）",
    "education": [
        {{"school": "学校名", "degree": "学历（本科/硕士/博士）", "major": "专业", "year": "毕业年份"}}
    ],
    "skills": ["技能1", "技能2", "技能3"],
    "experience": [
        {{"company": "公司/组织", "role": "职位", "duration": "时间段", "description": "工作描述摘要"}}
    ],
    "projects": [
        {{"name": "项目名称", "description": "项目描述摘要", "tech_stack": ["使用技术1"]}}
    ]
}}

如果某个字段无法识别，返回空数组 [] 或空字符串。
"""  # noqa: E501


def _resume_parse(file_path: str) -> str:
    """
    解析 PDF 简历文件，使用 LLM 做结构化信息提取。
    
    流程：
    1. pypdf 逐页提取原始文本
    2. LLM 分析文本，提取教育背景、技能栈、项目经历
    3. 返回结构化 Markdown 展示
    
    输入: PDF 文件的本地绝对路径
    输出: 结构化简历文本（Markdown 格式）
    """
    path = Path(file_path.strip().strip('"'))
    
    if not path.exists():
        return f"❌ 文件不存在: {file_path}\n请确认路径正确，或先在「简历优化」页面上传文件。"

    if path.suffix.lower() != ".pdf":
        return f"❌ 仅支持 PDF 格式，当前文件类型: {path.suffix}"

    # 第1步：PDF 原始文本提取
    try:
        reader = PdfReader(str(path))
        full_text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text_parts.append(text)
        full_text = "\n".join(full_text_parts)
    except Exception as e:
        return f"❌ PDF 解析失败: {e}"

    if not full_text.strip():
        return "❌ 该 PDF 文件可能为扫描件或图片格式，未提取到可读文本。"

    # 第2步：LLM 结构化提取（最多分析前 4000 字）
    text_to_analyze = full_text[:4000]
    prompt = RESUME_PARSE_PROMPT.format(resume_text=text_to_analyze)

    try:
        llm = get_llm(temperature=0.1, max_tokens=1024)
        response = llm.invoke(prompt)
        llm_output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        # LLM 解析失败时降级为原始文本展示
        return (
            f"⚠️ LLM 结构化解析失败: {e}\n\n"
            f"📝 **简历原始文本预览**（前 3000 字符）:\n\n"
            f"{full_text[:3000]}"
            + ("\n...(内容过长已截断)" if len(full_text) > 3000 else "")
        )

    # JSON 提取
    parsed = _extract_json(llm_output)

    # 第3步：结构化展示
    if "parse_error" in parsed:
        # JSON 解析失败，降级展示
        return (
            f"⚠️ LLM 返回格式异常，以下为原始分析结果：\n\n{llm_output}\n\n"
            f"---\n📝 **简历原文（前 3000 字符）**:\n\n{full_text[:3000]}"
        )

    result_parts = ["📄 **简历智能解析结果**\n"]

    # 姓名
    name = parsed.get("name", "")
    if name:
        result_parts.append(f"👤 **姓名**: {name}\n")

    # 教育背景
    edu_list = parsed.get("education", [])
    if edu_list:
        result_parts.append("---\n🎓 **教育背景**:")
        for edu in edu_list:
            parts = []
            if edu.get("school"): parts.append(edu["school"])
            if edu.get("degree"): parts.append(edu["degree"])
            if edu.get("major"): parts.append(edu["major"])
            if edu.get("year"): parts.append(f"({edu['year']})")
            result_parts.append(f"  - {' · '.join(parts)}")
    else:
        result_parts.append("---\n🎓 **教育背景**: 未识别")

    # 技能栈
    skills = parsed.get("skills", [])
    if skills:
        result_parts.append("\n---\n🛠 **技能栈**:")
        for s in skills:
            result_parts.append(f"  - {s}")
    else:
        result_parts.append("\n---\n🛠 **技能栈**: 未识别")

    # 工作/实习经历
    exp_list = parsed.get("experience", [])
    if exp_list:
        result_parts.append("\n---\n💼 **工作/实习经历**:")
        for exp in exp_list:
            company = exp.get("company", "")
            role = exp.get("role", "")
            duration = exp.get("duration", "")
            desc = exp.get("description", "")
            header = f"{company} — {role}" if company else role
            if duration: header += f" ({duration})"
            result_parts.append(f"  - **{header}**")
            if desc: result_parts.append(f"    {desc}")
    else:
        result_parts.append("\n---\n💼 **工作/实习经历**: 未识别")

    # 项目经历
    proj_list = parsed.get("projects", [])
    if proj_list:
        result_parts.append("\n---\n📁 **项目经历**:")
        for proj in proj_list:
            name_p = proj.get("name", "")
            desc_p = proj.get("description", "")
            tech = ", ".join(proj.get("tech_stack", []))
            result_parts.append(f"  - **{name_p}**")
            if tech: result_parts.append(f"    技术栈: {tech}")
            if desc_p: result_parts.append(f"    {desc_p}")
    else:
        result_parts.append("\n---\n📁 **项目经历**: 未识别")

    # 完整文本预览（供人工核对）
    truncated = full_text[:2000]
    result_parts.append(
        f"\n---\n📝 **完整文本预览**（前{min(len(full_text), 2000)}字符）:\n{truncated}"
        + ("\n...(内容过长已截断)" if len(full_text) > 2000 else "")
    )

    return "\n".join(result_parts)


resume_parse_tool = Tool(
    name="resume_parse",
    func=_resume_parse,
    description=(
        "解析PDF简历文件，提取并结构化展示教育背景、技能栈、项目经历。"
        "当你需要分析用户的简历内容时调用此工具。"
        "输入应为PDF文件的完整本地路径。"
    ),
)

# ═══════════════════════════════════════════════════════════════
# 工具 3：简历匹配评分
# ═══════════════════════════════════════════════════════════════

MATCH_SCORE_PROMPT = """你是一位资深HR和招聘专家。请分析以下简历与目标岗位JD的匹配度。

【目标岗位 JD】
{jd_text}

【求职者简历】
{resume_text}

请返回严格格式的 JSON（不要包含其他文字）：
{{
    "score": 85,
    "core_match": "匹配点1；匹配点2；匹配点3",
    "missing_skills": "缺失技能1；缺失技能2；缺失技能3",
    "improvement": "具体改进建议1；建议2；建议3"
}}

评分标准: 90-100 高度匹配，80-89 较匹配，70-79 部分匹配，<70 需显著提升。
请客观评估，给出具体分数和详细分析。"""


def _extract_json(text: str) -> dict:
    """
    JSON 提取器兜底方案。
    先用标准 json.loads 解析；失败后用大括号计数法查找完整 JSON；
    再失败则返回原始文本标记。解决 LLM 在 JSON 前后添加额外文字的问题。
    """
    # 尝试1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试2: 大括号计数提取
    try:
        brace_count = 0
        start_idx = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    json_str = text[start_idx : i + 1]
                    return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    # 尝试3: 兜底
    return {"raw_output": text, "score": "N/A", "parse_error": True}


def _resume_match(jd_text: str) -> str:
    """
    简历匹配评分工具。
    输入: "JD:::\n{JD文本}\n\n简历:::\n{简历文本}"
    输出: 匹配度得分、核心匹配点、缺失技能、改进方向
    """
    jd_content = ""
    resume_content = ""

    if "简历:::" in jd_text:
        parts = jd_text.split("简历:::", 1)
        jd_content = parts[0].replace("JD:::", "").strip()
        resume_content = parts[1].strip() if len(parts) > 1 else ""
    else:
        half = len(jd_text) // 2
        jd_content = jd_text[:half]
        resume_content = jd_text[half:]

    if not jd_content or not resume_content:
        return (
            "❌ 请提供完整的 JD 和简历内容。\n"
            "输入格式示例：\n"
            "JD:::\n{岗位JD}\n\n简历:::\n{简历全文}"
        )

    # 智能截断：尽量保留完整的语义片段
    jd_text = _smart_truncate(jd_content, max_chars=3000)
    resume_text = _smart_truncate(resume_content, max_chars=3000)

    prompt = MATCH_SCORE_PROMPT.format(
        jd_text=jd_text,
        resume_text=resume_text,
    )

    try:
        llm = get_llm(temperature=0.1, max_tokens=1024)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"❌ LLM 调用失败: {e}\n请检查 API 配置后重试。"

    result = _extract_json(content)

    if "parse_error" in result:
        return f"⚠️ 评分结果格式异常，以下是 LLM 原始输出：\n\n{content}"

    return (
        f"📊 **简历匹配度评估**\n\n"
        f"🎯 **匹配得分**: {result.get('score', 'N/A')} / 100\n\n"
        f"✅ **核心匹配点**:\n{result.get('core_match', '未识别')}\n\n"
        f"⚠️ **缺失技能**:\n{result.get('missing_skills', '未识别')}\n\n"
        f"💡 **改进方向**:\n{result.get('improvement', '未识别')}"
    )


resume_match_tool = Tool(
    name="resume_match",
    func=_resume_match,
    description=(
        "评估简历与目标岗位JD的匹配度，输出百分制得分与改进建议。"
        "输入格式: 'JD:::\n{JD全文}\n\n简历:::\n{简历全文}'"
    ),
)

# ═══════════════════════════════════════════════════════════════
# 工具 4：简历优化（STAR法则）
# ═══════════════════════════════════════════════════════════════

STAR_OPTIMIZE_PROMPT = """你是一位顶级简历优化专家，擅长用 STAR 法则改写经历。

STAR 法则：
- S (Situation): 背景情境
- T (Task): 任务目标
- A (Action): 采取行动
- R (Result): 量化成果

【Few-Shot 示例】
优化前: "负责公司的数据分析工作，用Python做了一些报表"
优化后: "[S]公司业务数据分散在多个系统，[T]为支撑运营决策需整合数据并产出日报，[A]独立搭建Python自动化数据管道，整合3个业务系统数据源，用Pandas完成清洗与建模，[R]日报产出效率提升70%，月均节省20人天人力成本"

【目标岗位 JD】
{jd_text}

【原始经历文本】
{original_text}

请严格按以下格式输出优化结果：
1. 先输出改写后的简历描述（2-3个要点，每个要点包含完整的 STAR 结构）
2. 在末尾附上「优化对比」表格，清晰标注改动点

不要输出与优化无关的闲聊。"""


def _resume_optimize(input_text: str) -> str:
    """
    基于 STAR 法则优化简历经历描述。
    输入: "岗位JD:::\n{JD}\n\n原始经历:::\n{经历文本}"
    输出: 优化后文案 + 优化前后对比
    """
    jd_text = ""
    original_text = ""

    if "原始经历:::" in input_text:
        parts = input_text.split("原始经历:::", 1)
        jd_text = parts[0].replace("岗位JD:::", "").strip()
        original_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        jd_text = input_text[:500]
        original_text = input_text[500:]

    if not original_text:
        return "❌ 请提供需要优化的经历描述文本。"

    # 智能截断：尽量保留完整语义片段
    jd_text = _smart_truncate(jd_text, max_chars=2500)
    original_text = _smart_truncate(original_text, max_chars=2500)

    prompt = STAR_OPTIMIZE_PROMPT.format(
        jd_text=jd_text,
        original_text=original_text,
    )

    try:
        llm = get_llm(temperature=0.4, max_tokens=1536)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"❌ LLM 调用失败: {e}"

    return f"✨ **STAR 法则简历优化**\n\n{content}"


resume_optimize_tool = Tool(
    name="resume_optimize",
    func=_resume_optimize,
    description=(
        "使用 STAR 法则优化简历经历描述，突出量化成果。"
        "输入格式: '岗位JD:::\n{JD}\n\n原始经历:::\n{经历}'"
    ),
)

# ═══════════════════════════════════════════════════════════════
# 工具 5：面试题生成
# ═══════════════════════════════════════════════════════════════

INTERVIEW_QUESTION_PROMPT = """你是一位资深技术面试官，请为以下岗位生成面试题。

【目标公司/岗位】
{target}

【面经知识库参考内容】
{reference}

请生成以下三类面试题（每类3-5道）：
1. 🔧 技术题: 与岗位相关的专业技术问题
2. 📁 项目题: 关于过往项目经历的深挖问题
3. 💬 HR/行为题: 软技能、职业规划、团队协作类问题

每道题请包含：
- 题目本身
- 考察点说明
- 简要的答题思路提示

输出格式清晰，分类明确。"""


def _interview_question(target: str) -> str:
    """
    生成模拟面试题，从面经知识库检索参考并结合LLM生成。
    输入: 目标公司+岗位（如"字节跳动 数据分析实习"）
    输出: 技术题、项目题、HR题 三类，每类3-5道
    """
    from modules.rag_knowledge import load_vector_store
    from config.settings import VECTOR_STORE_DIR

    store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
    reference_text = "未检索到面经参考材料。"

    try:
        vector_store = load_vector_store(store_path)
        results = search_knowledge(vector_store, target, top_k=5)
        if results and "检索异常" not in results[0].get("content", ""):
            reference_text = "\n\n".join(
                [f"[来源: {r['source']}]\n{r['content']}" for r in results]
            )
    except (FileNotFoundError, RuntimeError):
        pass

    ck = _cache_key("iv", target)
    ch = _cache_get(ck)
    if ch is not None:
        return ch

    if not target.strip():
        return "❌ 请提供目标公司和岗位信息，如「字节跳动 数据分析实习」。"

    prompt = INTERVIEW_QUESTION_PROMPT.format(
        target=target.strip(),
        reference=reference_text[:2000],
    )

    try:
        llm = get_llm(temperature=0.5, max_tokens=2048)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"❌ LLM 调用失败: {e}"

    result = f"🎤 **模拟面试题生成** - {target}\n\n{content}"
    _cache_set(ck, result)
    return result


interview_question_tool = Tool(
    name="interview_question",
    func=_interview_question,
    description=(
        "为目标岗位生成模拟面试题，包含技术题、项目题、HR题三类。"
        "输入为目标公司+岗位，如'字节跳动 数据分析实习'。"
    ),
)

# ═══════════════════════════════════════════════════════════════
# 工具函数：智能文本截断
# ═══════════════════════════════════════════════════════════════

def _smart_truncate(text: str, max_chars: int = 2500) -> str:
    """
    智能截断文本，优先保留开头的关键信息和结尾的要求部分。
    
    策略：
    - ≤ max_chars: 不截断
    - > max_chars: 取开头 75% + 结尾 25%（因为 JD 的"任职要求"和"加分项"通常靠后）
    """
    if len(text) <= max_chars:
        return text
    
    head_size = int(max_chars * 0.75)
    tail_size = max_chars - head_size - 50  # 留 50 字符给分隔符
    
    head = text[:head_size]
    tail = text[-tail_size:]
    
    return head + "\n\n...(中间部分已省略)...\n\n" + tail


# ═══════════════════════════════════════════════════════════════
# 工具 6：投递进度管理
# ═══════════════════════════════════════════════════════════════

def _application_tracker(command: str) -> str:
    """
    投递进度管理：支持自然语言指令操作 CSV 投递记录。
    通过关键词识别指令类型（新增/查询/更新/删除），纯规则匹配保证响应速度。

    输入示例:
    - "新增 字节跳动 数据分析实习 2025-03-15 已投递 内推"
    - "查询"
    - "更新 1 面试中"
    - "删除 2"
    """
    cmd = command.strip()

    if not APPLICATIONS_CSV.exists():
        APPLICATIONS_CSV.write_text("公司,岗位,投递日期,状态,备注\n", encoding="utf-8")

    try:
        df = pd.read_csv(APPLICATIONS_CSV, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["公司", "岗位", "投递日期", "状态", "备注"])

    # ── 查询 ──
    if cmd.startswith("查询") or cmd.startswith("查看"):
        if df.empty:
            return "📭 当前无投递记录。使用「新增」命令添加第一条记录吧！"
        output = ["📋 **投递进度总览**\n"]
        for idx, row in df.iterrows():
            output.append(
                f"#{idx+1} | 🏢 {row['公司']} | 💼 {row['岗位']} | "
                f"📅 {row['投递日期']} | 📌 {row['状态']} | 📝 {row.get('备注', '')}"
            )
        return "\n".join(output)

    # ── 新增 ──
    if cmd.startswith("新增"):
        parts = cmd.replace("新增", "", 1).strip().split(maxsplit=4)
        if len(parts) < 3:
            return (
                "❌ 格式错误。正确格式：新增 公司名 岗位 投递日期 状态 [备注]\n"
                "示例：新增 字节跳动 数据分析实习 2025-03-15 已投递 内推"
            )
        company = parts[0]
        position = parts[1]
        date = parts[2]
        status = parts[3] if len(parts) > 3 else "已投递"
        note = parts[4] if len(parts) > 4 else ""

        new_row = pd.DataFrame([{
            "公司": company, "岗位": position,
            "投递日期": date, "状态": status, "备注": note,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"✅ 已新增投递记录：{company} - {position}"

    # ── 更新 ──
    if cmd.startswith("更新"):
        parts = cmd.replace("更新", "", 1).strip().split(maxsplit=1)
        if len(parts) < 2:
            return "❌ 格式错误。正确格式：更新 序号 新状态\n示例：更新 1 面试中"
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            return "❌ 序号必须为数字。"
        if idx < 0 or idx >= len(df):
            return f"❌ 序号超出范围（共 {len(df)} 条记录）。"
        df.at[idx, "状态"] = parts[1]
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"✅ 已更新 #{idx+1} 状态为：{parts[1]}"

    # ── 删除 ──
    if cmd.startswith("删除"):
        parts = cmd.replace("删除", "", 1).strip().split()
        if not parts:
            return "❌ 格式错误。正确格式：删除 序号\n示例：删除 1"
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            return "❌ 序号必须为数字。"
        if idx < 0 or idx >= len(df):
            return f"❌ 序号超出范围（共 {len(df)} 条记录）。"
        removed = df.iloc[idx]
        df = df.drop(idx).reset_index(drop=True)
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"✅ 已删除投递记录 #{idx+1}：{removed['公司']} - {removed['岗位']}"

    # ── 兜底 ──
    return (
        "❓ 无法识别的指令。支持以下操作：\n"
        "  • 新增 公司 岗位 日期 状态 [备注]\n"
        "  • 查询（查看全部）\n"
        "  • 更新 序号 新状态\n"
        "  • 删除 序号"
    )


application_tracker_tool = Tool(
    name="application_tracker",
    func=_application_tracker,
    description=(
        "管理实习投递进度记录，支持新增、查询、更新状态、删除操作。"
        "输入格式: '新增 公司 岗位 日期 状态' / '查询' / '更新 序号 新状态' / '删除 序号'"
    ),
)


# ═══════════════════════════════════════════════════════════════
# 工具集合导出
# ═══════════════════════════════════════════════════════════════

ALL_TOOLS = [
    job_search_tool,
    resume_parse_tool,
    resume_match_tool,
    resume_optimize_tool,
    interview_question_tool,
    application_tracker_tool,
]
