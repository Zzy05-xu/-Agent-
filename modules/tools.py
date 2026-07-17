"""
Agent ���ߺ���ģ��
���� LangChain Tool ��淶��װ 6 ��רҵ��ְ�������ߡ�
���й�������Ϊ�ַ��������Ϊ��ʽ���ַ�������ȫ���� ReAct Agent ���á�

�����嵥��
1. job_search_tool      - ��λ������RAG����JD֪ʶ�⣩
2. resume_parse_tool    - ����������PDF�ı���ȡ+�ṹ����
3. resume_match_tool    - ����ƥ�����֣�LLM������
4. resume_optimize_tool - �����Ż���STAR�����д��
5. interview_question_tool - ���������ɣ�RAG+LLM��
6. application_tracker_tool - Ͷ�ݽ��ȹ����CSV��ɾ�Ĳ飩
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
    return "{0}:{1}".format(prefix, hash(text))

def _cache_get(key: str):
    if key in _ttl_cache:
        val, ts = _ttl_cache[key]
        if time.time() - ts < _CACHE_TTL:
            return val
        del _ttl_cache[key]
    return None

def _cache_set(key: str, value):
    _ttl_cache[key] = (value, time.time())
    if len(_ttl_cache) > 200:
        old = sorted(_ttl_cache, key=lambda k: _ttl_cache[k][1])[:100]
        for k in old:
            _ttl_cache.pop(k, None)

_csv_lock = threading.Lock()

# ── 向量库存储名称 ──
JD_STORE_NAME = "jd_store"

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _llm_rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """LLM Re-Ranking: FAISS coarse �� LLM fine scoring �� Top K.
    
    Solves: FAISS L2 distance has limited Chinese semantic discrimination.
    Two unrelated JDs can appear close in vector space.
    LLM truly understands semantics for precise filtering.
    """
    if len(candidates) <= top_k:
        return candidates

    logger = get_logger(__name__)
    try:
        candidate_text = ""
        for i, item in enumerate(candidates, 1):
            src = item.get("source", "unknown")
            content = item["content"][:600]
            candidate_text += "\n### CANDIDATE {0}\nSource: {1}\nContent: {2}\n".format(i, src, content)

        rerank_prompt = (
            'You are a hiring expert. User is searching for internship positions with query: "' + query + '".\n\n'
            + 'Here are ' + str(len(candidates)) + ' candidate job descriptions. Score each (1-10) on relevance to the query.\n\n'
            + candidate_text + '\n'
            + 'Output STRICT JSON only:\n'
            + '{"rankings": [{"index": 1, "score": 8, "reason": "brief"}], "top_indices": [3, 1, 5]}\n'
            + 'top_indices lists candidate numbers from highest to lowest score.'
        )

        from config.settings import invoke_llm_with_retry
        response = invoke_llm_with_retry(rerank_prompt, temperature=0.1, max_tokens=1024, max_retries=2)
        parsed = _extract_json(response)

        if "parse_error" in parsed:
            logger.warning("Re-Ranking JSON parse failed, fallback to FAISS order")
            return candidates[:top_k]

        top_indices = parsed.get("top_indices", [])
        if not top_indices:
            return candidates[:top_k]

        reranked = []
        seen = set()
        for idx in top_indices:
            if isinstance(idx, int) and 1 <= idx <= len(candidates) and idx not in seen:
                seen.add(idx)
                reranked.append(candidates[idx - 1])

        for i, cand in enumerate(candidates, 1):
            if i not in seen and len(reranked) < top_k:
                reranked.append(cand)

        logger.info("Re-Ranking: {0} -> {1} results".format(len(candidates), len(reranked)))
        return reranked[:top_k]

    except Exception as e:
        logger.warning("Re-Ranking failed ({0}), fallback to FAISS order".format(e))
        return candidates[:top_k]

# ���� 1����λ����
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _job_search(query: str) -> str:
    """
    �� JD ֪ʶ���м���ƥ���ʵϰ��λ��
    ����: �ؼ��ʻ��λҪ������
    ���: ȥ�غ�� Top ƥ�� JD ��Ϣ
    """
    ck = _cache_key("js", query)
    ch = _cache_get(ck)
    if ch is not None:
        return ch

    from modules.rag_knowledge import load_vector_store
    from config.settings import VECTOR_STORE_DIR

    jd_store_path = os.path.join(str(VECTOR_STORE_DIR), JD_STORE_NAME)
    
    try:
        vector_store = load_vector_store(jd_store_path)
    except (FileNotFoundError, RuntimeError) as e:
        return (
            f"?? ֪ʶ��δ������{e}\n\n"
            "?? �����ڲ������֪ʶ�������е��������֪ʶ�⡹��"
            "�� JD ���澭�ĵ��������������С�"
        )

    results = search_knowledge(vector_store, query, top_k=15)

    if not results or "�����쳣" in results[0].get("content", ""):
        return "? δ�ҵ�ƥ��ĸ�λ��Ϣ���볢�Ե����ؼ��ʺ����ԡ�"

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
        return "? δ�ҵ�ƥ��ĸ�λ��Ϣ���볢�Ե����ؼ��ʺ����ԡ�"

    # LLM Re-Ranking: FAISS coarse -> LLM fine Top 3
    results = _llm_rerank(query, results, top_k=3)

    output_parts = [f"?? **��λ���������Top {len(results)}��**\n"]
    for i, item in enumerate(results, 1):
        output_parts.append(
            f"---\n"
            f"**#{i}** | ?? ��Դ: {item['source']} | ���ƶȾ���: {item['score']}\n\n"
            f"{item['content']}\n"
        )

    result = "\n".join(output_parts)
    _cache_set(ck, result)
    return result


job_search_tool = Tool(
    name="job_search",
    func=_job_search,
    description=(
        "��ʵϰ��λ֪ʶ���м���ƥ��ĸ�λ��Ϣ��"
        "������Ҫ���û������ض����������ݷ�������˿�������ʵϰ��λʱ�����ô˹��ߡ�"
        "����ӦΪ�ؼ��ʻ��λ������������'���ݷ���ʵϰ ������'��"
        "�����������˾������λְ����ְҪ��н�ʷ�Χ�ȹؼ���Ϣ��"
    ),
)

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���� 2����������
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

# ���� LLM �����ṹ����ȡ Prompt ����
RESUME_PARSE_PROMPT = """����һλרҵ�ļ�������ר�ҡ�������¼���ԭ������ȡ�ؼ���Ϣ�������ϸ� JSON ��ʽ���ء�

{resume_text}

�뷵������ JSON �ṹ��ֻ��� JSON����Ҫ�������֣���
{{
    "name": "��������δ�ҵ�������ַ�����",
    "education": [
        {{"school": "ѧУ��", "degree": "ѧ��������/˶ʿ/��ʿ��", "major": "רҵ", "year": "��ҵ���"}}
    ],
    "skills": ["����1", "����2", "����3"],
    "experience": [
        {{"company": "��˾/��֯", "role": "ְλ", "duration": "ʱ���", "description": "��������ժҪ"}}
    ],
    "projects": [
        {{"name": "��Ŀ����", "description": "��Ŀ����ժҪ", "tech_stack": ["ʹ�ü���1"]}}
    ]
}}

���ĳ���ֶ��޷�ʶ�𣬷��ؿ����� [] ����ַ�����
"""  # noqa: E501


def _resume_parse(file_path: str) -> str:
    """Parse PDF resume, extract structured info via LLM, return Markdown."""
    path = Path(file_path.strip().strip('"'))
    
    if not path.exists():
        return f"[ERROR] File not found: {file_path}\nPlease check the path or upload in Resume tab."

    if path.suffix.lower() != ".pdf":
        return f"[ERROR] Only PDF supported, got: {path.suffix}"

    # Step 1: Extract raw text from PDF
    try:
        reader = PdfReader(str(path))
        full_text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text_parts.append(text)
        full_text = "\n".join(full_text_parts)
    except Exception as e:
        return f"[ERROR] PDF parse failed: {e}"

    if not full_text.strip():
        return "[ERROR] PDF may be scanned/image, no readable text extracted."

    # Step 2: LLM structured extraction (analyze up to 8000 chars for completeness)
    text_to_analyze = full_text[:8000]
    prompt = RESUME_PARSE_PROMPT.format(resume_text=text_to_analyze)

    try:
        llm = get_llm(temperature=0.1, max_tokens=1536)
        response = llm.invoke(prompt)
        llm_output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        # Fallback: show raw text
        preview = full_text[:5000]
        return (
            f"[WARN] LLM structured parse failed: {e}\n\n"
            f"=== RAW RESUME TEXT (first {min(len(full_text), 5000)} chars) ===\n\n"
            f"{preview}"
            + ("\n...(truncated)" if len(full_text) > 5000 else "")
        )

    # JSON extraction
    parsed = _extract_json(llm_output)

    # Step 3: Format structured output
    if "parse_error" in parsed:
        return (
            f"[WARN] LLM returned invalid format. Raw output:\n\n{llm_output}\n\n"
            f"---\n=== RAW RESUME TEXT (first 5000 chars) ===\n\n{full_text[:5000]}"
        )

    result_parts = ["=== RESUME PARSED RESULT ===\n"]

    # Name
    name = parsed.get("name", "")
    if name:
        result_parts.append(f"[Name] {name}\n")

    # Education
    edu_list = parsed.get("education", [])
    if edu_list:
        result_parts.append("---\n[Education]")
        for edu in edu_list:
            parts = []
            if edu.get("school"): parts.append(edu["school"])
            if edu.get("degree"): parts.append(edu["degree"])
            if edu.get("major"): parts.append(edu["major"])
            if edu.get("year"): parts.append(f"({edu['year']})")
            result_parts.append(f"  - {' | '.join(parts)}")
    else:
        result_parts.append("---\n[Education] Not detected")

    # Skills
    skills = parsed.get("skills", [])
    if skills:
        result_parts.append("\n---\n[Skills]")
        for s in skills:
            result_parts.append(f"  - {s}")
    else:
        result_parts.append("\n---\n[Skills] Not detected")

    # Experience
    exp_list = parsed.get("experience", [])
    if exp_list:
        result_parts.append("\n---\n[Experience]")
        for exp in exp_list:
            company = exp.get("company", "")
            role = exp.get("role", "")
            duration = exp.get("duration", "")
            desc = exp.get("description", "")
            header = f"{company} - {role}" if company else role
            if duration: header += f" ({duration})"
            result_parts.append(f"  - **{header}**")
            if desc: result_parts.append(f"    {desc}")
    else:
        result_parts.append("\n---\n[Experience] Not detected")

    # Projects
    proj_list = parsed.get("projects", [])
    if proj_list:
        result_parts.append("\n---\n[Projects]")
        for proj in proj_list:
            name_p = proj.get("name", "")
            desc_p = proj.get("description", "")
            tech = ", ".join(proj.get("tech_stack", []))
            result_parts.append(f"  - **{name_p}**")
            if tech: result_parts.append(f"    Tech: {tech}")
            if desc_p: result_parts.append(f"    {desc_p}")
    else:
        result_parts.append("\n---\n[Projects] Not detected")

    # Full text preview for manual verification
    preview_len = min(len(full_text), 5000)
    result_parts.append(
        f"\n---\n=== RAW TEXT PREVIEW (first {preview_len} chars) ===\n{full_text[:preview_len]}"
        + ("\n...(truncated)" if len(full_text) > 5000 else "")
    )

    return "\n".join(result_parts)

resume_parse_tool = Tool(
    name="resume_parse",
    func=_resume_parse,
    description=(
        "����PDF�����ļ�����ȡ���ṹ��չʾ��������������ջ����Ŀ������"
        "������Ҫ�����û��ļ�������ʱ���ô˹��ߡ�"
        "����ӦΪPDF�ļ�����������·����"
    ),
)

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���� 3������ƥ������
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

MATCH_SCORE_PROMPT = """����һλ����HR����Ƹר�ҡ���������¼�����Ŀ���λJD��ƥ��ȡ�

��Ŀ���λ JD��
{jd_text}

����ְ�߼�����
{resume_text}

�뷵���ϸ��ʽ�� JSON����Ҫ�����������֣���
{{
    "score": 85,
    "core_match": "ƥ���1��ƥ���2��ƥ���3",
    "missing_skills": "ȱʧ����1��ȱʧ����2��ȱʧ����3",
    "improvement": "����Ľ�����1������2������3"
}}

���ֱ�׼: 90-100 �߶�ƥ�䣬80-89 ��ƥ�䣬70-79 ����ƥ�䣬<70 ������������
��͹����������������������ϸ������"""


def _extract_json(text: str) -> dict:
    """
    JSON ��ȡ�����׷�����
    ���ñ�׼ json.loads ������ʧ�ܺ��ô����ż������������� JSON��
    ��ʧ���򷵻�ԭʼ�ı���ǡ���� LLM �� JSON ǰ����Ӷ������ֵ����⡣
    """
    # ����1: ֱ�ӽ���
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ����2: �����ż�����ȡ
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

    # ����3: ����
    return {"raw_output": text, "score": "N/A", "parse_error": True}


def _resume_match(jd_text: str) -> str:
    """
    ����ƥ�����ֹ��ߡ�
    ����: "JD:::\n{JD�ı�}\n\n����:::\n{�����ı�}"
    ���: ƥ��ȵ÷֡�����ƥ��㡢ȱʧ���ܡ��Ľ�����
    """
    jd_content = ""
    resume_content = ""

    if "����:::" in jd_text:
        parts = jd_text.split("����:::", 1)
        jd_content = parts[0].replace("JD:::", "").strip()
        resume_content = parts[1].strip() if len(parts) > 1 else ""
    else:
        half = len(jd_text) // 2
        jd_content = jd_text[:half]
        resume_content = jd_text[half:]

    if not jd_content or not resume_content:
        return (
            "? ���ṩ������ JD �ͼ������ݡ�\n"
            "�����ʽʾ����\n"
            "JD:::\n{��λJD}\n\n����:::\n{����ȫ��}"
        )

    # ���ܽضϣ�������������������Ƭ��
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
        return f"? LLM ����ʧ��: {e}\n���� API ���ú����ԡ�"

    result = _extract_json(content)

    if "parse_error" in result:
        return f"?? ���ֽ����ʽ�쳣�������� LLM ԭʼ�����\n\n{content}"

    return (
        f"?? **����ƥ�������**\n\n"
        f"?? **ƥ��÷�**: {result.get('score', 'N/A')} / 100\n\n"
        f"? **����ƥ���**:\n{result.get('core_match', 'δʶ��')}\n\n"
        f"?? **ȱʧ����**:\n{result.get('missing_skills', 'δʶ��')}\n\n"
        f"?? **�Ľ�����**:\n{result.get('improvement', 'δʶ��')}"
    )


resume_match_tool = Tool(
    name="resume_match",
    func=_resume_match,
    description=(
        "����������Ŀ���λJD��ƥ��ȣ�����ٷ��Ƶ÷���Ľ����顣"
        "�����ʽ: 'JD:::\n{JDȫ��}\n\n����:::\n{����ȫ��}'"
    ),
)

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���� 4�������Ż���STAR����
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

STAR_OPTIMIZE_PROMPT = """����һλ���������Ż�ר�ң��ó��� STAR �����д������

STAR ����
- S (Situation): �����龳
- T (Task): ����Ŀ��
- A (Action): ��ȡ�ж�
- R (Result): �����ɹ�

��Few-Shot ʾ����
�Ż�ǰ: "����˾�����ݷ�����������Python����һЩ����"
�Ż���: "[S]��˾ҵ�����ݷ�ɢ�ڶ��ϵͳ��[T]Ϊ֧����Ӫ�������������ݲ������ձ���[A]�����Python�Զ������ݹܵ�������3��ҵ��ϵͳ����Դ����Pandas�����ϴ�뽨ģ��[R]�ձ�����Ч������70%���¾���ʡ20���������ɱ�"

��Ŀ���λ JD��
{jd_text}

��ԭʼ�����ı���
{original_text}

���ϸ����¸�ʽ����Ż������
1. �������д��ļ���������2-3��Ҫ�㣬ÿ��Ҫ����������� STAR �ṹ��
2. ��ĩβ���ϡ��Ż��Աȡ����������ע�Ķ���

��Ҫ������Ż��޹ص����ġ�"""


def _resume_optimize(input_text: str) -> str:
    """
    ���� STAR �����Ż���������������
    ����: "��λJD:::\n{JD}\n\nԭʼ����:::\n{�����ı�}"
    ���: �Ż����İ� + �Ż�ǰ��Ա�
    """
    jd_text = ""
    original_text = ""

    if "ԭʼ����:::" in input_text:
        parts = input_text.split("ԭʼ����:::", 1)
        jd_text = parts[0].replace("��λJD:::", "").strip()
        original_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        jd_text = input_text[:500]
        original_text = input_text[500:]

    if not original_text:
        return "? ���ṩ��Ҫ�Ż��ľ��������ı���"

    # ���ܽضϣ�����������������Ƭ��
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
        return f"? LLM ����ʧ��: {e}"

    return f"? **STAR ��������Ż�**\n\n{content}"


resume_optimize_tool = Tool(
    name="resume_optimize",
    func=_resume_optimize,
    description=(
        "ʹ�� STAR �����Ż���������������ͻ�������ɹ���"
        "�����ʽ: '��λJD:::\n{JD}\n\nԭʼ����:::\n{����}'"
    ),
)

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���� 5������������
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

INTERVIEW_QUESTION_PROMPT = """����һλ��������Թ٣���Ϊ���¸�λ���������⡣

��Ŀ�깫˾/��λ��
{target}

���澭֪ʶ��ο����ݡ�
{reference}

�������������������⣨ÿ��3-5������
1. ?? ������: ���λ��ص�רҵ��������
2. ?? ��Ŀ��: ���ڹ�����Ŀ��������������
3. ?? HR/��Ϊ��: ����ܡ�ְҵ�滮���Ŷ�Э��������

ÿ�����������
- ��Ŀ����
- �����˵��
- ��Ҫ�Ĵ���˼·��ʾ

�����ʽ������������ȷ��"""


def _interview_question(target: str) -> str:
    """
    ����ģ�������⣬���澭֪ʶ������ο������LLM���ɡ�
    ����: Ŀ�깫˾+��λ����"�ֽ����� ���ݷ���ʵϰ"��
    ���: �����⡢��Ŀ�⡢HR�� ���࣬ÿ��3-5��
    """
    from modules.rag_knowledge import load_vector_store
    from config.settings import VECTOR_STORE_DIR

    store_path = os.path.join(str(VECTOR_STORE_DIR), JD_STORE_NAME)
    reference_text = "δ�������澭�ο����ϡ�"

    try:
        vector_store = load_vector_store(store_path)
        results = search_knowledge(vector_store, target, top_k=5)
        if results and "�����쳣" not in results[0].get("content", ""):
            reference_text = "\n\n".join(
                [f"[��Դ: {r['source']}]\n{r['content']}" for r in results]
            )
    except (FileNotFoundError, RuntimeError):
        pass

    ck = _cache_key("iv", target)
    ch = _cache_get(ck)
    if ch is not None:
        return ch

    if not target.strip():
        return "? ���ṩĿ�깫˾�͸�λ��Ϣ���硸�ֽ����� ���ݷ���ʵϰ����"

    prompt = INTERVIEW_QUESTION_PROMPT.format(
        target=target.strip(),
        reference=reference_text[:2000],
    )

    try:
        llm = get_llm(temperature=0.5, max_tokens=2048)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"? LLM ����ʧ��: {e}"

    result = f"?? **ģ������������** - {target}\n\n{content}"
    _cache_set(ck, result)
    return result


interview_question_tool = Tool(
    name="interview_question",
    func=_interview_question,
    description=(
        "ΪĿ���λ����ģ�������⣬���������⡢��Ŀ�⡢HR�����ࡣ"
        "����ΪĿ�깫˾+��λ����'�ֽ����� ���ݷ���ʵϰ'��"
    ),
)

# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���ߺ����������ı��ض�
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _smart_truncate(text: str, max_chars: int = 2500) -> str:
    """
    ���ܽض��ı������ȱ����ͷ�Ĺؼ���Ϣ�ͽ�β��Ҫ�󲿷֡�
    
    ���ԣ�
    - �� max_chars: ���ض�
    - > max_chars: ȡ��ͷ 75% + ��β 25%����Ϊ JD ��"��ְҪ��"��"�ӷ���"ͨ������
    """
    if len(text) <= max_chars:
        return text
    
    # Note: uses char count, not token count. Chinese ~1.5 chars/token,
    # so max_chars=2500 gives ~1500 tokens headroom (well within limits).
    head_size = int(max_chars * 0.75)
    tail_size = max_chars - head_size - 50  # �� 50 �ַ����ָ��
    
    head = text[:head_size]
    tail = text[-tail_size:]
    
    return head + "\n\n...(�м䲿����ʡ��)...\n\n" + tail


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���� 6��Ͷ�ݽ��ȹ���
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _application_tracker(command: str) -> str:
    """
    Ͷ�ݽ��ȹ����֧����Ȼ����ָ����� CSV Ͷ�ݼ�¼��
    ͨ���ؼ���ʶ��ָ�����ͣ�����/��ѯ/����/ɾ������������ƥ�䱣֤��Ӧ�ٶȡ�

    ����ʾ��:
    - "���� �ֽ����� ���ݷ���ʵϰ 2025-03-15 ��Ͷ�� ����"
    - "��ѯ"
    - "���� 1 ������"
    - "ɾ�� 2"
    """
    cmd = command.strip()

    if not APPLICATIONS_CSV.exists():
        APPLICATIONS_CSV.write_text("��˾,��λ,Ͷ������,״̬,��ע\n", encoding="utf-8")

    try:
        df = pd.read_csv(APPLICATIONS_CSV, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["��˾", "��λ", "Ͷ������", "״̬", "��ע"])

    # ���� ��ѯ ����
    if cmd.startswith("��ѯ") or cmd.startswith("�鿴"):
        if df.empty:
            return "?? ��ǰ��Ͷ�ݼ�¼��ʹ�á�������������ӵ�һ����¼�ɣ�"
        output = ["?? **Ͷ�ݽ�������**\n"]
        for idx, row in df.iterrows():
            output.append(
                f"#{idx+1} | ?? {row['��˾']} | ?? {row['��λ']} | "
                f"?? {row['Ͷ������']} | ?? {row['״̬']} | ?? {row.get('��ע', '')}"
            )
        return "\n".join(output)

    # ���� ���� ����
    if cmd.startswith("����"):
        parts = cmd.replace("����", "", 1).strip().split(maxsplit=4)
        if len(parts) < 3:
            return (
                "? ��ʽ������ȷ��ʽ������ ��˾�� ��λ Ͷ������ ״̬ [��ע]\n"
                "ʾ�������� �ֽ����� ���ݷ���ʵϰ 2025-03-15 ��Ͷ�� ����"
            )
        company = parts[0]
        position = parts[1]
        date = parts[2]
        status = parts[3] if len(parts) > 3 else "��Ͷ��"
        note = parts[4] if len(parts) > 4 else ""

        new_row = pd.DataFrame([{
            "��˾": company, "��λ": position,
            "Ͷ������": date, "״̬": status, "��ע": note,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"? ������Ͷ�ݼ�¼��{company} - {position}"

    # ���� ���� ����
    if cmd.startswith("����"):
        parts = cmd.replace("����", "", 1).strip().split(maxsplit=1)
        if len(parts) < 2:
            return "? ��ʽ������ȷ��ʽ������ ��� ��״̬\nʾ�������� 1 ������"
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            return "? ��ű���Ϊ���֡�"
        if idx < 0 or idx >= len(df):
            return f"? ��ų�����Χ���� {len(df)} ����¼����"
        df.at[idx, "״̬"] = parts[1]
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"? �Ѹ��� #{idx+1} ״̬Ϊ��{parts[1]}"

    # ���� ɾ�� ����
    if cmd.startswith("ɾ��"):
        parts = cmd.replace("ɾ��", "", 1).strip().split()
        if not parts:
            return "? ��ʽ������ȷ��ʽ��ɾ�� ���\nʾ����ɾ�� 1"
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            return "? ��ű���Ϊ���֡�"
        if idx < 0 or idx >= len(df):
            return f"? ��ų�����Χ���� {len(df)} ����¼����"
        removed = df.iloc[idx]
        df = df.drop(idx).reset_index(drop=True)
        df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
        return f"? ��ɾ��Ͷ�ݼ�¼ #{idx+1}��{removed['��˾']} - {removed['��λ']}"

    # ���� ���� ����
    return (
        "? �޷�ʶ���ָ�֧�����²�����\n"
        "  ? ���� ��˾ ��λ ���� ״̬ [��ע]\n"
        "  ? ��ѯ���鿴ȫ����\n"
        "  ? ���� ��� ��״̬\n"
        "  ? ɾ�� ���"
    )


application_tracker_tool = Tool(
    name="application_tracker",
    func=_application_tracker,
    description=(
        "����ʵϰͶ�ݽ��ȼ�¼��֧����������ѯ������״̬��ɾ��������"
        "�����ʽ: '���� ��˾ ��λ ���� ״̬' / '��ѯ' / '���� ��� ��״̬' / 'ɾ�� ���'"
    ),
)


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ���߼��ϵ���
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

ALL_TOOLS = [
    job_search_tool,
    resume_parse_tool,
    resume_match_tool,
    resume_optimize_tool,
    interview_question_tool,
    application_tracker_tool,
]
