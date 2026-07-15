"""
实习求职智能助手 Agent - Streamlit 主程序入口（优化版）

优化点：
- 流式输出：实时展示 Agent 思考过程和回答
- 配置持久化：切换 API 无需重新设置
- 投递管理：快捷表单替代文本指令
- 知识库：增量更新 + 文档统计
- Agent 缓存：会话级复用

页面结构：
- 侧边栏：API 配置、知识库管理、使用说明
- 主区域 4 个 Tab：
  1. 💬 智能对话助手 - 多轮对话 + 流式输出 + 思考链展示
  2. 📄 简历优化 - 上传 PDF + 匹配评分 + STAR 优化
  3. 🎤 模拟面试 - 面试题生成 + 问答模拟
  4. 📌 投递管理 - 可视化投递进度管理
"""
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ================================================================
# 页面基础配置
# ================================================================

st.set_page_config(
    page_title="实习求职智能助手 Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.schema import HumanMessage, AIMessage

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL_NAME,
    LOCAL_EMBEDDING_MODEL,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    DATA_DIR,
    RESUME_DIR,
    update_api_config,
    ensure_data_dirs,
    get_embedding_mode,
)
from modules.rag_knowledge import build_knowledge_base, load_vector_store, add_documents_to_store
from modules.agent_core import run_agent, run_agent_streaming
from modules.tools import (
    _resume_parse,
    _resume_match,
    _resume_optimize,
    _interview_question,
    _application_tracker,
)

# CSS
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .tool-result { background-color: rgba(128, 128, 128, 0.1); border-radius: 8px; padding: 12px; margin: 8px 0; font-size: 0.9rem; }
    .stExpander details { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 8px; }
    .thinking-box { background-color: rgba(100, 149, 237, 0.1); border-left: 3px solid #6495ED; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 0.85rem; }
    .streaming-cursor::after { content: "▌"; animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ================================================================
# Session State
# ================================================================

def init_session_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "agent_executor": None,
        "resume_text": "",
        "resume_path": "",
        "interview_active": False,
        "interview_questions": "",
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "LLM_MODEL_NAME": LLM_MODEL_NAME,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ================================================================
# 侧边栏
# ================================================================

with st.sidebar:
    st.title("⚙️ 配置面板")

    # API 配置
    with st.expander("🔑 API 配置", expanded=not bool(st.session_state.get("OPENAI_API_KEY", ""))):
        st.caption("配置将自动保存到本次会话")
        temp_api_key = st.text_input(
            "API Key",
            value=st.session_state.get("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-...",
            key="sidebar_api_key",
        )
        temp_base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            placeholder="https://api.openai.com/v1",
            key="sidebar_base_url",
        )
        temp_llm_model = st.text_input(
            "LLM 模型",
            value=st.session_state.get("LLM_MODEL_NAME", "gpt-3.5-turbo"),
            placeholder="gpt-3.5-turbo / deepseek-chat",
            key="sidebar_llm_model",
        )
        emb_mode = get_embedding_mode()
        mode_label = f"Embedding ({emb_mode})" if emb_mode else "Embedding（未初始化）"
        st.text_input(mode_label, value=LOCAL_EMBEDDING_MODEL, disabled=True, key="sidebar_emb_model")

        if st.button("✅ 应用配置", use_container_width=True):
            update_api_config(temp_api_key, temp_base_url, temp_llm_model)
            st.session_state.OPENAI_API_KEY = temp_api_key
            st.session_state.OPENAI_BASE_URL = temp_base_url
            st.session_state.LLM_MODEL_NAME = temp_llm_model
            st.session_state.vector_store = None
            st.success("✅ 配置已更新并保存！")

    st.divider()

    # 知识库管理
    with st.expander("📚 知识库管理", expanded=True):
        st.caption("将 data 目录下的 JD / 面经文档索引为向量库")

        # 文档统计
        data_path = Path(str(DATA_DIR))
        if data_path.exists():
            jd_count = len(list(data_path.rglob("jd_samples/*.txt"))) + len(list(data_path.rglob("jd_samples/*.md")))
            iv_count = len(list(data_path.rglob("interview/*.txt"))) + len(list(data_path.rglob("interview/*.md")))
            st.caption(f"📊 JD: {jd_count} 个 | 面经: {iv_count} 个")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔨 构建知识库", use_container_width=True, help="全量重建知识库"):
                if not st.session_state.get("OPENAI_API_KEY", ""):
                    st.error("请先配置 API Key！")
                else:
                    with st.spinner("正在构建知识库（首次需加载本地 Embedding 模型约 100MB）..."):
                        try:
                            store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
                            vs = build_knowledge_base(str(DATA_DIR), store_path)
                            st.session_state.vector_store = vs
                            st.success("✅ 知识库构建成功！")
                        except Exception as e:
                            st.error(f"构建失败: {e}")
                            st.caption("💡 提示: pip install sentence-transformers 且网络可访问 huggingface.co")
        with col2:
            if st.button("🔄 加载知识库", use_container_width=True, help="加载已有知识库"):
                if not st.session_state.get("OPENAI_API_KEY", ""):
                    st.error("请先配置 API Key！")
                else:
                    try:
                        store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
                        vs = load_vector_store(store_path)
                        st.session_state.vector_store = vs
                        st.success("✅ 知识库已加载！")
                    except Exception as e:
                        st.error(f"加载失败: {e}")

        # 增量更新按钮
        if st.button("📥 增量更新", use_container_width=True, help="仅索引新增文档，不重建全文索引"):
            if not st.session_state.get("OPENAI_API_KEY", ""):
                st.error("请先配置 API Key！")
            else:
                try:
                    store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
                    vs = add_documents_to_store(str(DATA_DIR), store_path)
                    st.session_state.vector_store = vs
                    st.success("✅ 知识库增量更新成功！")
                except FileNotFoundError:
                    st.error("尚无知识库，请先「构建知识库」")
                except Exception as e:
                    st.error(f"增量更新失败: {e}")

        # 清空知识库
        if st.button("🗑️ 清空知识库", use_container_width=True, help="删除向量库缓存文件（JD和面经文档不受影响）"):
            import shutil
            store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
            if os.path.exists(store_path):
                try:
                    shutil.rmtree(store_path)
                    st.session_state.vector_store = None
                    st.success("✅ 知识库缓存已清空！下次使用时需重新构建。")
                except Exception as e:
                    st.error(f"清空失败: {e}")
            else:
                st.info("知识库缓存不存在，无需清空。")


    st.divider()

    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        **快速开始：**
        1. 在「API 配置」中填入你的 Key
        2. 点击「构建知识库」初始化 RAG
        3. 切换到「智能对话助手」开始使用

        **4大功能模块：**
        - 💬 **对话助手**: 多轮对话 + 流式输出，AI帮你解决求职问题
        - 📄 **简历优化**: 上传简历，匹配JD，STAR改写
        - 🎤 **模拟面试**: 生成面试题，模拟问答
        - 📌 **投递管理**: 跟踪你的投递进度

        **兼容接口：**
        支持 OpenAI / DeepSeek 等兼容接口
        """)

# ================================================================
# 主区域标题
# ================================================================

st.markdown(
    '<h1 class="main-header">🎯 实习求职智能助手 Agent</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "基于 LangChain ReAct Agent + RAG 知识库 | 流式输出 · 自动重试 · AI重排序 | 覆盖求职全流程"
)

# ================================================================
# 4 个 Tab
# ================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 智能对话助手",
    "📄 简历优化",
    "🎤 模拟面试",
    "📌 投递管理",
])

# ================================================================
# Tab 1: 智能对话助手（流式输出）
# ================================================================

with tab1:
    st.subheader("💬 与求职助手对话")

    col_stream, col_clear = st.columns([4, 1])
    with col_stream:
        use_streaming = st.toggle("⚡ 流式输出", value=True, help="实时显示 Agent 思考过程")
    with col_clear:
        if st.button("🗑️ 清空对话", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.agent_executor = None



    # 对话历史展示
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.chat_history:
            st.info(
                "👋 你好！我是你的专属求职助手。\n\n"
                "我可以帮你：\n"
                "- 🔍 搜索匹配的实习岗位（AI 重排序确保准确性）\n"
                "- 📊 评估简历与岗位的匹配度\n"
                "- ✨ 用 STAR 法则优化简历\n"
                "- 🎤 生成模拟面试题\n"
                "- 📌 管理投递进度\n\n"
                "试试输入「帮我找数据分析的实习岗位」开始吧！"
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    with st.expander("🔍 查看思考过程"):
                        for i, (action, obs) in enumerate(msg["steps"], 1):
                            st.caption(f"**步骤 {i}** → 调用工具: `{action.tool}`")
                            st.caption(f"输入: `{str(action.tool_input)[:200]}`")
                            st.markdown(
                                f'<div class="tool-result">{str(obs)[:800]}</div>',
                                unsafe_allow_html=True,
                            )

    # 输入区域
    user_input = st.chat_input("输入你的求职问题...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        lc_history = []
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"][:2000]))

        if use_streaming:
            # 流式输出模式
            with st.chat_message("assistant"):
                full_text = ""
                thinking_steps = []
                status_placeholder = st.empty()
                text_placeholder = st.empty()
                thinking_expander = st.expander("🔍 查看思考过程", expanded=False)

                status_placeholder.markdown("🤔 *Agent 正在思考...*")

                try:
                    for event in run_agent_streaming(user_input, chat_history=lc_history, session_id="default"):
                        if event["type"] == "tool_start":
                            status_placeholder.markdown(
                                f'<div class="thinking-box">🔧 正在调用: **{event["tool"]}**</div>',
                                unsafe_allow_html=True,
                            )
                            thinking_steps.append({
                                "tool": event["tool"],
                                "input": event.get("input", ""),
                                "output": "",
                            })

                        elif event["type"] == "tool_end":
                            if thinking_steps:
                                thinking_steps[-1]["output"] = event.get("output", "")
                            status_placeholder.markdown(
                                f'<div class="thinking-box">✅ 工具 **{event["tool"]}** 执行完成</div>',
                                unsafe_allow_html=True,
                            )

                        elif event["type"] == "token":
                            full_text += event["content"]
                            text_placeholder.markdown(full_text + "▌")

                        elif event["type"] == "error":
                            full_text = event.get("output", full_text)
                            text_placeholder.markdown(full_text)

                        elif event["type"] == "done":
                            status_placeholder.empty()
                            text_placeholder.markdown(full_text)

                    if thinking_steps:
                        with thinking_expander:
                            for i, step in enumerate(thinking_steps, 1):
                                st.caption(f"**步骤 {i}** → 调用工具: `{step['tool']}`")
                                st.caption(f"输入: `{step['input'][:200]}`")
                                st.markdown(
                                    f'<div class="tool-result">{step["output"][:800]}</div>',
                                    unsafe_allow_html=True,
                                )

                    if not full_text:
                        full_text = "Agent 未返回有效回答，请重试。"

                except Exception as e:
                    full_text = f"流式输出异常: {e}\n\n请尝试关闭流式输出后重试。"
                    text_placeholder.markdown(full_text)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_text,
                    "steps": [(type("Action", (), {"tool": s["tool"], "tool_input": s["input"]})(), s["output"]) for s in thinking_steps],
                })
        else:
            # 非流式模式
            with st.spinner("🤔 Agent 正在思考..."):
                result = run_agent(user_input, chat_history=lc_history, session_id="default")
                answer = result["output"]
                steps = result.get("intermediate_steps", [])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "steps": steps,
            })

        st.rerun()

# ================================================================
# Tab 2: 简历优化
# ================================================================

with tab2:
    st.subheader("📄 简历优化中心")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📤 上传简历")
        uploaded_file = st.file_uploader(
            "选择 PDF 简历文件",
            type=["pdf"],
            help="上传你的 PDF 格式简历",
        )

        if uploaded_file is not None:
            temp_path = RESUME_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.resume_path = str(temp_path)

            with st.spinner("正在解析简历（LLM 结构化提取）..."):
                try:
                    resume_result = _resume_parse(str(temp_path))
                    st.session_state.resume_text = resume_result
                except Exception as e:
                    st.error(f"解析失败: {e}")
                    st.session_state.resume_text = ""

        if st.session_state.resume_text:
            st.markdown("#### 解析结果预览")
            st.markdown(st.session_state.resume_text[:800] + "...")

    with col_right:
        st.markdown("### 📋 输入目标岗位 JD")
        jd_input = st.text_area(
            "粘贴目标岗位的 JD（岗位描述）",
            height=200,
            placeholder="粘贴实习岗位的 JD 到这里...\n\n例如：\n岗位名称：数据分析实习生\n职责：...\n要求：...",
        )

        if st.button("📊 匹配度评分", type="primary", use_container_width=True):
            if not st.session_state.resume_text:
                st.error("请先上传并解析简历！")
            elif not jd_input.strip():
                st.error("请输入目标岗位 JD！")
            else:
                with st.spinner("正在评估匹配度（LLM 自动重试）..."):
                    match_input = f"JD:::\n{jd_input}\n\n简历:::\n{st.session_state.resume_text[:2000]}"
                    try:
                        match_result = _resume_match(match_input)
                        st.markdown(match_result)
                    except Exception as e:
                        st.error(f"评分失败: {e}")

        if st.button("✨ STAR 优化经历", type="primary", use_container_width=True):
            if not st.session_state.resume_text:
                st.error("请先上传并解析简历！")
            elif not jd_input.strip():
                st.error("请输入目标岗位 JD！")
            else:
                with st.spinner("正在优化简历（STAR法则改写）..."):
                    opt_input = f"岗位JD:::\n{jd_input}\n\n原始经历:::\n{st.session_state.resume_text[:1500]}"
                    try:
                        opt_result = _resume_optimize(opt_input)
                        st.markdown(opt_result)
                    except Exception as e:
                        st.error(f"优化失败: {e}")

# ================================================================
# Tab 3: 模拟面试
# ================================================================

with tab3:
    st.subheader("🎤 模拟面试练习")

    target_job = st.text_input(
        "目标公司和岗位",
        placeholder="例如：字节跳动 数据分析实习",
        help="输入你正在准备面试的目标公司和岗位",
    )

    col_gen, col_clear_q = st.columns([2, 1])
    with col_gen:
        if st.button("🎲 生成面试题", type="primary", use_container_width=True):
            if not target_job.strip():
                st.error("请先输入目标公司和岗位！")
            else:
                with st.spinner("正在生成面试题（RAG检索面经 + LLM生成）..."):
                    try:
                        questions = _interview_question(target_job)
                        st.session_state.interview_questions = questions
                        st.session_state.interview_active = True
                    except Exception as e:
                        st.error(f"生成失败: {e}")
    with col_clear_q:
        if st.button("🔄 重置", use_container_width=True):
            st.session_state.interview_active = False
            st.session_state.interview_questions = ""
            st.rerun()

    if st.session_state.interview_questions:
        st.markdown("---")
        st.markdown(st.session_state.interview_questions)

        st.markdown("---")
        st.markdown("### ✍️ 模拟回答练习")
        user_answer = st.text_area(
            "选择一道面试题，在这里写下你的回答...",
            height=150,
            placeholder="尝试用 STAR 法则回答：先描述情境和任务，再说明你的行动和成果...",
        )

        if st.button("💡 获取回答反馈", type="secondary"):
            if not user_answer.strip():
                st.warning("请先写下你的回答！")
            else:
                with st.spinner("正在生成反馈..."):
                    try:
                        from config.settings import invoke_llm_with_retry
                        feedback_prompt = (
                            f"你是一位面试官。以下是候选人对面试题的回答，请给出评价和改进建议。\n\n"
                            f"【面试题背景】\n{st.session_state.interview_questions[:1000]}\n\n"
                            f"【候选人回答】\n{user_answer}\n\n"
                            f"请从以下维度给出反馈：\n"
                            f"1. 整体结构（是否清晰有逻辑）\n"
                            f"2. 内容深度（是否体现技术能力和项目经验）\n"
                            f"3. 改进建议（具体可操作的建议）\n"
                        )
                        feedback = invoke_llm_with_retry(feedback_prompt, temperature=0.3, max_tokens=1024)
                        st.markdown("#### 📝 面试官反馈")
                        st.markdown(feedback)
                    except Exception as e:
                        st.error(f"生成反馈失败: {e}")

# ================================================================
# Tab 4: 投递管理（优化版）
# ================================================================

with tab4:
    st.subheader("📌 投递进度管理")

    import pandas as pd
    from config.settings import APPLICATIONS_CSV

    def load_applications():
        try:
            if APPLICATIONS_CSV.exists():
                df = pd.read_csv(APPLICATIONS_CSV, encoding="utf-8")
                if df.empty:
                    df = pd.DataFrame(columns=["公司", "岗位", "投递日期", "状态", "备注"])
                return df
        except Exception:
            pass
        return pd.DataFrame(columns=["公司", "岗位", "投递日期", "状态", "备注"])

    df = load_applications()

    # 统计面板
    if not df.empty:
        st.markdown("#### 📊 投递统计")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            total = len(df)
            st.metric("📬 总投递数", total)
        with col_s2:
            in_progress = len(df[df["状态"].isin(["已投递", "初筛中", "笔试", "面试中"])])
            st.metric("🔄 进行中", in_progress)
        with col_s3:
            offers = len(df[df["状态"] == "已拿Offer"])
            st.metric("🎉 Offer", offers)
        with col_s4:
            rejected = len(df[df["状态"] == "已拒"])
            st.metric("❌ 已拒", rejected)
        with col_s5:
            rate = f"{offers / total * 100:.0f}%" if total > 0 else "0%"
            st.metric("📈 Offer率", rate)

        st.markdown("---")
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.markdown("**📊 状态分布**")
            status_counts = df["状态"].value_counts()
            st.bar_chart(status_counts, use_container_width=True)
        with col_chart2:
            st.markdown("**🏢 投递公司 Top 5**")
            company_counts = df["公司"].value_counts().head(5)
            st.bar_chart(company_counts, use_container_width=True)
    else:
        st.info("👋 还没有投递记录，使用下方表单快速添加！")

    # 快捷添加表单
    st.markdown("---")
    st.markdown("#### ➕ 快捷添加投递")
    with st.form("quick_add_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            new_company = st.text_input("🏢 公司", placeholder="例如：字节跳动")
        with col_f2:
            new_position = st.text_input("💼 岗位", placeholder="例如：数据分析实习")
        with col_f3:
            new_status = st.selectbox("📌 状态", ["已投递", "初筛中", "笔试", "面试中", "已拿Offer", "已拒", "已放弃"])

        col_f4, col_f5 = st.columns([2, 1])
        with col_f4:
            new_note = st.text_input("📝 备注（可选）", placeholder="例如：内推、官网投递")
        with col_f5:
            new_date = st.date_input("📅 投递日期")
            submitted = st.form_submit_button("✅ 添加投递记录", type="primary", use_container_width=True)

        if submitted:
            if not new_company.strip() or not new_position.strip():
                st.error("公司和岗位不能为空！")
            else:
                new_row = pd.DataFrame([{
                    "公司": new_company.strip(),
                    "岗位": new_position.strip(),
                    "投递日期": new_date.strftime("%Y-%m-%d"),
                    "状态": new_status,
                    "备注": new_note.strip(),
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
                st.success(f"✅ 已添加：{new_company} - {new_position}")
                st.rerun()

    # 交互式表格
    st.markdown("---")
    st.markdown("#### 📋 投递记录（点击单元格直接编辑）")

    if not df.empty:
        df["序号"] = range(1, len(df) + 1)
        cols_order = ["序号", "公司", "岗位", "投递日期", "状态", "备注"]
        df_display = df[cols_order]
    else:
        df_display = pd.DataFrame(columns=["序号", "公司", "岗位", "投递日期", "状态", "备注"])

    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "序号": st.column_config.NumberColumn("序号", disabled=True),
            "公司": st.column_config.TextColumn("🏢 公司"),
            "岗位": st.column_config.TextColumn("💼 岗位"),
            "投递日期": st.column_config.TextColumn("📅 投递日期"),
            "状态": st.column_config.SelectboxColumn(
                "📌 状态",
                options=["已投递", "初筛中", "笔试", "面试中", "已拿Offer", "已拒", "已放弃"],
                default="已投递",
            ),
            "备注": st.column_config.TextColumn("📝 备注"),
        },
        key="app_table_editor",
    )

    col_save, col_export = st.columns([1, 1])
    with col_save:
        if st.button("💾 保存修改", type="primary", use_container_width=True):
            save_df = edited_df.drop(columns=["序号"], errors="ignore")
            save_df.to_csv(APPLICATIONS_CSV, index=False, encoding="utf-8")
            st.success("✅ 投递记录已保存！")
            st.rerun()
    with col_export:
        if not df.empty:
            csv_data = df.drop(columns=["序号"], errors="ignore").to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 导出 CSV",
                csv_data,
                "投递记录.csv",
                "text/csv",
                use_container_width=True,
            )
