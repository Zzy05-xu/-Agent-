"""
实习求职智能助手 Agent - Streamlit 主程序入口

页面结构：
- 侧边栏：API 配置、知识库管理、使用说明
- 主区域 4 个 Tab：
  1. 💬 智能对话助手 - 多轮对话 + 思考链展示
  2. 📄 简历优化 - 上传 PDF + 匹配评分 + STAR 优化
  3. 🎤 模拟面试 - 面试题生成 + 问答模拟
  4. 📌 投递管理 - 可视化投递进度管理
"""
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ═══════════════════════════════════════════════════════════════
# 页面基础配置
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="实习求职智能助手 Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 将项目根目录加入 sys.path，确保模块导入正常
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
from modules.rag_knowledge import build_knowledge_base, load_vector_store
from modules.agent_core import run_agent
from modules.tools import (
    _resume_parse,
    _resume_match,
    _resume_optimize,
    _interview_question,
    _application_tracker,
)

# ── 自定义 CSS 样式（适配深色模式） ──
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .tool-result {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .stExpander details {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    """初始化所有 session_state 变量，避免重复加载。"""
    defaults = {
        "chat_history": [],        # 对话历史 [(role, content), ...]
        "vector_store": None,      # 向量库实例缓存
        "agent_executor": None,    # Agent 实例缓存
        "resume_text": "",         # 已解析的简历文本
        "resume_path": "",         # 用户上传的简历路径
        "interview_active": False, # 模拟面试是否进行中
        "interview_questions": "", # 当前面试题
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ═══════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ 配置面板")
    
    # ── 1. API 配置区域 ──
    with st.expander("🔑 API 配置", expanded=not bool(OPENAI_API_KEY)):
        st.caption("临时覆盖环境变量，仅本次会话有效")
        temp_api_key = st.text_input(
            "API Key",
            value=OPENAI_API_KEY,
            type="password",
            placeholder="sk-...",
        )
        temp_base_url = st.text_input(
            "Base URL",
            value=OPENAI_BASE_URL,
            placeholder="https://api.openai.com/v1",
        )
        temp_llm_model = st.text_input(
            "LLM 模型",
            value=LLM_MODEL_NAME,
            placeholder="gpt-3.5-turbo / deepseek-chat",
        )
        emb_mode = get_embedding_mode()
        mode_label = f"Embedding ({emb_mode})" if emb_mode else "Embedding（未初始化）"
        temp_emb_model = st.text_input(
            mode_label,
            value=LOCAL_EMBEDDING_MODEL,
            disabled=True,
            help="本地模型: pip install sentence-transformers | 备选: 在.env配置EMBEDDING_API_KEY",
        )
        if st.button("✅ 应用配置", use_container_width=True):
            update_api_config(temp_api_key, temp_base_url, temp_llm_model, temp_emb_model)
            st.session_state.vector_store = None  # 切换模型后需重建向量库
            st.success("配置已更新！")

    st.divider()
    
    # ── 2. 知识库管理 ──
    with st.expander("📚 知识库管理", expanded=True):
        st.caption("将 data 目录下的 JD / 面经文档索引为向量库")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔨 构建知识库", use_container_width=True):
                if not temp_api_key:
                    st.error("请先配置 API Key！")
                else:
                    with st.spinner("正在构建知识库（首次需加载本地 Embedding 模型约 100MB）..."):
                        try:
                            store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
                            vs = build_knowledge_base(str(DATA_DIR), store_path)
                            st.session_state.vector_store = vs
                            st.success("✅ 知识库构建成功！")
                        except Exception as e:
                            import traceback
                            err_msg = str(e)
                            st.error(f"构建失败: {err_msg}")
                            st.caption(f"💡 提示: 请确认已执行 pip install sentence-transformers 且网络可访问 huggingface.co")
        with col2:
            if st.button("🔄 加载知识库", use_container_width=True):
                if not temp_api_key:
                    st.error("请先配置 API Key！")
                else:
                    try:
                        store_path = os.path.join(str(VECTOR_STORE_DIR), "jd_store")
                        vs = load_vector_store(store_path)
                        st.session_state.vector_store = vs
                        st.success("✅ 知识库已加载！")
                    except Exception as e:
                        st.error(f"加载失败: {e}")

    st.divider()
    
    # ── 3. 使用说明 ──
    with st.expander("📖 使用说明"):
        st.markdown("""
        **快速开始：**
        1. 在「API 配置」中填入你的 Key
        2. 点击「构建知识库」初始化 RAG
        3. 切换到「智能对话助手」开始使用
        
        **4大功能模块：**
        - 💬 **对话助手**: 多轮对话，AI帮你解决求职问题
        - 📄 **简历优化**: 上传简历，匹配JD，STAR改写
        - 🎤 **模拟面试**: 生成面试题，模拟问答
        - 📌 **投递管理**: 跟踪你的投递进度
        
        **兼容接口：**
        支持 OpenAI / DeepSeek 等兼容接口
        在「API 配置」中修改 Base URL 即可
        """)

# ═══════════════════════════════════════════════════════════════
# 主区域标题
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<h1 class="main-header">🎯 实习求职智能助手 Agent</h1>',
    unsafe_allow_html=True,
)
st.caption(
    "基于 LangChain ReAct Agent + RAG 知识库 | 覆盖岗位检索、简历优化、面试备考、投递管理全流程"
)

# ═══════════════════════════════════════════════════════════════
# 4 个 Tab 页
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 智能对话助手",
    "📄 简历优化",
    "🎤 模拟面试",
    "📌 投递管理",
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: 智能对话助手
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.subheader("💬 与求职助手对话")

    # ── 对话历史展示 ──
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.chat_history:
            st.info(
                "👋 你好！我是你的专属求职助手。\n\n"
                "我可以帮你：\n"
                "- 🔍 搜索匹配的实习岗位\n"
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
                            st.caption(f"输入: `{action.tool_input[:200]}`")
                            st.markdown(
                                f'<div class="tool-result">{obs[:500]}</div>',
                                unsafe_allow_html=True,
                            )

    # ── 输入区域 ──
    user_input = st.chat_input("输入你的求职问题...", key="chat_input")
    if user_input:
        # 添加用户消息
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 构建 LangChain 格式的对话历史（最近 10 轮，避免 Token 超限）
        lc_history = []
        for msg in st.session_state.chat_history[-20:]:  # 最多 10 轮=20 条
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

        with st.spinner("🤔 Agent 正在思考..."):
            result = run_agent(user_input, chat_history=lc_history)
            answer = result["output"]
            steps = result.get("intermediate_steps", [])

        # 添加助手消息
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "steps": steps,
        })
        st.rerun()

    # ── 清空对话按钮 ──
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ 清空对话", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# Tab 2: 简历优化
# ═══════════════════════════════════════════════════════════════

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
            # 保存到临时目录
            temp_path = RESUME_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.resume_path = str(temp_path)

            with st.spinner("正在解析简历..."):
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
                with st.spinner("正在评估匹配度..."):
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
                with st.spinner("正在优化简历..."):
                    opt_input = f"岗位JD:::\n{jd_input}\n\n原始经历:::\n{st.session_state.resume_text[:1500]}"
                    try:
                        opt_result = _resume_optimize(opt_input)
                        st.markdown(opt_result)
                    except Exception as e:
                        st.error(f"优化失败: {e}")

# ═══════════════════════════════════════════════════════════════
# Tab 3: 模拟面试
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🎤 模拟面试练习")

    # ── 岗位选择 ──
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
                with st.spinner("正在生成面试题..."):
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

    # ── 面试题展示与模拟回答 ──
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
                        from config.settings import get_llm
                        llm = get_llm(temperature=0.3, max_tokens=1024)
                        feedback_prompt = (
                            f"你是一位面试官。以下是候选人对面试题的回答，请给出评价和改进建议。\n\n"
                            f"【面试题背景】\n{st.session_state.interview_questions[:1000]}\n\n"
                            f"【候选人回答】\n{user_answer}\n\n"
                            f"请从以下维度给出反馈：\n"
                            f"1. 整体结构（是否清晰有逻辑）\n"
                            f"2. 内容深度（是否体现技术能力和项目经验）\n"
                            f"3. 改进建议（具体可操作的建议）\n"
                        )
                        response = llm.invoke(feedback_prompt)
                        feedback = response.content if hasattr(response, "content") else str(response)
                        st.markdown("#### 📝 面试官反馈")
                        st.markdown(feedback)
                    except Exception as e:
                        st.error(f"生成反馈失败: {e}")

# ═══════════════════════════════════════════════════════════════
# Tab 4: 投递管理（交互式编辑 + 统计看板）
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📌 投递进度管理")

    import pandas as pd
    from config.settings import APPLICATIONS_CSV

    # ── 读取数据 ──
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

    # ── 统计面板 ──
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

        # 状态分布图
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
        st.info("👋 还没有投递记录，快开始投递吧！")

    # ── 交互式表格 ──
    st.markdown("---")
    st.markdown("#### 📋 投递记录（点击单元格直接编辑）")

    df["序号"] = range(1, len(df) + 1)
    cols_order = ["序号", "公司", "岗位", "投递日期", "状态", "备注"]
    df_display = df[cols_order] if not df.empty else df

    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        num_rows="dynamic",  # 支持动态增删行
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

    # ── 保存按钮 ──
    col_save, col_export, col_quick = st.columns([1, 1, 2])
    with col_save:
        if st.button("💾 保存修改", type="primary", use_container_width=True):
            # 移除序号列后保存
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

    # ── 快捷操作 ──
    with col_quick:
        st.markdown("**快捷指令**（新增/删除）")
        quick_input = st.text_input(
            "指令",
            placeholder="新增 字节跳动 数据分析 2025-03-15 已投递",
            key="quick_cmd_input",
            label_visibility="collapsed",
        )
    col_go, _ = st.columns([1, 5])
    with col_go:
        if st.button("🚀 执行指令", use_container_width=True):
            if quick_input.strip():
                try:
                    result = _application_tracker(quick_input)
                    st.success(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"执行失败: {e}")