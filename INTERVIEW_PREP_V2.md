
# 🎯 实习求职智能助手 Agent — 面试深度准备手册（完整版）

> 原则：面试官不会让你念项目介绍，他们会追问技术决策、具体问题、以及你当时怎么想的。
> 本文档覆盖：技术栈详解、架构决策、追问应答、踩坑记录。每个技术选型都附面试标准回答。

---

## 一、30秒项目介绍（面试必问第一题）

> 我做了一个面向大学生实习求职的 AI Agent 平台。核心是用 **LangChain ReAct Agent** 框架搭建智能体，集成了 **6 个工具**（岗位检索、简历解析、匹配评分、STAR 优化、面试题生成、投递管理），底层接入了 **RAG 知识库**（FAISS + 本地 BAAI Embedding），前端用 **Streamlit** 做的。用户通过浏览器打开就能用自然对话完成求职全流程。
>
> 技术上我最满意的三点：一是引入了 **LLM Re-Ranking 重排序**（FAISS 粗筛 15 条 → LLM 逐条打分精选 Top 3），检索准确率大幅提升；二是做了**流式输出**，用户能看到 Agent 每一步在想什么、调了什么工具；三是设计了**三层异常降级**，不管是网络抖动还是 API 欠费，系统都不会崩，能给用户明确的解决引导。

---

## 二、技术栈全景图（逐个拆解）

### 🧠 LangChain（Agent 框架核心）

| 维度 | 说明 |
|------|------|
| **是什么** | LangChain 是当前最主流的 LLM 应用开发框架，提供 Agent、Chain、Tool、Memory 等抽象 |
| **在项目中怎么用的** | 使用 create_react_agent 构建 ReAct 范式的智能体；AgentExecutor 管理执行循环；Tool 类封装 6 个工具函数；PromptTemplate 管理中文系统提示词 |
| **为什么选它** | 相比原生 API 调用，LangChain 提供了完整的 Agent 生命周期管理（停止条件、错误处理、中间步骤记录）；相比 LangGraph（适合复杂多 Agent 编排），ReAct 对 6 个工具的单 Agent 场景最简洁够用 |
| **常见追问** | "为什么不用 AutoGPT/BabyAGI？" → 那些是实验性项目，LangChain 是工业级框架，文档完善，社区活跃 |

### 🤖 ReAct Agent（推理范式）

| 维度 | 说明 |
|------|------|
| **是什么** | ReAct = Reasoning + Acting，是 LLM Agent 的经典范式：Thought（思考）→ Action（调工具）→ Observation（看结果）→ Final Answer（回答）。循环执行直到得出结论或达到最大迭代次数 |
| **在项目中怎么用的** | 中文系统提示词定义了完整的 ReAct 格式规范，max_iterations=8 防止死循环，handle_parsing_errors=True 自动纠偏格式错误 |
| **为什么选它** | Function Calling 绑死 OpenAI 格式不可迁移；ReAct 是通用范式，切换 DeepSeek/OpenAI/国产模型都不需要改代码 |
| **常见追问** | "Agent 调用工具失败了怎么办？" → LLM 自动重试 3 次（tenacity 指数退避），超过则降级为基础对话模式 |

### 📚 RAG（检索增强生成）

| 维度 | 说明 |
|------|------|
| **是什么** | RAG = Retrieval-Augmented Generation，核心思想是把外部知识（JD、面经文档）向量化存储，用户提问时先检索相关文档片段，再拼接为 Prompt 让 LLM 基于真实资料回答——解决 LLM 的幻觉问题和知识时效性问题 |
| **在项目中怎么用的** | 收集 40+ 份 JD + 20+ 份面经 → 文本分割 → 向量化存入 FAISS → 用户提问时语义检索 → LLM 基于检索结果生成回答 |
| **为什么选它** | 没有 RAG，Agent 回答"找数据分析实习"只能胡编乱造；有了 RAG，Agent 能从真实 JD 库中检索匹配岗位，回答有据可查 |
| **常见追问** | "检索不准确怎么办？" → 我做了 LLM Re-Ranking：FAISS 粗筛 15 条 → LLM 给每条打分 → 取 Top 3，准确率比纯向量检索高很多（向量距离在中文字符上区分度有限） |

### 🔢 FAISS（向量数据库）

| 维度 | 说明 |
|------|------|
| **是什么** | Facebook 开源的向量相似度搜索库，可以理解为一个"只存数学向量的数据库"，通过向量的余弦/L2 距离衡量"语义相似度" |
| **在项目中怎么用的** | 所有 JD 和面经文本通过 Embedding 模型转成向量 → FAISS.from_documents() 构建索引 → save_local() 持久化到磁盘 → 使用时 similarity_search_with_score() 检索 Top K |
| **为什么选它** | 零部署（pip install faiss-cpu 即可）、本地文件持久化（copy 给别人直接能用）、适合万级以下文档片段。对比：Chroma 功能重叠但不如 FAISS 读取快；Milvus/Pinecone 需要独立服务部署，杀鸡不用牛刀 |
| **常见追问** | "FAISS 的原理是什么？" → 核心是 IVF（倒排索引）+ PQ（乘积量化），先聚类再在簇内搜索，牺牲少量精度换大量速度 |

### 🧬 Embedding（BAAI/bge-small-zh-v1.5）

| 维度 | 说明 |
|------|------|
| **是什么** | Embedding 模型把任意文本映射为一个固定维度的浮点数向量（本项目是 512 维），语义相近的文本其向量距离就小 |
| **在项目中怎么用的** | 使用 HuggingFace 的 sentence-transformers 加载 bge-small-zh-v1.5，CPU 推理，normalize_embeddings=True 做向量归一化，用于 JD 和面经文档的向量化存储与检索 |
| **为什么选它** | bge 系列是中文语义检索的 SOTA 开源模型，small 版 100MB 大小 + CPU 推理 10ms 以内，相比 m3e-base（400MB+）更轻量，相比 text2vec 检索评分更高。且完全本地运行，零 API 费用 |
| **常见追问** | "怎么评价 Embedding 模型的好坏？" → 看 MTEB/C-MTEB 榜单上的检索子项评分；bge-small 在中文检索任务上排名前 10，volume x speed 三者最优 |

### 📊 LLM Re-Ranking（重排序）

| 维度 | 说明 |
|------|------|
| **是什么** | 两阶段检索策略：第一阶段用向量数据库快速粗筛大量候选（速度快但精度有限），第二阶段用 LLM 对候选结果逐条打分排序（速度慢但精度高），最终取 Top K 返回 |
| **在项目中怎么用的** | FAISS 粗筛 15 条 → 构造 Re-Ranking Prompt（原文 + 每条候选内容）→ LLM 为每条打分（1-10）→ 按得分排序取 Top 3 返回 |
| **为什么选它** | 纯 FAISS 的 L2 距离对中文语义区分度不够（两个不相关的 JD 可能在向量空间距离很近），LLM 能真正理解语义做精准筛选 |
| **常见追问** | "成本怎么控制？" → 加 TTL 缓存（相同查询 5 分钟内不重复调用）；只用廉价模型（如 DeepSeek）做 Re-Ranking |


### 🖥️ Streamlit（Web UI）

| 维度 | 说明 |
|------|------|
| **是什么** | 专为数据/AI 应用设计的 Python Web 框架，写纯 Python 代码就能生成网页，不需要写 HTML/CSS/JS |
| **在项目中怎么用的** | 构建 4 个 Tab 页（智能对话/简历优化/模拟面试/投递管理）+ 侧边栏配置面板；st.chat_message 做对话气泡；st.data_editor 做可编辑表格；session_state 管理跨页面状态 |
| **为什么选它** | 项目目标是快速验证 AI Agent 能力，不需要单独开发前后端。Streamlit 让"写完 Python 就能跑网页"，极大降低 UI 开发成本。对比 Gradio：Gradio 适合单个 Demo，Streamlit 适合多页面复杂应用 |
| **常见追问** | "Streamlit 的缺点？" → 不支持多用户并发、页面状态复杂时 session_state 管理容易乱、生产部署不如 FastAPI+Svelte。但我这项目是个人工具级别，够用了 |

### ⚡ 流式输出（Streaming）

| 维度 | 说明 |
|------|------|
| **是什么** | LLM 正常是"一次性返回全部结果"，流式输出是"生成一个字就返回一个字"，类似 ChatGPT 逐字弹出的效果 |
| **在项目中怎么用的** | 使用 LangChain 的 astream_events（v2 版本 API），通过 threading + queue 把异步流包装成同步生成器。UI 层用 st.empty() 占位再逐 token 更新，同时监听 on_tool_start/on_tool_end 事件展示工具调用过程 |
| **为什么选它** | 用户体验问题。没有流式输出时，Agent 思考 15-30 秒用户只能干等，不知道到底在跑还是卡死了。流式输出让用户看到"Agent 正在思考 → 正在调工具 → 正在生成回答"，等待不再焦虑 |
| **常见追问** | "Windows 上 astream_events 有什么坑？" → asyncio 在 Windows 上默认事件循环有兼容问题，需要 new_event_loop() 手动创建 + threading 包装 |

### 🔁 Tenacity（自动重试）

| 维度 | 说明 |
|------|------|
| **是什么** | Python 重试库，支持指数退避（1s → 2s → 4s → ...）、最大重试次数、自定义重试条件 |
| **在项目中怎么用的** | 所有 LLM 调用都通过 invoke_llm_with_retry() 包装，max_retries=3，wait_exponential 指数退避，retry_if_exception_type(Exception) 捕获所有异常 |
| **为什么选它** | DeepSeek API 偶发 503 或网络超时，不加重试用户一次失败就报错体验很差。加了重试后，偶发网络抖动对用户完全透明 |
| **常见追问** | "指数退避为什么好？" → 避免大量请求同时重试造成雪崩，给服务端缓冲时间 |

### 🛡️ 三层异常降级

| 维度 | 说明 |
|------|------|
| **是什么** | 不是统一 catch 所有异常，而是按异常类型分类处理，每类有独立的降级策略 |
| **在项目中怎么用的** | Layer 1（API 超时）→ 用纯 LLM 直接回答 + 引导使用侧边栏功能；Layer 2（Token 超限）→ 截断对话历史后用 LLM 回答；Layer 3（其他错误）→ 通用兜底回答 + 错误信息展示。Embedding 也做了三级降级：本地模型 → 在线 API → 报错提示 + 解决方案 |
| **为什么选它** | 核心原则：降级后服务不中断，所有错误路径都返回对用户有意义的文本，不白屏、不报看不懂的错 |
| **常见追问** | "为什么分开处理而不是统一 try-catch？" → 超时可以重试，Token 超限需要截断上下文而不是重试，策略完全不同 |

---

## 三、架构设计类追问

### Q：「为什么不直接调 GPT API，为什么要用 Agent？」

> 普通 Chatbot 是"一问一答"，Agent 是"规划-执行-反馈"的闭环。
>
> 举个例子：用户说"帮我找数据分析实习，再看看我的简历匹不匹配"。普通 Chatbot 只能编造岗位信息或给泛泛建议。我的 Agent 会自动：Thought → Action(调 job_search 工具拿到真实 JD) → Observation → Thought → Action(调 resume_match 评分) → Final Answer(整合两次工具调用的结果输出)。
>
> 核心区别：Agent 能自主判断什么时候该调用哪个工具，而 Chatbot 只能基于训练数据猜答案。

### Q：「系统提示词怎么设计的？」

> 四个设计原则：
> 1. **角色锚定**—首句"你是专业求职辅导助手"，角色越具体行为越可控
> 2. **工具使用规范**—明确告诉它"先分析需求，再决定调用哪些工具"、"不要编造信息，基于工具返回数据回答"
> 3. **ReAct 格式约束**—给出正确示例和错误示例，配合 handle_parsing_errors=True 自动纠偏
> 4. **降级指引**—"工具调用失败时友好告知用户并给出替代建议"
>
> 小技巧：在工具 description 字段写明"什么时候该调用这个工具"，比在系统提示词里列举效果好得多——LangChain 会把 description 注入 prompt。

### Q：「chunk_size 为什么 JD 用 1500，面经用 800？」

> 试过 500/800/1000/1500/2000 五个值。JD 文档结构完整（岗位职责+任职要求+加分项通常 1500-3000 字），用 1500 保证一份 JD 最多切 2-3 块，信息完整；面经是问答结构、单个题目较短，用 800 保证检索粒度精确，不会把 5 道题的答案糊在一起。overlap 保留 10-15% 避免边界切断关键信息。

### Q：「检索结果不准怎么办？」

> 三个层面：1）阈值过滤，相似度距离 > 阈值的直接丢弃；2）LLM Re-Ranking（FAISS 粗筛 15 → LLM 打分取 Top 3）；3）检查源文档覆盖度。目前项目做了前两层。

### Q：「工具之间有依赖关系怎么处理？」

> Agent 自动串行处理。ReAct 每轮只能调一个工具，但多轮之间是串行依赖的——第一轮拿到岗位信息，第二轮基于岗位信息去匹配简历。不需要我手动编排，系统提示词引导它"先搜索再评估"就行。

---

## 四、工程踩坑记录

| # | 坑 | 怎么发现的 | 怎么解决的 |
|---|-----|----------|----------|
| 1 | HuggingFaceEmbeddings 模块路径找不到 | import 报错 | 它在 langchain_community 而不是 langchain_huggingface——LangChain 版本间经常模块迁移 |
| 2 | HuggingFace 被墙，模型下载失败 | 构建知识库时报连接超时 | 设置 HF_ENDPOINT=https://hf-mirror.com 国内镜像；或提供在线 Embedding API 降级方案 |
| 3 | sentence-transformers 强升 numpy 2.x，pandas 不兼容 | 安装依赖后 pandas 报错 | 锁定 numpy<2.0，实测 numpy==1.26.4 兼容全链 |
| 4 | Streamlit expander 不能嵌套 | UI 代码报 rendering error | 拆成 flat 结构，每个 expander 独立 |
| 5 | astream_events 在 Windows 上报 event loop 错误 | 启用流式输出时报 asyncio 错误 | new_event_loop() + threading 包装 |
| 6 | DeepSeek API 偶发 503 | 用户反馈偶尔报错 | 引入 tenacity 指数退避重试 3 次 |
| 7 | LLM 返回 JSON 格式不稳定 | 简历解析工具间歇失败 | 四级 JSON 提取器兜底：直接解析 → 正则匹配代码块 → 范围提取 → 大括号计数 |
| 8 | .env 文件被 git 追踪 | git status 发现 | 在 settings.py 启动时增加 git ls-files 检查并输出安全警告 |


---

## 五、面试高频追问速查清单

| 面试官问 | 答对要点 |
|---------|---------|
| Agent 和普通 Chatbot 区别？ | Agent 有工具调用能力，自主规划执行；Chatbot 只能基于训练数据回答 |
| 为什么用 ReAct 不用 Function Calling？ | ReAct 是通用范式不绑模型；Function Calling 只对 OpenAI 格式 |
| chunk_size 怎么确定的？ | 做过实验，JD 信息完整需 1500，面经问答结构需 800 |
| FAISS 原理？ | IVF + PQ 近似检索，先聚类再簇内搜索 |
| 为什么选 bge？ | 中文检索 SOTA、100MB 轻量、CPU 推理、零费用、C-MTEB 检索子项 Top 10 |
| 怎么保证检索准确？ | FAISS 粗筛 + LLM Re-Ranking 精排 + 相似度阈值过滤 |
| 流式输出怎么实现的？ | astream_events v2 + threading + queue，兼容 Windows |
| 系统崩了怎么办？ | 三层降级：超时→直接 LLM 回答；Token 超限→截断后回答；其他→兜底提示 |
| JSON 解析不稳定怎么办？ | 四级提取器：json.loads → 正则代码块 → 范围提取 → 大括号计数 |
| 最大的收获？ | 不是代码本身，是"真实环境远比理想环境复杂"，从 API 抖动到网络封锁到依赖冲突，每个问题都需要 trade-off |

---

## 六、简历项目描述（两种篇幅）

### 精简版（一页简历，约 100 字）

`
实习求职智能助手 Agent | Python, LangChain, FAISS, Streamlit | 独立开发

- 基于 LangChain ReAct Agent 构建智能体，集成岗位检索、简历匹配、
  STAR优化、面试题生成等 6 个工具，实现"思考-行动-回答"推理链路
- 搭建 RAG 知识库（FAISS + BAAI/bge 本地 Embedding），
  引入 LLM Re-Ranking 提升检索准确率
- 实现流式输出、LLM 自动重试、三层异常降级等可靠性设计
- Streamlit 构建 4-Tab Web 界面，覆盖求职全流程
`

### 详细版（两页简历，约 160 字）

`
实习求职智能助手 Agent | Python, LangChain, FAISS, Streamlit, HuggingFace | 独立开发

- 基于 LangChain create_react_agent 构建 ReAct 范式 AI Agent，设计中文系统
  提示词定义角色与行为规范，集成 6 个工具自动完成岗位搜索-简历优化-面试准备的闭环
- 搭建 RAG 全链路：收集 40+ JD + 20+ 面经 → 按文档类型智能分割
  (JD 1500/面经 800) → FAISS 向量存储 → LLM Re-Ranking 重排序提升检索相关性
- 工程层面：流式输出实时展示 Agent 思考链；tenacity 指数退避自动重试；
  三层异常降级(超时/Token超限/调用失败)保证服务不中断；四级 JSON 解析兜底
- 使用 Streamlit 构建 4 个 Tab 页 + 侧边栏配置面板，集成可视化投递管理看板
`

---

## 七、相关资源

- 📂 GitHub: [github.com/Zzy05-xu/jobsearch-agent](https://github.com/Zzy05-xu/jobsearch-agent)
- 📖 完整架构文档：见仓库 README.md
- 🔧 问题排查记录：见仓库 TROUBLESHOOTING.md
