# 🎯 实习求职智能助手 Agent — 完整项目文档

> **基于 LangChain ReAct Agent + RAG 知识库 + Streamlit 的一站式实习求职 AI 平台**
> 
> 作者：许绍钧 | 技术栈：Python + LangChain + FAISS + DeepSeek + HuggingFace | 更新日期：2026-07-14

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构图](#2-技术架构图)
3. [完整运行流程](#3-完整运行流程)
4. [文件逐一详解](#4-文件逐一详解)
   - [4.1 根目录文件](#41-根目录文件)
   - [4.2 config/ 配置层](#42-config-配置层)
   - [4.3 modules/ 核心逻辑层](#43-modules-核心逻辑层)
   - [4.4 data/ 数据层](#44-data-数据层)
5. [每个代码文件的详细说明](#5-每个代码文件的详细说明)
6. [快速启动指南](#6-快速启动指南)
7. [技术亮点（简历提炼版）](#7-技术亮点简历提炼版)
8. [优化记录](#8-优化记录)

---

## 1. 项目概述

### 做什么的？

这是一个面向**大学生实习求职**全流程的 AI 助手。用户通过浏览器打开一个 Web 页面，就能用自然对话的方式完成：

| 功能 | 说明 |
|------|------|
| 🔍 **岗位检索** | 从本地知识库中搜索匹配的实习岗位（支持数据分析、后端、前端、算法、产品等多个方向） |
| 📄 **简历优化** | 上传 PDF 简历 → AI 自动提取结构化信息 → 与目标 JD 匹配打分 → 用 STAR 法则改写经历 |
| 🎤 **模拟面试** | 生成技术题 / 项目题 / HR 行为题三类面试题，用户回答后获取面试官风格反馈 |
| 📌 **投递管理** | 可视化表格记录投递进度，支持增删改查 + 统计看板（Offer 率、状态分布图等） |

### 怎么实现的？

底层是一个 **ReAct Agent**（"思考→行动→观察→回答"循环），它持有 6 个工具函数，能自主决定调用哪个工具、传什么参数。同时集成了 **RAG（检索增强生成）** 知识库：所有 JD 和面经文档被转成向量存到本地，Agent 可以用语义搜索找到最相关的信息。

---

## 2. 技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    🖥️  Streamlit Web UI                      │
│  main.py (555行) — 4个Tab页 + 侧边栏 + Session State        │
├─────────────────────────────────────────────────────────────┤
│               🧠 Agent Core (agent_core.py)                  │
│  ReAct 范式: Thought → Action → Observation → Final Answer │
│  多轮对话记忆 (最近10轮) + 三层异常降级                      │
├──────────────────┬──────────────────────────────────────────┤
│   🔧 6个Tool      │        📚 RAG 知识库 (rag_knowledge.py)  │
│   (tools.py)      │                                          │
│                   │  文档加载 → 文本分割 → FAISS向量化        │
│ 1.岗位检索 ───────┼─→ 语义检索 (search_knowledge)            │
│ 2.简历解析        │  支持 .txt / .md / .pdf                  │
│ 3.简历匹配评分    │                                          │
│ 4.STAR简历优化    │  Embedding: BAAI/bge-small-zh-v1.5       │
│ 5.面试题生成 ─────┼─→ 面经知识库检索                         │
│ 6.投递管理        │  (本地CPU运行，零费用)                    │
├──────────────────┴──────────────────────────────────────────┤
│               ⚙️ 配置层 (config/)                            │
│  settings.py: API配置 + LLM参数化缓存 + Embedding三级降级    │
│  logger.py:   统一日志系统 (按天轮转)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 完整运行流程

用户启动程序后，整个系统的运行流程如下：

```
Step 1: 启动
  └─ streamlit run main.py
  └─ main.py 导入所有模块
  └─ config/settings.py 加载 .env 环境变量
  └─ 自动创建 data/ 子目录
  └─ 检查 .env 是否被 git 追踪（安全告警）
  └─ config/logger.py 初始化日志系统

Step 2: 构建知识库（用户在侧边栏点击按钮）
  └─ 调用 modules/rag_knowledge.py → build_knowledge_base()
  └─ 子步骤:
      ├─ load_documents(): 递归扫描 data/jd_samples/ 和 data/interview/
      │   用 TextLoader 加载 .txt/.md, PyPDFLoader 加载 .pdf
      ├─ split_documents(): RecursiveCharacterTextSplitter 分割
      │   chunk_size=500, chunk_overlap=50
      └─ build_vector_store(): FAISS 向量化 + 保存到 data/vector_store/

Step 3: 用户在 Tab1 输入"帮我找后端开发的实习岗位"
  └─ main.py 构建 LangChain 对话历史（最近10轮）
  └─ 调用 modules/agent_core.py → run_agent(query, chat_history)
  └─ Agent 内部循环（最多8轮）:
      ├─ Thought: Agent分析用户意图 → "需要调用 job_search 工具"
      ├─ Action: job_search
      ├─ Action Input: "后端开发 实习"
      ├─ Observation: 从 FAISS 检索 Top3 最相似 JD 片段
      ├─ Thought: "信息足够了，可以整理回答"
      └─ Final Answer: 格式化展示岗位信息
  └─ main.py 展示回答 + 可展开的思考链

Step 4: 用户在 Tab2 上传简历 PDF
  └─ main.py 保存到 data/resume/
  └─ 调用 tools.py → _resume_parse()
  └─ 子步骤:
      ├─ pypdf 逐页提取原始文本
      ├─ LLM 分析文本，返回 JSON {name, education, skills, experience, projects}
      └─ 格式化为 Markdown 展示
  └─ 用户粘贴 JD → 点击"匹配度评分"
      └─ _resume_match() → LLM 返回 {score, core_match, missing_skills, improvement}
  └─ 用户点击"STAR优化"
      └─ _resume_optimize() → LLM 用 STAR 法则改写经历

Step 5: 用户在 Tab3 输入目标岗位 → 点击"生成面试题"
  └─ 调用 tools.py → _interview_question()
  └─ 先从 RAG 知识库检索面经作为参考材料
  └─ LLM 生成三类题目（技术/项目/HR）
  └─ 用户写下回答 → LLM 给出多维度反馈

Step 6: 用户在 Tab4 管理投递
  └─ st.data_editor 渲染可编辑表格（单元格内直接修改）
  └─ 新增/删除行 → 点击"保存修改" → 写入 data/applications.csv
  └─ 统计看板实时展示: 总投递数/进行中/Offer数/Offer率 + 状态柱状图 + 公司Top5
```

---

## 4. 文件逐一详解

### 4.1 根目录文件

| 文件 | 大小 | 作用 |
|------|------|------|
| `main.py` | 555行 / 24KB | **Streamlit 主入口**。定义了整个 Web UI 的页面结构：4个Tab页 + 侧边栏。负责 Session State 管理、用户交互响应、模块调用编排 |
| `requirements.txt` | 1.5KB | Python 依赖清单。锁定了 langchain / faiss-cpu / streamlit / pypdf / pandas 的版本号，确保可复现 |
| `.env.example` | 1.2KB | 环境变量模板。包含 OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL_NAME / LOCAL_EMBEDDING_MODEL 的填写说明。**可安全提交到 git** |
| `.env` | 1.3KB | 真实环境变量。包含实际的 API Key。**.gitignore 已排除，不可提交** |
| `.gitignore` | 432B | Git 忽略规则。排除 .env、__pycache__、vector_store、logs、虚拟环境等敏感和自动生成的文件 |
| `README.md` | 本文档 | 项目完整分析文档 |
| `INTERVIEW_PREP.md` | 12KB | 面试准备指南。教用户如何在简历上描述这个项目 + 常见面试问题回答思路 |
| `INTERVIEW_PREP_V2.md` | 7.5KB | 面试准备指南补充版 |
| `TROUBLESHOOTING.md` | 7.5KB | 问题排障记录。记录了开发过程中遇到的技术坑和解决方案 |

### 4.2 config/ 配置层

| 文件 | 行数 | 作用 |
|------|------|------|
| `config/__init__.py` | 2行 | Python 包标识文件。使 config 目录可作为模块导入 |
| `config/settings.py` | 247行 / 9KB | **全局配置核心**。管理 API Key、LLM 实例、Embedding 实例、数据目录路径 |
| `config/logger.py` | 102行 / 2.9KB | **统一日志系统**。TimedRotatingFileHandler 按天轮转，同时输出控制台和文件 |

### 4.3 modules/ 核心逻辑层

| 文件 | 行数 | 作用 |
|------|------|------|
| `modules/__init__.py` | 2行 | Python 包标识文件 |
| `modules/agent_core.py` | 246行 / 9.4KB | **ReAct Agent 核心**。系统提示词、Agent 创建、调用入口、异常降级 |
| `modules/rag_knowledge.py` | 282行 / 9.3KB | **RAG 知识库**。文档加载→分割→向量化→检索，完整 RAG 管线 |
| `modules/tools.py` | 669行 / 26KB | **6个 Agent 工具函数**。最复杂的模块，包含岗位检索、简历解析、简历匹配、STAR优化、面试题生成、投递管理 |

### 4.4 data/ 数据层

| 目录/文件 | 说明 |
|-----------|------|
| `data/jd_samples/` | 6 个示例岗位 JD（.txt格式） |
| `data/jd_samples/jd_data_analyst_internet.txt` | 数据分析实习（互联网）- 未来星球科技 |
| `data/jd_samples/jd_business_analyst_ecommerce.txt` | 商业分析实习 - 电商方向 |
| `data/jd_samples/jd_backend_dev.txt` | 后端开发实习（Java）- 星辰科技 |
| `data/jd_samples/jd_algorithm_intern.txt` | 算法实习（推荐系统）- 极光智能 |
| `data/jd_samples/jd_frontend_dev.txt` | 前端开发实习 - 蓝图科技 |
| `data/jd_samples/jd_product_intern.txt` | 产品实习 - 远航互娱 |
| `data/interview/` | 4 份面经资料 |
| `data/interview/interview_da_tech1.txt` | 数据分析技术面高频题（15题，含SQL/Python/统计学/项目深挖/HR） |
| `data/interview/interview_da_biz2.txt` | 数据分析业务面高频题 |
| `data/interview/interview_backend_tech.txt` | 后端开发技术面高频题（10题，含Java/Spring/数据库/中间件/系统设计） |
| `data/interview/interview_frontend_tech.txt` | 前端开发技术面高频题（7题，含JS核心/ReactVue/性能优化） |
| `data/resume/` | 简历上传临时目录（运行时自动创建） |
| `data/vector_store/` | FAISS 向量库持久化目录（构建知识库后生成） |
| `data/applications.csv` | 投递记录 CSV（运行时自动创建） |
| `logs/` | 应用日志目录（运行时自动创建，按天轮转保留7天） |

---

## 5. 每个代码文件的详细说明

---

### 📄 `config/__init__.py` (2 行)

```python
# 空文件，仅标记 config 为 Python package
```

**作用**: 让 Python 把 `config/` 目录识别为可导入的包，这样其他模块可以写 `from config.settings import ...`。

---

### 📄 `config/settings.py` (247 行)

**整体作用**: 全局配置中心，负责 4 件事：加载环境变量、管理 LLM/Embedding 实例、管理数据目录路径、运行时安全检测。

#### 逐段详解：

```
第1-30行   : 文档字符串 + imports
             ├─ load_dotenv: 从 .env 文件加载环境变量
             ├─ ChatOpenAI: LangChain 封装的标准 LLM 接口
             └─ OpenAIEmbeddings: OpenAI 兼容的 Embedding 接口

第33-44行  : 环境变量读取
             ├─ OPENAI_API_KEY: API 密钥（必填，否则无法运行）
             ├─ OPENAI_BASE_URL: API 地址（默认 OpenAI，可改为 DeepSeek）
             ├─ LLM_MODEL_NAME: 模型名（默认 gpt-3.5-turbo）
             ├─ LOCAL_EMBEDDING_MODEL: 本地 Embedding 模型名
             ├─ EMBEDDING_API_KEY / EMBEDDING_API_BASE: Embedding API 备选方案
             └─ EMBEDDING_MODEL_NAME: 在线 Embedding 模型名

第47-52行  : 数据目录路径定义
             ├─ DATA_DIR → data/
             ├─ JD_SAMPLES_DIR → data/jd_samples/
             ├─ INTERVIEW_DIR → data/interview/
             ├─ RESUME_DIR → data/resume/
             ├─ VECTOR_STORE_DIR → data/vector_store/
             └─ APPLICATIONS_CSV → data/applications.csv

第55-70行  : ensure_data_dirs() 函数
             自动创建所有数据子目录和空的 applications.csv

第72-80行  : _check_env_safety() 函数
             启动时执行 git ls-files --error-unmatch .env 检查 .env 是否被追踪
             如是，输出安全警告（不阻塞运行）

第85-91行  : 全局缓存变量
             ├─ _llm_cache: dict — 按 (temperature, max_tokens) 键缓存多个 LLM 实例
             ├─ _embedding_instance: 缓存的 Embedding 实例
             └─ _embedding_mode: 当前模式 "local" | "api" | None

第94-110行 : _try_get_local_embedding() 函数
             尝试加载本地 HuggingFace Embedding 模型
             需要 pip install sentence-transformers
             模型: BAAI/bge-small-zh-v1.5（中文语义检索最优）
             首次运行自动下载约 100MB

第113-120行: _create_api_embedding() 函数
             创建 OpenAI 兼容的在线 Embedding 实例
             作为本地模型不可用时的备选方案

第123-157行: get_llm(temperature, max_tokens) 函数 【核心】
             ┌─ 参数化缓存机制
             ├─ 按 (temperature, max_tokens) 构建 key
             ├─ 如果缓存中存在，直接返回（避免重复创建）
             ├─ 否则创建新的 ChatOpenAI 实例存入缓存
             └─ 不同场景用不同 temperature:
                  · Agent 主调用: 0.2（平衡准确和灵活）
                  · 简历匹配评分: 0.1（追求一致性）
                  · STAR 简历优化: 0.4（需要创造性改写）
                  · 面试题生成: 0.5（需要题目多样性）

第160-198行: get_embedding() 函数 【核心】
             ┌─ 三级自动降级策略
             ├─ Level 1: 本地 BAAI/bge-small-zh-v1.5（零费用，推荐）
             ├─ Level 2: OpenAI 兼容 Embedding API（需配置 Key）
             └─ Level 3: 抛出 RuntimeError + 解决方案提示
             首次成功即缓存 _embedding_instance，后续直接返回

第201-203行: get_embedding_mode() — 返回当前模式字符串

第206-212行: reset_instances() — 清空所有缓存（切换 API 配置时调用）

第215-220行: update_api_config() — Streamlit 侧边栏调用的动态配置更新
```

**关键设计决策**:
- 为什么用参数化缓存而不是单例？因为不同工具需要不同 temperature，单例会互相覆盖
- 为什么先尝试本地 Embedding？零 API 费用，中文语义检索效果好，离线可用

---

### 📄 `config/logger.py` (102 行)

**整体作用**: 统一日志系统，替代代码中散落的 `print()` 语句。

#### 逐段详解：

```
第1-18行   : 文档字符串 + imports
             ├─ logging: Python 标准库日志框架
             └─ TimedRotatingFileHandler: 按时间自动轮转日志文件

第20-24行  : 模块级变量
             ├─ _loggers: dict — 按 name 缓存 logger 实例
             ├─ _initialized: bool — 防止重复初始化
             ├─ LOG_DIR: logs/ 目录
             ├─ LOG_FILE: logs/app.log
             └─ LOG_LEVEL: 从环境变量读取，默认 INFO

第27-28行  : _ensure_log_dir() — 创建 logs/ 目录

第31-63行  : _init_root_logger() — 初始化根 logger
             ├─ 创建 "jobsearch_agent" 根 logger
             ├─ 配置 Console Handler: 输出到 stdout（Streamlit 终端可见）
             ├─ 配置 File Handler: TimedRotatingFileHandler
             │   └─ when="midnight": 每天午夜轮转
             │   └─ backupCount=7: 保留最近 7 天日志
             ├─ 格式: [2026-07-14 10:30:00] [INFO   ] [模块名] 消息内容
             └─ 文件初始化失败时降级为仅控制台输出

第66-82行  : get_logger(name) 函数
             各模块调用 get_logger(__name__) 获取自己的 logger
             按 name 缓存，避免重复创建

第85-88行  : get_log_file_path() 函数
             供 Streamlit UI 显示日志文件下载链接
```

**使用方式**：
```python
from config.logger import get_logger
logger = get_logger(__name__)
logger.info("正在加载知识库...")
logger.error("加载失败", exc_info=True)  # 自动记录完整堆栈
```

---

### 📄 `modules/__init__.py` (2 行)

```python
# 空文件，标记 modules 为 Python package
```

---
### 📄 `modules/agent_core.py` (246 行)

**整体作用**: ReAct Agent 的核心模块。定义系统提示词、创建 Agent 实例、提供调用入口、全链路异常降级。

#### 逐段详解：

```
第1-17行   : 文档字符串 + imports
             ├─ create_react_agent: LangChain 官方 ReAct Agent 工厂函数
             ├─ AgentExecutor: 封装 Agent 的执行循环和错误处理
             ├─ ConversationBufferMemory: 对话历史管理（已 import，预留扩展）
             ├─ PromptTemplate: 提示词模板引擎
             └─ BaseMessage: LangChain 消息基类（用于 chat_history 类型标注）

第22-60行  : REACT_SYSTEM_PROMPT（系统提示词）【核心】
             这是整个 Agent 的"人格设定"，定义了：
             ├─ 角色定位: "专业求职辅导助手，专注于帮助大学生准备实习求职"
             ├─ 核心能力清单: 5 大能力
             ├─ 行为规范: 先分析→再调用工具→基于结果回答→不编造→用 Markdown
             ├─ 可用工具: {tools}（运行时由 LangChain 自动注入工具列表）
             ├─ 工具名称: {tool_names}（运行时注入）
             └─ ReAct 格式规范: Thought → Action → Action Input → Observation → Final Answer

第64-70行  : REACT_PROMPT（提示词模板）
             将系统提示词 + 对话历史 + 用户输入 + Agent 中间步骤组合为完整 prompt
             模板变量: {chat_history} {input} {agent_scratchpad}

第75-112行 : create_agent(temperature, max_iterations) 函数
             ├─ 调用 get_llm(temperature) 获取 LLM 实例
             ├─ 调用 create_react_agent 创建 ReAct Agent
             │   └─ 注入 6 个工具 + ReAct Prompt
             ├─ 用 AgentExecutor 封装:
             │   ├─ max_iterations=8: 最多 8 轮 Thought-Action 循环
             │   ├─ handle_parsing_errors=True: 输出格式异常自动重试
             │   ├─ return_intermediate_steps=True: 返回思考链
             │   └─ max_execution_time=120: 120 秒超时
             └─ 返回配置好的 executor

第117-165行: run_agent(query, chat_history) 函数 【核心入口】
             ┌─ 返回固定结构:
             │   {output, intermediate_steps, has_error, error_msg}
             ├─ 创建 Agent → 构建 invoke_args
             │   ├─ "input": 用户查询
             │   └─ "chat_history": 对话历史（最近10轮的 LangChain 消息列表）
             ├─ 调用 executor.invoke()
             └─ 异常捕获 → 三层降级:

第166-196行: 三层异常降级逻辑
             ├─ Layer 1: timeout / timed out → API 超时
             │   └─ 降级方案: 直接 LLM 回答 + 功能引导
             ├─ Layer 2: token / context_length → Token 超限
             │   └─ 降级方案: 截断后直接回答 + 功能引导
             └─ Layer 3: 其他所有异常 → 调用失败
                 └─ 降级方案: 基础对话模式 + 6 个功能入口引导

第199-232行: _fallback_direct_answer(query, prefix) 函数
             降级兜底方案:
             ├─ 用 LLM 直接回答用户问题（不经过 Agent 工具链）
             ├─ 末尾附加 6 个侧边栏/标签页的快速功能入口
             └─ 如果 LLM 也挂了 → 返回硬编码的故障提示
```

**关键设计决策**:
- 为什么用 `max_iterations=8`？防止 Agent 陷入无限工具调用循环
- 为什么用 `max_execution_time=120`？超时兜底，防止单个请求占满资源
- 为什么 chat_history 限制最近10轮？防止 Token 超限，同时保证足够的上下文

---

### 📄 `modules/rag_knowledge.py` (282 行)

**整体作用**: RAG 知识库全链路实现。把 data/ 目录下的文档转成 FAISS 向量库，支持语义检索。

#### 逐段详解：

```
第1-18行   : 文档字符串 + imports
             ├─ TextLoader: 加载 .txt / .md 文本文件
             ├─ PyPDFLoader: 加载 PDF 文件
             ├─ FAISS: Facebook 开源的轻量级向量数据库
             ├─ RecursiveCharacterTextSplitter: 语义级文本分割器
             └─ Document: LangChain 文档对象

第23-84行  : load_documents(data_dir) 函数
             递归扫描目录，按扩展名分发加载器:
             ├─ .txt / .md → TextLoader (autodetect_encoding=True)
             ├─ .pdf → PyPDFLoader
             ├─ 每个 Document 的 metadata["source"] 记录来源文件路径
             ├─ 单个文件加载失败不阻塞整体（打印警告继续）
             └─ 找不到任何文档时抛出 ValueError

第89-110行 : split_documents(documents, chunk_size=500, chunk_overlap=50)
             语义分割策略:
             ├─ chunk_size=500: 每段约 500 字符（适配 JD/面经的短文本特性）
             ├─ chunk_overlap=50: 相邻片段重叠 50 字符（10%），保证跨片段语义连贯
             └─ separators: ["\n\n", "\n", "。", ".", " ", ""]
                优先级从高到低，优先在段落/句子边界切分

第115-137行: build_vector_store(documents, save_path)
             ├─ 调用 get_embedding() 获取 Embedding 模型
             ├─ FAISS.from_documents() 构建向量索引
             └─ save_local() 持久化到磁盘（data/vector_store/jd_store/）

第142-162行: load_vector_store(save_path)
             ├─ 检查向量库文件是否存在
             ├─ FAISS.load_local() 加载 + 设置 allow_dangerous_deserialization=True
             │   （安全说明: 因为是自己生成的向量库，不是从不可信来源加载）
             └─ 失败时抛出 RuntimeError

第167-200行: search_knowledge(vector_store, query, top_k=3)
             ├─ similarity_search_with_score() 执行相似度检索
             ├─ 返回: [{content, source, score}, ...]
             │   └─ score 是 L2 距离，越小越相似
             └─ 检索异常时返回错误信息而非抛出异常

第205-249行: build_knowledge_base(data_dir, vector_store_path)
             一站式构建入口（Streamlit 侧边栏按钮调用）:
             ├─ Step 1: load_documents() — 加载
             ├─ Step 2: split_documents() — 分割
             ├─ Step 3: build_vector_store() — 向量化+存储
             └─ 每步有独立的 try/except + 日志输出
```

---

### 📄 `modules/tools.py` (669 行)

**整体作用**: 6 个 Agent 工具函数的实现。这是项目最复杂的模块，每个工具都是一个独立的 LangChain Tool。

#### 全局说明：

```
第1-17行   : 文档字符串 — 6 个工具清单

第18-28行  : imports
             ├─ json: LLM 返回的 JSON 解析
             ├─ pandas: CSV 投递数据读写
             ├─ LangChain Tool: 工具封装基类
             ├─ pypdf: PDF 文本提取
             └─ get_llm, APPLICATIONS_CSV: 从 settings 导入
             └─ search_knowledge: 从 rag_knowledge 导入

第229-254行: _extract_json(text) — JSON 提取器（被工具3、工具6共用）
             三步兜底策略:
             ├─ 尝试1: json.loads() 直接解析
             ├─ 尝试2: 大括号计数法查找完整 JSON（解决 LLM 前后加文字的问题）
             └─ 尝试3: 返回 {"raw_output": text, "parse_error": True}

第575-596行: _smart_truncate(text, max_chars) — 智能截断
             策略: 保留开头75% + 结尾25%
             原因: JD 的"任职要求"和"加分项"通常在文档后半段
```

---

#### 🔧 工具 1: job_search_tool — 岗位检索

```
第33-69行  : _job_search(query)
             ┌─ 从 data/vector_store/jd_store/ 加载 FAISS 向量库
             ├─ search_knowledge(query, top_k=3) 语义检索
             └─ 格式化输出: 岗位名 / 来源文件 / 相似度得分 / 内容片段

第71-82行  : job_search_tool = Tool(...)
             ├─ name="job_search": Agent 通过此名称调用
             ├─ func=_job_search: 绑定的函数
             └─ description: 告诉 Agent 何时使用此工具、输入应是什么格式
```

#### 🔧 工具 2: resume_parse_tool — 简历解析

```
第88-93行  : RESUME_PARSE_PROMPT — LLM 结构化提取提示词
             让 LLM 从简历原文中提取:
             ├─ name: 姓名
             ├─ education: [{school, degree, major, year}]
             ├─ skills: [技能列表]
             ├─ experience: [{company, role, duration, description}]
             └─ projects: [{name, description, tech_stack}]

第96-210行 : _resume_parse(file_path) — LLM 驱动的简历解析
             三步流程:
             ├─ Step 1: pypdf 逐页提取原始文本
             ├─ Step 2: LLM 分析文本 → 输出 JSON
             │   └─ 失败时降级为纯文本展示
             ├─ Step 3: 解析 JSON → Markdown 结构化展示
             │   ├─ 姓名 / 教育背景 / 技能栈 / 工作经历 / 项目经历
             │   └─ 每个字段未识别时显示"未识别"
             └─ 末尾附原文前 2000 字符供人工核对
```

#### 🔧 工具 3: resume_match_tool — 简历匹配评分

```
第255-275行: MATCH_SCORE_PROMPT — 评分提示词
             要求 LLM 返回严格 JSON:
             {score, core_match, missing_skills, improvement}

第278-318行: _resume_match(jd_text) — 匹配评分
             ├─ 从输入中分离 JD 和简历（按 "简历:::" 分隔符）
             ├─ 智能截断到 3000 字符
             ├─ LLM 评估 → JSON 解析
             └─ 格式化为百分制得分 + 匹配点 + 缺失技能 + 改进方向
```

#### 🔧 工具 4: resume_optimize_tool — STAR 简历优化

```
第352-370行: STAR_OPTIMIZE_PROMPT — STAR 优化提示词
             ├─ 定义 STAR 法则: Situation / Task / Action / Result
             ├─ 附 Few-Shot 示例（优化前 vs 优化后）
             └─ 要求输出优化后文案 + 对比表格

第373-409行: _resume_optimize(input_text)
             ├─ 分离 JD 和原始经历
             ├─ 智能截断到 2500 字符
             └─ LLM 生成 STAR 改写
```

#### 🔧 工具 5: interview_question_tool — 面试题生成

```
第414-457行: INTERVIEW_QUESTION_PROMPT — 面试题生成提示词
             包含三类题目: 技术题 / 项目深挖题 / HR 行为题

第460-498行: _interview_question(target)
             ├─ 先从 RAG 面经知识库检索参考材料（top_k=5）
             ├─ 如果知识库未就绪，跳过参考直接用 LLM 生成
             └─ 生成三类题目，每道含题目+考察点+答题思路
```

#### 🔧 工具 6: application_tracker_tool — 投递管理

```
第565-659行: _application_tracker(command) — 纯规则匹配的 CSV 增删改查
             支持 4 种指令:
             ├─ "查询" / "查看": 读取 CSV 并格式化输出
             ├─ "新增 公司 岗位 日期 状态 [备注]": append 新行
             ├─ "更新 序号 新状态": 修改指定行的状态列
             ├─ "删除 序号": 删除指定行
             └─ 否则: 返回指令格式说明
             
             注意: 此工具供 Agent 通过文本指令调用
             同时 main.py 的 Tab4 也直接调用 _application_tracker 
             （用于快捷指令输入框）
```

#### 🔧 工具集合导出

```
第662-668行: ALL_TOOLS — 6 个工具列表
             按固定顺序导出给 agent_core.py 的 create_react_agent()
             顺序会影响 Agent 在提示词中看到的工具列表排列
```

---
### 📄 `main.py` (555 行)

**整体作用**: Streamlit Web UI 主入口。定义了完整的页面布局和用户交互逻辑。

#### 逐段详解：

```
第1-11行   : 文档字符串 — 页面结构描述
             4 个 Tab: 智能对话助手 / 简历优化 / 模拟面试 / 投递管理

第12-15行  : 标准库 imports
             os, sys, tempfile, Path

第17行     : streamlit as st — Web UI 框架

第23-28行  : st.set_page_config()
             ├─ page_title: "实习求职智能助手 Agent"
             ├─ page_icon: "🎯"
             ├─ layout="wide": 宽屏布局
             └─ initial_sidebar_state="expanded": 侧边栏默认展开

第32-34行  : sys.path 处理
             将项目根目录加入 Python 路径，确保模块导入正常

第35行     : from langchain.schema import HumanMessage, AIMessage
             用于构建多轮对话历史

第37-50行  : 项目模块 imports
             ├─ config.settings: 全局配置
             ├─ modules.rag_knowledge: RAG 知识库
             ├─ modules.agent_core: ReAct Agent
             └─ modules.tools: 6 个工具函数（Tab2/Tab3/Tab4 直接调用）

第53-68行  : 自定义 CSS 样式
             深色/浅色模式通用，主要美化 tool-result 和 Expander

第73-90行  : init_session_state() 函数
             初始化所有 Streamlit session_state 变量:
             ├─ chat_history: [] — 对话历史
             ├─ vector_store: None — 向量库缓存
             ├─ agent_executor: None — Agent 缓存（预留）
             ├─ resume_text: "" — 已解析的简历文本
             ├─ resume_path: "" — 上传的简历路径
             ├─ interview_active: False — 面试进行中标志
             └─ interview_questions: "" — 当前面试题

第96行     : init_session_state() 调用 — 每次页面刷新时执行

第100-203行: 侧边栏（Sidebar）
             ├─ ⚙️ 配置面板
             │   ├─ st.text_input: API Key（password类型，隐藏输入）
             │   ├─ st.text_input: Base URL（默认 OpenAI，可改为 DeepSeek）
             │   ├─ st.text_input: LLM 模型名
             │   ├─ st.text_input: Embedding 模型名（disabled只读）
             │   └─ st.button("应用配置"): 更新配置 + 重置向量库缓存
             │
             ├─ 📚 知识库管理
             │   ├─ st.button("构建知识库"):
             │   │   调用 build_knowledge_base() → 加载→分割→向量化→保存
             │   └─ st.button("加载知识库"):
             │       调用 load_vector_store() → 加载已保存的向量库
             │
             └─ 📖 使用说明
                 └─ st.markdown: 快速开始步骤 + 4大功能模块简介

第205-209行: 主区域标题
             ├─ st.markdown: "🎯 实习求职智能助手 Agent"
             └─ st.caption: 技术栈简介

第216-220行: 4 个 Tab 页
             st.tabs(["💬 智能对话助手", "📄 简历优化", "🎤 模拟面试", "📌 投递管理"])

第226-286行: Tab 1 — 💬 智能对话助手
             ├─ 对话历史容器 (height=450px)
             │   ├─ 无历史时: 显示欢迎信息和功能介绍
             │   └─ 有历史时: 逐条渲染
             │       ├─ st.chat_message("user"): 用户消息
             │       └─ st.chat_message("assistant"): 助手消息
             │           └─ st.expander("查看思考过程"): 工具调用详情
             │               ├─ 工具名 (action.tool)
             │               ├─ 输入参数 (action.tool_input)
             │               └─ 输出结果 (observation)
             │
             ├─ st.chat_input("输入你的求职问题..."): 用户输入框
             │   └─ 发送后:
             │       ├─ 构建 LangChain 格式对话历史 (最近10轮=20条消息)
             │       │   ├─ HumanMessage: 用户消息
             │       │   └─ AIMessage: 助手消息
             │       ├─ 调用 run_agent(user_input, chat_history=lc_history)
             │       ├─ 展示回答 + 思考链
             │       └─ st.rerun() 刷新页面
             │
             └─ st.button("清空对话"): 重置 chat_history

第292-348行: Tab 2 — 📄 简历优化
             ├─ 左列: 📤 上传简历
             │   ├─ st.file_uploader: 选择 PDF
             │   ├─ 保存到 data/resume/
             │   ├─ 调用 _resume_parse() 解析（pypdf → LLM 结构化）
             │   └─ 展示解析结果预览（前800字符）
             │
             └─ 右列: 📋 输入目标 JD
                 ├─ st.text_area: 粘贴 JD
                 ├─ st.button("匹配度评分"):
                 │   └─ 调用 _resume_match() → LLM 打分
                 └─ st.button("STAR 优化经历"):
                     └─ 调用 _resume_optimize() → LLM 改写

第354-418行: Tab 3 — 🎤 模拟面试
             ├─ st.text_input: 目标公司和岗位
             ├─ st.button("生成面试题"):
             │   └─ 调用 _interview_question(target_job)
             │       ├─ RAG 检索面经知识库
             │       └─ LLM 生成技术/项目/HR 三类题
             │
             ├─ 面试题展示: st.markdown
             ├─ st.text_area("写下你的回答..."):
             │   └─ st.button("获取回答反馈"):
             │       └─ LLM 从结构/深度/改进三维度评价
             │
             └─ st.button("重置"): 清空面试状态

第424-554行: Tab 4 — 📌 投递管理 【升级版】
             ├─ 📊 统计看板 (5个 metric 卡片)
             │   ├─ 总投递数
             │   ├─ 进行中（已投递/初筛中/笔试/面试中）
             │   ├─ Offer 数
             │   ├─ 已拒数
             │   └─ Offer 率
             │
             ├─ 📊 可视化图表
             │   ├─ st.bar_chart: 状态分布柱状图
             │   └─ st.bar_chart: 投递公司 Top 5 柱状图
             │
             ├─ 📋 交互式表格
             │   └─ st.data_editor:
             │       ├─ num_rows="dynamic": 支持动态增删行
             │       ├─ 状态列: SelectboxColumn (下拉选择)
             │       ├─ 序号列: 只读（disabled=True）
             │       └─ 用户可直接在单元格内编辑
             │
             ├─ 💾 st.button("保存修改"): 写入 CSV
             ├─ 📥 st.download_button("导出 CSV"): 下载投递记录
             └─ 🚀 快捷指令输入框:
                 └─ 调用 _application_tracker("新增/查询/更新/删除 ...")
```

---

## 6. 快速启动指南

### 前置要求

- Python 3.10+
- Git（可选）
- 一个 OpenAI 兼容的 API Key（推荐 DeepSeek，便宜好用）

### 安装步骤

```bash
# 1. 进入项目目录
cd jobsearch-agent

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 安装本地 Embedding 模型依赖（推荐，零费用）
pip install sentence-transformers

# 4. 配置 API Key
# 编辑 .env 文件（如果不存在，复制 .env.example 为 .env）
# 填入你的真实 API Key:

# OPENAI_API_KEY=sk-your-real-key-here
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# LLM_MODEL_NAME=deepseek-chat

# 5. 启动应用
streamlit run main.py

# 6. 浏览器打开 http://localhost:8501
```

### 首次使用流程

```
1. 侧边栏 "API 配置" → 确认 API Key 正确
2. 侧边栏 "知识库管理" → 点击 "🔨 构建知识库"
   （等待约 30 秒，首次需下载 Embedding 模型 ~100MB）
3. 看到 "✅ 知识库构建成功" 提示
4. 切换到 "💬 智能对话助手" Tab
5. 输入 "帮我找后端开发的实习岗位" 开始使用！
```

### 如果只用 OpenAI 而非 DeepSeek

修改 `.env` 中的两行：
```
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-3.5-turbo
```

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| numpy 版本冲突 | `pip install "numpy<2.0,>=1.26"` |
| 找不到 sentence-transformers | `pip install sentence-transformers` |
| 知识库构建失败 | 检查网络能否访问 huggingface.co（国内可能需代理） |
| .env 被 git 追踪 | `git rm --cached .env` 然后确认 .gitignore 包含 .env |

---

## 7. 技术亮点（简历提炼版）

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **ReAct Agent 架构** | 基于 LangChain 原生 `create_react_agent`，自定义中文提示词，6 工具协同，`max_iterations=8` 防死循环 |
| 2 | **RAG 全链路** | 文档加载（TextLoader/PyPDFLoader）→ 语义分割（chunk=500, overlap=50）→ FAISS 向量化 → 本地持久化 → 相似度检索 |
| 3 | **LLM 参数化缓存** | 按 `(temperature, max_tokens)` 维度缓存实例，不同场景独立参数互不干扰 |
| 4 | **多轮对话记忆** | ReAct prompt 注入最近 10 轮对话历史，Agent 具备上下文感知能力 |
| 5 | **三层异常降级** | API 超时 / Token 超限 / 调用失败分别捕获，自动切换为基础对话 + 功能引导，服务零中断 |
| 6 | **JSON 结构化校验** | 内置大括号计数法提取器，解决 LLM 在 JSON 前后加额外文字导致解析失败的痛点 |
| 7 | **LLM 驱动简历解析** | pypdf 提取原文 → LLM 输出 JSON（姓名/教育/技能/经历/项目），替代简单关键词匹配 |
| 8 | **三级 Embedding 降级** | 本地 BAAI/bge-small-zh-v1.5（首选）→ OpenAI 兼容 API（备选）→ 明确报错 + 方案提示 |
| 9 | **交互式投递管理** | Streamlit `data_editor` 单元格直接编辑 + 动态增删行 + 5 指标统计看板 + 可视化图表 |
| 10 | **统一日志系统** | `TimedRotatingFileHandler` 按天轮转保留 7 天，所有模块集成结构化日志 |
| 11 | **安全加固** | `.gitignore` 保护 `.env` + 启动时自动检测 git 追踪状态并告警 |

---

## 8. 优化记录

本次优化（2026-07-14）解决了原项目的 10 个问题：

| # | 问题 | 解决方式 |
|---|------|---------|
| 1 | 知识库文档太少（仅 4 个） | 新增 4 个 JD + 2 份面经，扩充至 10 个文档 |
| 2 | 无多轮对话记忆 | ReAct prompt 注入 `{chat_history}`，main.py 构建最近 10 轮 LangChain 历史 |
| 3 | 每次重建 Agent | —（Agent 创建开销很小，核心瓶颈在 LLM API，通过 #4 解决） |
| 4 | LLM 单例 temperature 冲突 | 改为参数化缓存 `_llm_cache: dict`，按 `(temperature, max_tokens)` 键存储 |
| 5 | JD/简历截断一刀切 | `_smart_truncate()`：保留开头 75% + 结尾 25% |
| 6 | 简历解析用关键词匹配 | 升级为 LLM 结构化提取：pypdf → LLM JSON → Markdown 展示 |
| 7 | 投递管理不交互 | 升级为 `st.data_editor` + `num_rows="dynamic"` |
| 8 | 无统计可视化 | 新增 5 个 `st.metric` 卡片 + 2 个 `st.bar_chart` |
| 9 | 散落的 print() 无日志 | 新建 `config/logger.py`，所有模块集成 |
| 10 | .env 安全风险 | 创建 `.gitignore` + 添加 `_check_env_safety()` 启动检查 |

---

## 📁 完整项目结构

```
jobsearch-agent/
│
├── main.py                         # Streamlit 主入口 (555行)
├── requirements.txt                # Python 依赖清单
├── .env.example                    # 环境变量模板（可提交git）
├── .env                            # 真实环境变量（.gitignore已排除）
├── .gitignore                      # Git 忽略规则
├── README.md                       # 📖 本文档
├── INTERVIEW_PREP.md               # 面试准备指南
├── INTERVIEW_PREP_V2.md            # 面试准备指南（补充版）
├── TROUBLESHOOTING.md              # 问题排障记录
│
├── config/                         # ⚙️ 配置层
│   ├── __init__.py                 #   包标识
│   ├── settings.py                 #   全局配置核心 (247行)
│   └── logger.py                   #   统一日志系统 (102行)
│
├── modules/                        # 🧠 核心逻辑层
│   ├── __init__.py                 #   包标识
│   ├── agent_core.py               #   ReAct Agent 核心 (246行)
│   ├── rag_knowledge.py            #   RAG 知识库全链路 (282行)
│   └── tools.py                    #   6个Agent工具函数 (669行)
│
├── data/                           # 📊 数据层
│   ├── jd_samples/                 #   6个示例岗位JD
│   │   ├── jd_data_analyst_internet.txt
│   │   ├── jd_business_analyst_ecommerce.txt
│   │   ├── jd_backend_dev.txt
│   │   ├── jd_algorithm_intern.txt
│   │   ├── jd_frontend_dev.txt
│   │   └── jd_product_intern.txt
│   ├── interview/                  #   4份面经资料
│   │   ├── interview_da_tech1.txt
│   │   ├── interview_da_biz2.txt
│   │   ├── interview_backend_tech.txt
│   │   └── interview_frontend_tech.txt
│   ├── resume/                     #   简历上传临时目录
│   ├── vector_store/               #   FAISS向量库（构建后生成）
│   └── applications.csv           #   投递记录（运行时生成）
│
└── logs/                           # 📝 应用日志（运行时生成）
    └── app.log                     #   按天轮转，保留7天
```