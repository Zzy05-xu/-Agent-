# 🎯 实习求职智能助手 Agent — 完整项目文档

> **基于 LangChain ReAct Agent + RAG 知识库 + Streamlit 的一站式实习求职 AI 平台**
>
> 技术栈：Python 3.10+ · LangChain 0.2 · FAISS · DeepSeek/OpenAI · HuggingFace BGE · Streamlit 1.36
> 最后更新：2026-07-15

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构图](#2-技术架构图)
3. [完整运行流程](#3-完整运行流程)
4. [项目文件逐一详解](#4-项目文件逐一详解)
   - [4.1 根目录文件](#41-根目录文件)
   - [4.2 config/ — 配置层](#42-config--配置层)
   - [4.3 modules/ — 核心逻辑层](#43-modules--核心逻辑层)
   - [4.4 data/ — 数据层](#44-data--数据层)
5. [每个代码文件逐行分析](#5-每个代码文件逐行分析)
   - [5.1 config/settings.py](#51-configsettingspy)
   - [5.2 config/logger.py](#52-configloggerpy)
   - [5.3 modules/rag_knowledge.py](#53-modulesrag_knowledgepy)
   - [5.4 modules/tools.py](#54-modulestoolspy)
   - [5.5 modules/agent_core.py](#55-modulesagent_corepy)
   - [5.6 main.py](#56-mainpy)
6. [快速启动指南](#6-快速启动指南)
7. [技术亮点（简历提炼版）](#7-技术亮点简历提炼版)
8. [优化记录](#8-优化记录)
9. [完整项目结构](#9-完整项目结构)

---

## 1. 项目概述

### 做什么的？

这是一个面向**大学生实习求职**全流程的 AI 助手。用户通过浏览器打开一个 Web 页面，就能用自然对话的方式完成：

| 功能 | 说明 |
|------|------|
| 🔍 **岗位检索** | 从本地知识库中语义搜索匹配的实习岗位（FAISS 粗筛 + LLM 重排序，支持数据分析/后端/前端/算法/产品/商业分析 6 个方向） |
| 📄 **简历优化** | 上传 PDF 简历 → LLM 自动提取结构化信息 → 与目标 JD 匹配打分 → 用 STAR 法则改写经历 |
| 🎤 **模拟面试** | RAG 检索面经知识库 → LLM 生成技术/项目/HR 三类面试题 → 用户回答后获取面试官风格多维度反馈 |
| 📌 **投递管理** | 可视化表格记录投递进度，支持表单快捷添加 + 单元格直接编辑 + 5 指标统计看板 + 柱状图 + CSV 导出 |

### 怎么实现的？

底层是一个 **ReAct Agent**（"思考→行动→观察→回答"循环），它持有 6 个工具函数，能自主决定调用哪个工具、传什么参数。同时集成了 **RAG（检索增强生成）** 知识库：所有 JD 和面经文档被转成向量存到本地 FAISS 索引，Agent 可以用语义搜索找到最相关信息。

**优化版新增能力：**
- ⚡ **流式输出**：实时展示 Agent 思考过程和工具调用
- 🔄 **自动重试**：LLM 调用失败自动指数退避重试 3 次
- 🎯 **AI 重排序**：FAISS 粗筛 15 条 → LLM 精选 Top 3
- 📥 **增量更新**：新增文档无需全量重建

---

## 2. 技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    🖥️  Streamlit Web UI                      │
│  main.py (653行) — 4个Tab页 + 侧边栏 + Session State        │
│  流式输出 Toggle · 配置持久化 · 表单式投递 · 知识库统计      │
├─────────────────────────────────────────────────────────────┤
│               🧠 Agent Core (agent_core.py)                  │
│  ReAct 范式: Thought → Action → Observation → Final Answer │
│  流式生成器 · 会话级缓存 · 三层异常降级                      │
├──────────────────┬──────────────────────────────────────────┤
│   🔧 6个Tool      │        📚 RAG 知识库 (rag_knowledge.py)  │
│   (tools.py)      │                                          │
│                   │  文档加载 → 智能分割(JD1500/面经800)      │
│ 1.岗位检索(含重排序)│  → FAISS向量化 → 增量更新               │
│ 2.简历解析(LLM提取) │                                          │
│ 3.简历匹配评分      │  Embedding: BAAI/bge-small-zh-v1.5      │
│ 4.STAR简历优化      │  (本地CPU运行，零费用)                   │
│ 5.面试题生成        │                                          │
│ 6.投递管理          │  LLM Re-Ranking 重排序                  │
├──────────────────┴──────────────────────────────────────────┤
│               ⚙️ 配置层 (config/)                            │
│  settings.py: API配置 + LLM参数化缓存 + Embedding三级降级    │
│              + invoke_llm_with_retry 自动重试                 │
│              + session_state 配置持久化                       │
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
      ├─ split_documents(): JD→chunk_size=1500, 面经→chunk_size=800
      │   自动检测文档类型，保证 JD 信息完整性
      └─ build_vector_store(): FAISS 向量化 + 保存到 data/vector_store/

Step 3: 用户在 Tab1 输入"帮我找后端开发的实习岗位"
  └─ main.py 构建 LangChain 对话历史（最近10轮）
  └─ 调用 modules/agent_core.py → run_agent() 或 run_agent_streaming()
  └─ Agent 内部循环（最多8轮）:
      ├─ Thought: Agent分析用户意图 → "需要调用 job_search 工具"
      ├─ Action: job_search
      ├─ Action Input: "后端开发 实习"
      ├─ Observation: FAISS 粗筛 15 条 → LLM Re-Ranking 精选 Top 3
      ├─ Thought: "信息足够了，可以整理回答"
      └─ Final Answer: 格式化展示岗位信息
  └─ main.py 展示回答 + 可展开的思考链（流式模式下实时展示）

Step 4: 用户在 Tab2 上传简历 PDF
  └─ main.py 保存到 data/resume/
  └─ 调用 tools.py → _resume_parse()
  └─ 子步骤:
      ├─ pypdf 逐页提取原始文本
      ├─ LLM（带自动重试）分析文本，返回 JSON {name, education, skills, experience, projects}
      └─ 格式化为 Markdown 展示
  └─ 用户粘贴 JD → 点击"匹配度评分"
      └─ _resume_match() → LLM（带重试）返回 {score, core_match, missing_skills, improvement}
  └─ 用户点击"STAR优化"
      └─ _resume_optimize() → LLM（带重试）用 STAR 法则改写经历

Step 5: 用户在 Tab3 输入目标岗位 → 点击"生成面试题"
  └─ 调用 tools.py → _interview_question()
  └─ 先从 RAG 知识库检索面经作为参考材料
  └─ LLM（带重试）生成三类题目（技术/项目/HR）
  └─ 用户写下回答 → LLM 给出多维度反馈

Step 6: 用户在 Tab4 管理投递
  └─ st.form 快捷添加表单：填写公司/岗位/状态/日期/备注
  └─ st.data_editor 渲染可编辑表格（单元格内直接修改）
  └─ 新增/删除行 → 点击"保存修改" → 写入 data/applications.csv
  └─ 统计看板实时展示: 总投递数/进行中/Offer数/Offer率 + 状态柱状图 + 公司Top5

Step 7: 增量更新（用户新增 JD 文件后）
  └─ 侧边栏点击"增量更新"
  └─ 调用 add_documents_to_store() → 仅加载新文档 → 分割 → 追加到 FAISS 索引
  └─ 无需全量重建，节省时间
```

---

## 4. 项目文件逐一详解

### 4.1 根目录文件

| 文件 | 大小 | 作用 |
|------|------|------|
| `main.py` | 653行 / 28KB | **Streamlit 主入口**。定义了完整 Web UI 的页面结构：4个Tab页 + 侧边栏。负责 Session State 管理、流式输出控制、用户交互响应、模块调用编排 |
| `requirements.txt` | 37行 / 1KB | Python 依赖清单。锁定 langchain / faiss-cpu / streamlit / pypdf / pandas 版本号，tenacity 设为 >=8.0.0 兼容已安装版本 |
| `.env.example` | 23行 / 1.2KB | 环境变量模板。包含 API Key / Base URL / 模型名 / Embedding 配置的填写说明。**可安全提交到 git** |
| `.env` | 23行 / 1.2KB | 真实环境变量。包含实际的 API Key。**.gitignore 已排除，不可提交** |
| `.gitignore` | 432B | Git 忽略规则。排除 .env、__pycache__、vector_store、logs、虚拟环境等 |
| `README.md` | 本文档 | 项目完整分析文档 |
| `INTERVIEW_PREP_V2.md` | 117行 / 7.4KB | 面试准备深度手册。教用户如何在简历上描述这个项目 + 常见面试问题回答思路 |
| `TROUBLESHOOTING.md` | 218行 / 7.4KB | 问题排障记录。记录了开发过程中遇到的技术坑和解决方案 |

### 4.2 config/ — 配置层

| 文件 | 行数 | 作用 |
|------|------|------|
| `config/__init__.py` | 1行 | Python 包标识文件。使 config 目录可作为模块导入 |
| `config/settings.py` | 346行 / 12.3KB | **全局配置核心**。管理 API Key（含 session_state 持久化）、LLM 参数化缓存、Embedding 三级降级、`invoke_llm_with_retry` 自动重试函数 |
| `config/logger.py` | 102行 / 2.8KB | **统一日志系统**。TimedRotatingFileHandler 按天轮转，同时输出控制台和文件 |

### 4.3 modules/ — 核心逻辑层

| 文件 | 行数 | 作用 |
|------|------|------|
| `modules/__init__.py` | 1行 | Python 包标识文件 |
| `modules/agent_core.py` | 364行 / 14.8KB | **ReAct Agent 核心**。系统提示词、Agent 创建与缓存、标准调用 `run_agent`、流式调用 `run_agent_streaming`、三层异常降级 |
| `modules/rag_knowledge.py` | 155行 / 6.6KB | **RAG 知识库全链路**。文档加载→智能分割（JD 1500/面经 800）→FAISS 向量化→增量更新→语义检索 |
| `modules/tools.py` | 682行 / 29.3KB | **6个 Agent 工具函数**。最复杂的模块，含增强 JSON 解析、LLM Re-Ranking 重排序、岗位检索、简历解析、匹配评分、STAR优化、面试题生成、投递管理 |

### 4.4 data/ — 数据层

| 目录/文件 | 说明 |
|-----------|------|
| `data/jd_samples/` | 6 个示例岗位 JD（.txt格式） |
| `data/jd_samples/jd_data_analyst_internet.txt` | 数据分析实习（互联网）— 未来星球科技 |
| `data/jd_samples/jd_business_analyst_ecommerce.txt` | 商业分析实习 — 电商方向 |
| `data/jd_samples/jd_backend_dev.txt` | 后端开发实习（Java）— 星辰科技 |
| `data/jd_samples/jd_algorithm_intern.txt` | 算法实习（推荐系统）— 极光智能 |
| `data/jd_samples/jd_frontend_dev.txt` | 前端开发实习 — 蓝图科技 |
| `data/jd_samples/jd_product_intern.txt` | 产品实习 — 远航互娱 |
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


## 5. 每个代码文件逐行分析

---

### 📄 `config/__init__.py` (1行)

```python
# 空文件，仅标记 config 为 Python package
```

**作用**：让 Python 把 `config/` 目录识别为可导入的包，这样其他模块可以写 `from config.settings import ...`。

---

### 📄 `config/settings.py` (346行)

**整体作用**：全局配置中心，负责 6 件事：加载环境变量、管理 LLM/Embedding 实例、LLM 自动重试、session_state 配置持久化、管理数据目录路径、运行时安全检测。

#### 逐段详解：

```
第1-18行   : 文档字符串 + imports
             ├─ load_dotenv: 从 .env 文件加载环境变量
             ├─ ChatOpenAI: LangChain 封装的标准 LLM 接口
             ├─ OpenAIEmbeddings: OpenAI 兼容的 Embedding 接口
             └─ tenacity: 可选，提供指数退避重试（未安装时优雅降级）

第24-36行  : 环境变量加载
             ├─ 优先从 PROJECT_ROOT/.env 加载
             └─ 找不到则调用 load_dotenv() 搜索默认路径

第40-62行  : 配置项读取
             ├─ OPENAI_API_KEY: API 密钥（必填）
             ├─ OPENAI_BASE_URL: API 地址（默认 OpenAI，可改为 DeepSeek）
             ├─ LLM_MODEL_NAME: 模型名（默认 gpt-3.5-turbo）
             ├─ LOCAL_EMBEDDING_MODEL: 本地 Embedding 模型名
             ├─ EMBEDDING_API_KEY / EMBEDDING_API_BASE: Embedding API 备选
             ├─ EMBEDDING_MODEL_NAME: 在线 Embedding 模型名
             └─ LLM_MAX_RETRIES / LLM_RETRY_MIN_WAIT / LLM_RETRY_MAX_WAIT: 重试配置

第68-79行  : 数据目录路径定义
             ├─ DATA_DIR → data/
             ├─ JD_SAMPLES_DIR → data/jd_samples/
             ├─ INTERVIEW_DIR → data/interview/
             ├─ RESUME_DIR → data/resume/
             ├─ VECTOR_STORE_DIR → data/vector_store/
             └─ APPLICATIONS_CSV → data/applications.csv

第82-92行  : ensure_data_dirs() 函数
             自动创建所有数据子目录和空的 applications.csv

第95-112行 : _check_env_safety() 函数
             启动时执行 git ls-files --error-unmatch .env 检查 .env 是否被追踪
             如是，输出安全警告（不阻塞运行）

第118-123行: 全局缓存变量
             ├─ _embedding_instance: 缓存的 Embedding 实例
             ├─ _embedding_mode: 当前模式 "local" | "api" | None
             └─ _llm_cache: dict — 按 (temperature, max_tokens, api_key_hash, base_url, model) 键缓存

第127-140行: _try_get_local_embedding() 函数
             尝试加载本地 HuggingFace Embedding 模型
             需要 pip install sentence-transformers
             模型: BAAI/bge-small-zh-v1.5（中文语义检索最优）
             首次运行自动下载约 100MB

第143-150行: _create_api_embedding() 函数
             创建 OpenAI 兼容的在线 Embedding 实例
             作为本地模型不可用时的备选方案

第153-176行: _get_active_api_key / _get_active_base_url / _get_active_llm_model
             优先从 st.session_state 读取配置，其次回退到环境变量
             实现配置持久化：侧边栏修改的 API Key 在页面刷新后不丢失

第179-217行: get_llm(temperature, max_tokens) 函数 【核心】
             ┌─ 参数化缓存机制
             ├─ 按 (temperature, max_tokens, api_key, base_url, model) 构建 key
             ├─ 如果缓存中存在，直接返回（避免重复创建）
             ├─ 否则创建新的 ChatOpenAI 实例存入缓存
             ├─ API Key 变化时自动失效旧缓存
             └─ 不同场景用不同 temperature:
                  · Agent 主调用: 0.2（平衡准确和灵活）
                  · 简历匹配评分: 0.1（追求一致性）
                  · STAR 简历优化: 0.4（需要创造性改写）
                  · 面试题生成: 0.5（需要题目多样性）

第220-254行: invoke_llm_with_retry() 函数 【核心新增】
             ┌─ LLM 调用自动重试封装
             ├─ 优先使用 tenacity 库的指数退避策略
             │   └─ 等待时间: 1s → 2s → 4s → 抛出异常
             ├─ tenacity 未安装时降级为简单循环重试
             └─ 所有 tools.py 的 LLM 调用均通过此函数

第257-304行: get_embedding() 函数 【核心】
             ┌─ 三级自动降级策略
             ├─ Level 1: 本地 BAAI/bge-small-zh-v1.5（零费用，推荐）
             ├─ Level 2: OpenAI 兼容 Embedding API（需配置 Key）
             └─ Level 3: 抛出 RuntimeError + 解决方案提示
             首次成功即缓存 _embedding_instance，后续直接返回

第307-309行: get_embedding_mode() — 返回当前模式字符串

第312-316行: reset_instances() — 清空所有缓存（切换 API 配置时调用）

第319-338行: update_api_config() — 动态配置更新
             ├─ 更新全局变量
             ├─ 同步写入 st.session_state（实现持久化）
             └─ 清空所有缓存实例
```

---

### 📄 `config/logger.py` (102行)

**整体作用**：统一日志系统，替代代码中散落的 `print()` 语句。

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
             ├─ 配置 Console Handler: 输出到 stdout
             ├─ 配置 File Handler: TimedRotatingFileHandler
             │   └─ when="midnight": 每天午夜轮转
             │   └─ backupCount=7: 保留最近 7 天日志
             ├─ 格式: [2026-07-15 10:30:00] [INFO   ] [模块名] 消息内容
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

### 📄 `modules/rag_knowledge.py` (155行)

**整体作用**：RAG 知识库全链路实现。把 data/ 目录下的文档转成 FAISS 向量库，支持语义检索和增量更新。

#### 逐段详解：

```
第1-27行   : 文档字符串 + imports
             ├─ TextLoader: 加载 .txt / .md 文本文件
             ├─ PyPDFLoader: 加载 PDF 文件
             ├─ FAISS: Facebook 开源的轻量级向量数据库
             ├─ RecursiveCharacterTextSplitter: 语义级文本分割器
             └─ Document: LangChain 文档对象

第30-72行  : load_documents(data_dir) 函数
             递归扫描目录，按扩展名分发加载器:
             ├─ .txt / .md → TextLoader (encoding="utf-8")
             ├─ .pdf → PyPDFLoader
             ├─ 每个 Document 的 metadata["source"] 记录来源文件路径
             ├─ 单个文件加载失败不阻塞整体（打印警告继续）
             └─ 找不到任何文档时抛出 ValueError

第75-80行  : _detect_doc_type(doc) 函数
             根据 metadata["source"] 路径判断文档类型:
             ├─ 路径含 "interview" → "interview"（面经）
             └─ 其他 → "jd"（岗位JD）

第83-114行 : split_documents() 函数 【优化点】
             按文档类型分别分割:
             ├─ JD 文档: chunk_size=1500, chunk_overlap=200
             │   └─ 保证一个 JD（2000-3000字）最多分 2-3 块，信息完整
             ├─ 面经文档: chunk_size=800, chunk_overlap=100
             │   └─ 面经结构松散，适度细粒度
             ├─ 分隔符优先级: \n\n → \n → 。 → . → 空格 → 字符
             └─ 手动指定 chunk_size 时覆盖自动判断

第117-135行: build_vector_store() 函数
             ├─ 调用 get_embedding() 获取 Embedding 模型
             ├─ FAISS.from_documents() 构建向量索引
             └─ save_local() 持久化到磁盘

第138-152行: load_vector_store() 函数
             ├─ 检查向量库文件是否存在
             ├─ FAISS.load_local() 加载（allow_dangerous_deserialization=True）
             └─ 失败时抛出 RuntimeError

第155-176行: search_knowledge() 函数
             ├─ similarity_search_with_score() 执行相似度检索
             ├─ 支持自定义 top_k（供 Re-Ranking 使用，可设 15）
             ├─ 返回: [{content, source, score}, ...]
             │   └─ score 是 L2 距离，越小越相似
             └─ 检索异常时返回错误信息而非抛出异常

第179-210行: add_documents_to_store() 函数 【新增】
             增量添加新文档到已有向量库:
             ├─ 加载现有向量库
             ├─ 加载新文档 → 分割 → vector_store.add_documents()
             └─ save_local() 持久化

第213-248行: build_knowledge_base() 函数
             一站式构建入口（Streamlit 侧边栏按钮调用）:
             ├─ Step 1: load_documents()
             ├─ Step 2: split_documents()（自动判断 JD/面经）
             ├─ Step 3: build_vector_store()
             └─ 每步有独立的 try/except + 日志输出
```

---


### 📄 `modules/tools.py` (682行)

**整体作用**：6 个 Agent 工具函数的实现。这是项目最复杂的模块，每个工具都是一个独立的 LangChain Tool。优化版新增增强 JSON 解析和 LLM Re-Ranking 重排序。

#### 全局说明：

```
第1-25行   : 文档字符串 — 6 个工具清单 + 优化点说明

第26-33行  : imports
             ├─ json: LLM 返回的 JSON 解析
             ├─ re: 正则表达式（增强 JSON 提取）
             ├─ pandas: CSV 投递数据读写
             ├─ LangChain Tool: 工具封装基类
             ├─ pypdf: PDF 文本提取
             ├─ invoke_llm_with_retry: 从 settings 导入的自动重试函数
             └─ search_knowledge: 从 rag_knowledge 导入

第39-90行  : _extract_json(text) — 增强版 JSON 提取器 【优化点】
             四层兜底策略:
             ├─ 方法1: json.loads() 直接解析
             ├─ 方法2: 正则匹配 ```json``` 或 ``` 代码块
             ├─ 方法3: 从第一个 { 到最后一个 } 的范围提取
             ├─ 方法4: 大括号计数法查找完整 JSON
             └─ 全部失败: 返回 {"raw_output": text, "parse_error": True}

第96-170行 : _llm_rerank() — LLM Re-Ranking 重排序 【新增优化】
             ┌─ 解决 FAISS L2 距离对中文语义区分度有限的问题
             ├─ Step 1: 构建候选文档列表文本
             ├─ Step 2: LLM 为每个文档打分（1-10）
             ├─ Step 3: 按分数排序，取 top_k
             ├─ 使用 invoke_llm_with_retry 保证可靠性
             └─ 降级方案: 返回原始 FAISS 排序
```

---

#### 🔧 工具 1: job_search_tool — 岗位检索（含重排序）

```
第175-229行: _job_search(query)
             ┌─ 从 data/vector_store/jd_store/ 加载 FAISS 向量库
             ├─ Step 1: FAISS 粗筛 → top_k=15（扩大召回）
             ├─ Step 2: _llm_rerank() 重排序 → 精选 top 3
             ├─ 降级: 重排序失败时返回原始 FAISS 前 3
             └─ 格式化输出: 岗位名 / 来源文件 / 相似度得分 / 内容片段

第232-243行: job_search_tool = Tool(...)
             ├─ name="job_search": Agent 通过此名称调用
             └─ description: 告知 Agent 何时使用、输入格式
```

#### 🔧 工具 2: resume_parse_tool — 简历解析

```
第249-263行: RESUME_PARSE_PROMPT — LLM 结构化提取提示词
             让 LLM 从简历原文中提取:
             {name, education, skills, experience, projects}

第266-365行: _resume_parse(file_path)
             三步流程:
             ├─ Step 1: pypdf 逐页提取原始文本
             ├─ Step 2: invoke_llm_with_retry() 分析文本 → JSON
             ├─ Step 3: 解析 JSON → Markdown 结构化展示
             │   ├─ 姓名 / 教育背景 / 技能栈 / 工作经历 / 项目经历
             │   └─ 每个字段未识别时显示"未识别"
             └─ 末尾附原文前 2000 字符供人工核对
```

#### 🔧 工具 3: resume_match_tool — 简历匹配评分

```
第368-394行: MATCH_SCORE_PROMPT — 评分提示词
             要求 LLM 返回严格 JSON:
             {score, core_match, missing_skills, improvement}

第397-446行: _resume_match(jd_text)
             ├─ 从输入中分离 JD 和简历
             ├─ 智能截断到 3000 字符
             ├─ invoke_llm_with_retry() 评估 → JSON 解析
             └─ 格式化为百分制得分 + 匹配点 + 缺失技能 + 改进方向
```

#### 🔧 工具 4: resume_optimize_tool — STAR 简历优化

```
第449-487行: STAR_OPTIMIZE_PROMPT — STAR 优化提示词
             ├─ 定义 STAR 法则: Situation / Task / Action / Result
             ├─ 附 Few-Shot 示例（优化前 vs 优化后）
             └─ 要求输出优化后文案 + 对比表格

第490-524行: _resume_optimize(input_text)
             ├─ 分离 JD 和原始经历
             ├─ 智能截断到 2500 字符
             └─ invoke_llm_with_retry() 生成 STAR 改写
```

#### 🔧 工具 5: interview_question_tool — 面试题生成

```
第527-567行: INTERVIEW_QUESTION_PROMPT — 面试题生成提示词
             包含三类题目: 技术题 / 项目深挖题 / HR 行为题

第570-607行: _interview_question(target)
             ├─ 先从 RAG 面经知识库检索参考材料（top_k=5）
             ├─ 如果知识库未就绪，跳过参考直接用 LLM 生成
             └─ invoke_llm_with_retry() 生成三类题目
```

#### 🔧 工具 6: application_tracker_tool — 投递管理

```
第613-674行: _application_tracker(command)
             纯规则匹配的 CSV 增删改查，支持 4 种指令:
             ├─ "查询" / "查看": 读取 CSV 并格式化输出
             ├─ "新增 公司 岗位 日期 状态 [备注]": append 新行
             ├─ "更新 序号 新状态": 修改指定行的状态列
             ├─ "删除 序号": 删除指定行
             └─ 否则: 返回指令格式说明
```

#### 工具集合导出

```
第677-685行: ALL_TOOLS — 6 个工具列表
             按固定顺序导出给 agent_core.py
             [job_search, resume_parse, resume_match, resume_optimize, interview_question, application_tracker]
```

---

### 📄 `modules/agent_core.py` (364行)

**整体作用**：ReAct Agent 核心模块。定义系统提示词、创建/缓存 Agent 实例、提供标准调用和流式调用两种入口、全链路异常降级。

#### 逐段详解：

```
第1-23行   : 文档字符串 + imports
             ├─ create_react_agent: LangChain 官方 ReAct Agent 工厂函数
             ├─ AgentExecutor: 封装 Agent 的执行循环和错误处理
             ├─ PromptTemplate: 提示词模板引擎
             ├─ BaseMessage: 对话历史类型标注
             └─ invoke_llm_with_retry: 从 settings 导入

第26-29行  : Agent 缓存机制 【优化点】
             ├─ _agent_cache: Dict[str, AgentExecutor]
             │   └─ 按 session_id 键存储，会话级复用
             └─ clear_agent_cache(): 清除指定会话或全部缓存

第34-70行  : REACT_SYSTEM_PROMPT（系统提示词）【核心】
             这是整个 Agent 的"人格设定"，定义了：
             ├─ 角色定位: "专业求职辅导助手"
             ├─ 核心能力清单: 5 大能力
             ├─ 行为规范: 先分析→再调用工具→基于结果回答→不编造→用 Markdown
             ├─ 可用工具: {tools}（运行时由 LangChain 自动注入）
             └─ ReAct 格式规范: Thought → Action → Action Input → Observation → Final Answer

第74-80行  : REACT_PROMPT（提示词模板）
             将系统提示词 + 对话历史 + 用户输入 + Agent 中间步骤组合为完整 prompt
             模板变量: {chat_history} {input} {agent_scratchpad}

第85-114行 : create_agent() 函数
             ├─ 调用 get_llm() 获取 LLM 实例
             ├─ 调用 create_react_agent 创建 ReAct Agent
             │   └─ 注入 6 个工具 + ReAct Prompt
             ├─ 用 AgentExecutor 封装:
             │   ├─ max_iterations=8: 最多 8 轮循环
             │   ├─ handle_parsing_errors=True: 格式异常自动重试
             │   ├─ return_intermediate_steps=True: 返回思考链
             │   └─ max_execution_time=120: 120 秒超时
             └─ 返回配置好的 executor

第117-133行: get_or_create_agent() 函数 【优化点】
             会话级缓存复用:
             ├─ 按 session_id + temperature 构建键
             ├─ 命中缓存直接返回（避免每轮重建 Agent）
             └─ 未命中则创建并缓存

第138-206行: run_agent(query, chat_history, session_id) 函数 【核心入口】
             ┌─ 返回固定结构:
             │   {output, intermediate_steps, has_error, error_msg}
             ├─ 通过 get_or_create_agent() 获取 Agent
             ├─ 构建 invoke_args: {input, chat_history}
             ├─ 调用 executor.invoke()
             └─ 异常捕获 → 三层降级:

三层异常降级逻辑:
             ├─ Layer 1: timeout → API 超时
             │   └─ 降级: _fallback_direct_answer() + 功能引导
             ├─ Layer 2: token / context_length → Token 超限
             │   └─ 降级: 截断后直接回答 + 功能引导
             └─ Layer 3: 其他所有异常
                 └─ 降级: 基础对话 + 6 个功能入口引导

第211-312行: run_agent_streaming() 函数 【新增核心优化】
             ┌─ 生成器模式，逐步 yield 事件:
             │   ├─ {"type": "tool_start", "tool": "工具名", "input": "参数"}
             │   ├─ {"type": "tool_end", "tool": "工具名", "output": "结果"}
             │   ├─ {"type": "token", "content": "流式文字"}
             │   ├─ {"type": "done", "output": "完整回答"}
             │   └─ {"type": "error", "error_msg": "错误", "output": "降级回答"}
             ├─ 使用 astream_events 实现异步流式
             ├─ 降级方案: astream_events 不可用时自动切换为非流式
             │   └─ 按词分割输出模拟流式效果
             └─ 异常处理: 失败时 yield error 事件 + 降级回答

第317-362行: _fallback_direct_answer(query, prefix) 函数
             降级兜底方案:
             ├─ 用 invoke_llm_with_retry() 直接回答（带重试）
             ├─ 末尾附加 6 个快速功能入口
             └─ 如果 LLM 也失败 → 返回硬编码故障提示
```

---


### 📄 `main.py` (653行)

**整体作用**：Streamlit Web UI 主入口。定义了完整的页面布局和用户交互逻辑。优化版新增流式输出、配置持久化、表单式投递管理、增量更新按钮。

#### 逐段详解：

```
第1-24行   : 文档字符串 — 页面结构 + 优化点描述

第25-31行  : 标准库 + streamlit imports

第33-38行  : st.set_page_config()
             ├─ page_title: "实习求职智能助手 Agent"
             ├─ layout="wide": 宽屏布局
             └─ initial_sidebar_state="expanded": 侧边栏默认展开

第40-42行  : sys.path 处理
             将项目根目录加入 Python 路径

第44-45行  : LangChain 消息类型导入
             HumanMessage, AIMessage: 用于构建多轮对话历史

第47-60行  : 项目模块 imports
             ├─ config.settings: 全局配置 + invoke_llm_with_retry
             ├─ modules.rag_knowledge: RAG 知识库 + 增量更新
             ├─ modules.agent_core: run_agent + run_agent_streaming
             └─ modules.tools: 6 个工具函数

第63-76行  : 自定义 CSS 样式
             ├─ .tool-result: 工具调用结果卡片
             ├─ .thinking-box: 思考过程提示框（蓝色边框）
             └─ .streaming-cursor: 流式光标动画（▌闪烁）

第81-100行 : init_session_state() 函数
             初始化所有 Streamlit session_state 变量:
             ├─ chat_history: []
             ├─ vector_store: None
             ├─ resume_text / resume_path: 简历相关
             ├─ interview_active / interview_questions: 面试相关
             └─ OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL_NAME: 配置持久化

第106行    : init_session_state() 调用

第110-240行: 侧边栏（Sidebar）
             ├─ ⚙️ API 配置面板
             │   ├─ st.text_input: API Key（password类型）
             │   ├─ st.text_input: Base URL
             │   ├─ st.text_input: LLM 模型名
             │   ├─ st.text_input: Embedding 模型（disabled只读）
             │   └─ st.button("应用配置"):
             │       调用 update_api_config() + 写入 session_state 实现持久化
             │
             ├─ 📚 知识库管理
             │   ├─ 文档统计: JD 数量 + 面经数量
             │   ├─ st.button("构建知识库"): 全量重建
             │   ├─ st.button("加载知识库"): 加载已有
             │   └─ st.button("增量更新"): 【新增】仅索引新文档
             │
             └─ 📖 使用说明

第242-253行: 主区域标题

第257-263行: 4 个 Tab 页
             st.tabs(["💬 智能对话助手", "📄 简历优化", "🎤 模拟面试", "📌 投递管理"])

第269-354行: Tab 1 — 💬 智能对话助手（流式输出版）
             ├─ 流式输出 Toggle: 用户可选择开启/关闭
             ├─ 清空对话按钮: 同时清理 Agent 缓存
             ├─ 对话历史容器 (height=450px)
             │   └─ 有历史时逐条渲染 + 可展开思考过程
             │
             ├─ st.chat_input → 发送消息:
             │   ├─ 构建 LangChain 对话历史（最近10轮=20条）
             │   │
             │   ├─ 【流式模式】:
             │   │   ├─ status_placeholder: 实时显示 "🔧 正在调用: tool_name"
             │   │   ├─ text_placeholder: 逐字显示 LLM 回答（带▌光标）
             │   │   ├─ thinking_steps: 收集工具调用详情
             │   │   └─ thinking_expander: 可展开查看完整思考链
             │   │
             │   └─ 【非流式模式】:
             │       └─ st.spinner + run_agent() → 一次性返回

第360-420行: Tab 2 — 📄 简历优化
             ├─ 左列: 📤 上传 PDF
             │   ├─ st.file_uploader → 保存到 data/resume/
             │   ├─ _resume_parse() → LLM 结构化提取
             │   └─ 展示解析结果预览
             │
             └─ 右列: 📋 输入 JD
                 ├─ st.text_area: 粘贴 JD
                 ├─ st.button("匹配度评分") → _resume_match()
                 └─ st.button("STAR 优化") → _resume_optimize()

第425-483行: Tab 3 — 🎤 模拟面试
             ├─ st.text_input: 目标公司和岗位
             ├─ st.button("生成面试题") → _interview_question()
             ├─ 面试题展示
             ├─ st.text_area("写下回答")
             └─ st.button("获取反馈") → invoke_llm_with_retry() 三维度评价

第490-648行: Tab 4 — 📌 投递管理（表单优化版）
             ├─ 📊 统计看板 (5个 metric):
             │   总投递数 / 进行中 / Offer数 / 已拒数 / Offer率
             ├─ 📊 可视化图表:
             │   状态分布柱状图 + 公司 Top 5 柱状图
             ├─ ➕ 快捷添加表单 【优化点】:
             │   └─ st.form + 公司/岗位/状态/日期/备注字段
             ├─ 📋 交互式表格:
             │   └─ st.data_editor + num_rows="dynamic"
             ├─ 💾 保存按钮 → 写入 CSV
             └─ 📥 导出 CSV → st.download_button
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

# 2. 安装所有依赖
pip install -r requirements.txt

# 3. 安装本地 Embedding 模型依赖（推荐，零费用）
pip install sentence-transformers

# 4. 配置 API Key
# 复制 .env.example 为 .env
# 编辑 .env 文件，填入真实 API Key:

# OPENAI_API_KEY=sk-your-real-key-here
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# LLM_MODEL_NAME=deepseek-chat

# 5. 启动应用
streamlit run main.py

# 6. 浏览器打开 http://localhost:8501
```

### 首次使用流程

```
1. 侧边栏 "API 配置" → 填入 API Key，点击"应用配置"
2. 侧边栏 "知识库管理" → 点击 "🔨 构建知识库"
   （等待约 30 秒，首次需下载 Embedding 模型 ~100MB）
3. 看到 "✅ 知识库构建成功" 提示
4. 切换到 "💬 智能对话助手" Tab
5. 确认 "⚡ 流式输出" Toggle 已开启（默认开启）
6. 输入 "帮我找后端开发的实习岗位" 开始使用！
```

### 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 可选：安装本地 Embedding（推荐，零 API 费用）
pip install sentence-transformers

# 启动应用
streamlit run main.py

# 指定端口
streamlit run main.py --server.port 8502

# 生产模式（关闭自动重载）
streamlit run main.py --server.runOnSave false
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
| 知识库构建失败 | 检查网络能否访问 huggingface.co（国内可能需代理），或配置 HF_ENDPOINT 镜像 |
| 流式输出不工作 | 关闭 Toggle 切换为非流式模式，或检查 LangChain 版本 |
| tenacity 依赖冲突 | `tenacity>=8.0.0` 与 langchain 兼容，系统已有 8.5.0 |
| .env 被 git 追踪 | `git rm --cached .env` 然后确认 .gitignore 包含 .env |
| API 配置刷新后丢失 | 优化版已通过 session_state 自动持久化 |

---

## 7. 技术亮点（简历提炼版）

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **ReAct Agent 架构** | 基于 LangChain 原生 `create_react_agent`，自定义中文提示词，6 工具协同，`max_iterations=8` 防死循环 |
| 2 | **RAG 全链路 + 智能分割** | 文档加载→JD 1500/面经 800 智能分割→FAISS 向量化→本地持久化→语义检索 |
| 3 | **LLM Re-Ranking 重排序** | FAISS 粗筛 15 条 → LLM 逐条打分 → 精选 Top 3，检索准确率大幅提升 |
| 4 | **LLM 调用自动重试** | tenacity 指数退避（1s→2s→4s），所有 LLM 调用点全覆盖，网络抖动零影响 |
| 5 | **流式输出** | `astream_events` 实现实时 Agent 思考展示，用户可看到工具调用→结果→逐字回答全过程 |
| 6 | **LLM 参数化缓存** | 按 `(temperature, max_tokens, api_key, base_url, model)` 五维缓存，场景隔离 + 配置感知 |
| 7 | **多轮对话 + Agent 缓存** | 最近 10 轮对话注入 prompt + 会话级 Agent 复用，上下文连贯 |
| 8 | **三层异常降级** | API 超时 / Token 超限 / 调用失败分别捕获，自动切换为基础对话 + 功能引导，服务零中断 |
| 9 | **增强 JSON 解析** | 四层兜底：直接解析→正则代码块→范围提取→大括号计数，解决 LLM 格式不稳定的痛点 |
| 10 | **LLM 驱动简历解析** | pypdf 提取原文 → LLM 输出 JSON（姓名/教育/技能/经历/项目），替代简单关键词匹配 |
| 11 | **三级 Embedding 降级** | 本地 BAAI/bge-small-zh-v1.5（首选）→ OpenAI 兼容 API（备选）→ 明确报错 + 方案提示 |
| 12 | **交互式投递管理** | `st.data_editor` 单元格直接编辑 + 表单快捷添加 + 5 指标统计看板 + 可视化图表 |
| 13 | **增量知识库更新** | `add_documents_to_store()` 仅索引新文档，无需全量重建 |
| 14 | **配置持久化** | Streamlit session_state 存储 API 配置，页面刷新不丢失 |
| 15 | **统一日志 + 安全** | TimedRotatingFileHandler 按天轮转 + .gitignore 保护 .env + 启动时 git 追踪检测 |

---

## 8. 优化记录

本次优化（2026-07-15）解决了原项目的 10 个问题：

| # | 优先级 | 问题 | 解决方式 | 涉及文件 |
|---|--------|------|---------|---------|
| 1 | P2 | Agent 没有真正的对话记忆，每次重建 | Agent 会话级缓存 + session_id 参数 | `agent_core.py`, `main.py` |
| 2 | P1 | RAG 检索效果差，FAISS L2 距离区分度不够 | LLM Re-Ranking 重排序：粗筛 15 → 精选 3 | `tools.py` |
| 3 | P1 | chunk_size=500 太小，JD 信息被切碎 | 智能分割：JD→1500, 面经→800, 自动检测类型 | `rag_knowledge.py` |
| 4 | P1 | JSON 解析只有大括号计数，不够鲁棒 | 四层兜底：直接解析→正则代码块→范围提取→大括号计数 | `tools.py` |
| 5 | P0 | LLM 调用没有重试机制，网络抖动直接报错 | tenacity 指数退避 + invoke_llm_with_retry 封装 | `settings.py`, `tools.py`, `agent_core.py` |
| 6 | P0 | 缺少流式输出，用户等待 10-30 秒无反馈 | run_agent_streaming + astream_events + UI Toggle | `agent_core.py`, `main.py` |
| 7 | P2 | 知识库构建后不能增量更新，新文件需全量重建 | add_documents_to_store() + 侧边栏增量更新按钮 | `rag_knowledge.py`, `main.py` |
| 8 | P3 | 环境变量更新后刷新页面会被覆盖回去 | session_state 配置读取 + update_api_config 双向同步 | `settings.py`, `main.py` |
| 9 | P2 | 投递快捷指令输入框体验差，空格分隔易出错 | st.form 表单式输入：公司/岗位/状态/日期/备注 | `main.py` |
| 10 | P3 | 没有异步能力，同步阻塞 | astream_events 异步流式 + event loop 封装 | `agent_core.py` |

---

## 9. 完整项目结构

```
jobsearch-agent/
│
├── main.py                         # Streamlit 主入口 (653行)
├── requirements.txt                # Python 依赖清单
├── .env.example                    # 环境变量模板（可提交git）
├── .env                            # 真实环境变量（.gitignore已排除）
├── .gitignore                      # Git 忽略规则
├── README.md                       # 📖 本文档
├── INTERVIEW_PREP_V2.md            # 面试准备深度手册
├── TROUBLESHOOTING.md              # 问题排障记录
│
├── config/                         # ⚙️ 配置层
│   ├── __init__.py                 #   包标识
│   ├── settings.py                 #   全局配置核心 (346行)
│   │   ├── 环境变量加载
│   │   ├── LLM 参数化缓存
│   │   ├── invoke_llm_with_retry (自动重试)
│   │   ├── Embedding 三级降级
│   │   ├── session_state 配置持久化
│   │   └── .env 安全检测
│   └── logger.py                   #   统一日志系统 (102行)
│
├── modules/                        # 🧠 核心逻辑层
│   ├── __init__.py                 #   包标识
│   ├── agent_core.py               #   ReAct Agent 核心 (364行)
│   │   ├── REACT_SYSTEM_PROMPT
│   │   ├── create_agent()
│   │   ├── get_or_create_agent()   [会话缓存]
│   │   ├── run_agent()             [标准调用]
│   │   ├── run_agent_streaming()   [流式调用]
│   │   └── _fallback_direct_answer()
│   ├── rag_knowledge.py            #   RAG 知识库 (155行)
│   │   ├── load_documents()
│   │   ├── split_documents()       [智能分割]
│   │   ├── build_vector_store()
│   │   ├── load_vector_store()
│   │   ├── search_knowledge()
│   │   ├── add_documents_to_store() [增量更新]
│   │   └── build_knowledge_base()
│   └── tools.py                    #   6个工具函数 (682行)
│       ├── _extract_json()         [增强解析]
│       ├── _llm_rerank()           [AI重排序]
│       ├── _job_search()
│       ├── _resume_parse()
│       ├── _resume_match()
│       ├── _resume_optimize()
│       ├── _interview_question()
│       ├── _application_tracker()
│       └── ALL_TOOLS
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
│   │   └── jd_store/
│   └── applications.csv           #   投递记录（运行时生成）
│
└── logs/                           # 📝 应用日志（运行时生成）
    └── app.log                     #   按天轮转，保留7天
```
