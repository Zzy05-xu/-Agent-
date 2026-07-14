# 🎯 实习求职智能助手 Agent — 面试深度准备手册

> 原则：面试官不会让你"念项目介绍"，他们会追问你的技术决策、遇到的具体问题、以及你怎么想的。
> 以下每个问题都配有**面试官追问的深层含义**和你的**应对话术**。

---

## 一、开场类

### Q1：「简单介绍一下你做的这个项目」

**面试官真实意图**：30秒内看你能不能抓住重点。

**推荐回答（30秒版）**：

> 我做了一个面向大学生实习求职的 AI Agent。用 LangChain 搭建 ReAct 框架，集成6个工具——能自动搜索岗位、评估简历匹配度、用 STAR 法则改写经历、生成面试题、追踪投递进度。LLM 用的 DeepSeek（中文强且便宜），Embedding 用的本地 BAAI/bge 模型（零成本方案）。技术上我比较满意的是设计了三级降级策略，不管是本地模型下载失败还是 API 欠费，系统都不会崩，能给用户明确的解决方案。

---

## 二、架构设计类

### Q2：「为什么不直接调 GPT API 写个聊天机器人？为什么要用 Agent？」

**关键认知**：

> 普通 Chatbot 是"一问一答"，Agent 是"规划-执行-反馈"的闭环。
>
> 用户说"帮我找数据分析的实习，再评估我的简历适不适合"。普通 Chatbot 要么编造岗位信息，要么只能给泛泛建议。而我的 Agent 会：Thought→Action(调 job_search 拿到真实 JD)→Observation→Thought→Action(调 resume_match)→Final Answer(整合两次工具调用的结果)。
>
> ReAct 是最经典的 LLM Agent 范式。优势：可解释（每步 Thought 可见）、可控（max_iterations 防死循环）、LangChain 原生支持一行创建。我也调研过 Function Calling（绑死 OpenAI）和 LangGraph（杀鸡用牛刀），ReAct 对 6 个工具的场景最合适。

### Q3：「如果工具之间有依赖关系怎么办？」

> Agent 自动处理依赖。ReAct 流程是串行的——第一次 Action 拿到结果后，Agent 根据 Observation 决定下一次 Action。比如"找字节跳动岗位→评估匹配度→优化不匹配的技能点"，Agent 会分 3 轮自动执行。关键点是我不需要手动编排，系统提示词里告诉它"先分析需求再决定调用哪些工具"就行。`max_iterations=8` 用于防止 Token 消耗过大和死循环。

### Q4：「你的系统提示词是怎么设计的？」

> 三个核心原则：
> - **角色锚定**：首句必须"专业求职辅导助手"，角色越具体行为越可控
> - **输出约束**：明确告知"不要编造信息，基于工具返回数据回答"、"遇错友好告知并给替代建议"
> - **格式模板**：完整 ReAct 格式（Thought/Action/Observation/Final Answer），配合 `handle_parsing_errors=True` 兜底
>
> 小技巧：在工具描述里写明"什么时候该调用这个工具"，比在系统提示词里列举工具列表效果好得多。

---

## 三、RAG 深度类

### Q5：「chunk_size 为什么选 500？」

> 试过 200/500/1000 三个值：200 太碎语义不完整，1000 太长精度下降，500 刚好保证每个片段完整语义且检索精度不丢。overlap=50 是 10% 重叠率。分隔符优先级 `\n\n > \n > 。 > 空格` 保证在自然段落边界切分。

### Q6：「FAISS 为什么比 Chroma/Milvus 更适合？」

> FAISS 零部署（pip install）、本地文件持久化、适合 <10万条。Milvus 需要 Server，我的 50 个文档片段用 FAISS 足够。Chroma 也行，但 FAISS 的 save_local 方案拷到别人电脑直接能用。

### Q7：「检索结果不相关怎么办？」

> 三个层面优化：1) 调 top_k + 相似度阈值，加入关键词+向量混合检索（Hybrid Search）2) 检查源文档覆盖度，chunk_size 按文档类型动态调整 3) 引入 Re-ranking——LLM 读一遍 Top3 重新排序。目前项目做了第一层。

### Q8：「为什么选 bge-small-zh-v1.5 而不是 m3e 或 text2vec？」

> m3e-base 400MB+，是 bge 的 4 倍，对 CPU 不友好。text2vec 侧重句子相似度，检索 MTEB 评分不如 bge。bge-small 在 C-MTEB 检索子项排名前十，体积 100MB，CPU 推理 10ms 内，中文检索效果×体积×速度三者最优。

---

## 四、工程实践类（核心亮点）

### Q9：「三级降级策略具体怎么实现？」

> 因为 Embedding 是 Agent 核心依赖但也是最脆弱环节——本地模型可能没装/下载失败，API 可能 Key 不对/欠费/超时。
>
> 代码实现：`get_embedding()` 里 try 加载本地模型 → 失败则 try 在线 API → 都失败则抛出明确 RuntimeError 含两种解决方案。
>
> 验证过的失败场景：没装 sentence-transformers → 走 API；huggingface 被墙 → 配 HF_ENDPOINT 镜像；API 过期 → 回落本地（如已下载）；两级都挂 → 用户看到明确解决方案。

### Q10：「Agent 异常降级怎么做的？」

> 分类降级而非统一 catch：**超时**→换直接 LLM 回答+引导手动功能；**Token 超限**→截断上下文后用 LLM；**未知错误**→通用兜底+错误说明。核心原则：降级后服务不中断，所有路径返回对用户有意义的回答。

### Q11：「JSON 提取器有没有更优雅方案？」

> 大括号计数法 30 行覆盖 95% 场景。更优雅方案：LangChain StructuredOutputParser（但不够灵活）、OpenAI JSON Mode（绑死 GPT）、Pydantic+Instructor（最优雅但需额外依赖）。我的项目定位"最小依赖"，选了可控的实现。

---

## 五、踩坑与反思

### Q12：「最大的坑？」

> 不是代码逻辑，是**环境兼容性**。连踩 4 坑：`langchain_huggingface` 不存在 → `langchain_community` 已内置；huggingface.co 被墙 → `HF_ENDPOINT` 镜像；sentence-transformers 升级 numpy 到 2.5.1 → 锁定 `numpy<2.0`；Streamlit expander 不能嵌套。教训：锁定版本、预判网络问题、每个外部依赖都要有 fallback。

### Q13：「重做一次会怎么设计？」

> 三个架构调整：1) Tool Registry——工具自描述+自动注册，解耦 Agent 核心 2) Tool Router——轻量 LLM 做意图识别缩小候选工具范围 3) 评估体系——用 RAGAS 做 RAG hit rate 和 Agent 回答准确率的自动评估。

---

## 六、课程项目 vs 独立项目

> 课程项目：需求/技术栈/数据/评分标准都是给定的。我的项目：需求是自己分析求职痛点、架构是自己设计三层+三级降级、数据是自己写示例 JD 和面经、容错处理了 8 种真实环境问题。最大的区别：课程项目跑通就行，我的项目是在**真实环境**（DeepSeek、HuggingFace 被墙、numpy 冲突）里跑通的。

---

## 七、简历描述（精简版）

```
实习求职智能助手 Agent | Python, LangChain, FAISS, Streamlit | 独立开发

- 基于 LangChain ReAct 框架搭建智能体，集成岗位检索、简历匹配评分、
  STAR 优化、面试题生成、投递管理等 6 个工具，实现 Thought→Action→Final Answer 推理链路
- 实现 RAG 全链路：RecursiveCharacterTextSplitter 语义分割 + FAISS 向量索引
  + 本地 BAAI/bge-small-zh-v1.5 中文 Embedding，零成本语义检索
- 设计三级 Embedding 降级策略：本地模型→在线 API→兜底提示，确保多环境可用
- 全链路异常处理：API 超时/Token 超限分类降级、JSON 大括号计数提取器校验
- 使用 Streamlit 构建 4 Tab 交互界面，session_state 管理对话历史与向量库缓存
```
