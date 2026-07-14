# 🎯 实习求职智能助手 Agent — 问题记录与解决方案

> 项目路径: `C:\Users\ASUS\Documents\Codex\2026-07-13\agent-5-python-langchain-rag-react\jobsearch-agent\`  
> 最终架构: LLM(DeepSeek) + 本地 Embedding(BAAI/bge-small-zh-v1.5) + FAISS + Streamlit  
> 记录时间: 2026-07-13

---

## 问题 1：缺少 langchain_huggingface 模块

**现象**  
```
ModuleNotFoundError: No module named 'langchain_huggingface'
```

**原因**  
`config/settings.py` 最早从 `langchain_huggingface` 导入 `HuggingFaceEmbeddings`，但这是一个独立 pip 包，未安装。

**解决方案**  
将导入改为 `langchain_community.embeddings`（`langchain-community` 0.2.10 已内置 `HuggingFaceEmbeddings`）：

```python
# 旧
from langchain_huggingface import HuggingFaceEmbeddings

# 新
from langchain_community.embeddings import HuggingFaceEmbeddings
```

**涉及文件**: `config/settings.py`

---

## 问题 2：Streamlit 页面 SyntaxError — positional argument follows keyword argument

**现象**  
```
File "main.py", line 133
          )
          ^
SyntaxError: positional argument follows keyword argument
```

**原因**  
修改侧边栏 Embedding 输入框时，正则替换错误，导致 `st.text_input()` 参数列表中混入了孤立的 `EMBEDDING_MODEL_NAME,`，使位置参数出现在关键字参数之后。

**解决方案**  
完整重写该 `st.text_input()` 调用块，确保参数顺序正确：

```python
temp_emb_model = st.text_input(
    mode_label,
    value=LOCAL_EMBEDDING_MODEL,
    disabled=True,
    help="本地模型: pip install sentence-transformers | 备选: 在.env配置EMBEDDING_API_KEY",
)
```

**涉及文件**: `main.py`（第 127-134 行）

---

## 问题 3：构建知识库时找不到可加载文件

**现象**  
```
构建失败: ❌ 在 data 下未找到任何可加载的 .txt/.md/.pdf 文件。
```

**原因**  
初步怀疑 `TextLoader` 的 `autodetect_encoding` 参数在 `langchain-community` 0.2.10 中行为不同，导致所有文件加载都抛异常被 `except` 吞掉。

**解决方案**  
将 `TextLoader` 参数从 `autodetect_encoding=True` 改为 `encoding="utf-8"`，同时增强异常日志：

```python
# 旧
loader = TextLoader(str(file_path), autodetect_encoding=True)

# 新
loader = TextLoader(str(file_path), encoding="utf-8")
```

并在 catch 块中加入 `traceback.format_exc()` 输出详细错误。

**涉及文件**: `modules/rag_knowledge.py`

---

## 问题 4：构建知识库一直转圈、无响应

**现象**  
点击「构建知识库」后 spinner 一直转，没有成功也没有报错提示。

**原因**  
`build_knowledge_base` 内部调用 `get_embedding()` → 尝试加载本地模型 → 抛出 `RuntimeError` → 但在旧的 `build_knowledge_base` 中 `build_vector_store` 的异常虽然 `raise` 了，但异常信息在 Streamlit 的 spinner 中展示不完整。

**解决方案**  
将 `build_knowledge_base` 拆为三步，每步独立 try/except 并 `raise RuntimeError`，确保 Streamlit 的 `st.error()` 能完整展示错误信息。

```python
try:
    print("📂 正在加载文档...", flush=True)
    docs = load_documents(data_dir)
except Exception as e:
    raise RuntimeError(f"文档加载失败: {e}")
# ... 同理分割和向量化
```

**涉及文件**: `modules/rag_knowledge.py`

---

## 问题 5：StreamlitAPIException — expander 不能嵌套

**现象**  
```
StreamlitAPIException: Expanders may not be nested inside other expanders.
```

**原因**  
`main.py` 中构建知识库的 `except` 块里用了 `st.expander("详细信息")`，但这段代码在侧边栏已有的 `st.expander("知识库管理")` 内部，Streamlit 不允许 expander 嵌套。

**解决方案**  
将错误展示从 `st.expander` 改为 `st.caption` 提示：

```python
st.error(f"构建失败: {err_msg}")
st.caption("💡 提示: 请确认已执行 pip install sentence-transformers 且网络可访问 huggingface.co")
```

**涉及文件**: `main.py`

---

## 问题 6：HuggingFace 模型下载 404 错误

**现象**  
```
构建失败: 向量库构建失败: ❌ 向量库构建失败: Error code: 404
```

**原因**  
`HuggingFaceEmbeddings` 尝试从 `huggingface.co` 下载 `BAAI/bge-small-zh-v1.5` 模型，但国内网络无法访问 HuggingFace Hub，导致 404。

**解决方案**  
在 `.env` 中配置 HuggingFace 国内镜像站 `hf-mirror.com`：

```
HF_ENDPOINT=https://hf-mirror.com
```

`load_dotenv()` 会自动将该环境变量写入 `os.environ`，`sentence-transformers` 底层会自动读取 `HF_ENDPOINT` 来走镜像下载。

**涉及文件**: `.env`

---

## 问题 7：numpy 版本冲突

**现象**  
```
ERROR: pip's dependency resolver does not currently take into account all the packages ...
faiss-cpu 1.8.0.post1 requires numpy<2.0,>=1.0, but you have numpy 2.5.1 which is incompatible.
langchain 0.2.11 requires numpy<2.0.0,>=1.26.0, but you have numpy 2.5.1 which is incompatible.
langchain-community 0.2.10 requires numpy<2.0.0,>=1.26.0, but you have numpy 2.5.1 which is incompatible.
```

**原因**  
环境中已安装 numpy 2.5.1，但 `faiss-cpu`、`langchain`、`langchain-community` 都要求 `numpy<2.0`。`sentence-transformers` 安装时附带升级了 numpy 到最新版。

**解决方案**  
强制降级 numpy 到兼容版本：

```bash
pip install "numpy>=1.26,<2.0"
```

**注意**: `numpy 1.26` 与 `numpy 2.x` 的 API 差异不影响本项目使用的 LangChain/FAISS/Pandas 接口。

---

## 问题 8：DeepSeek API 不支持 Embedding

**现象（潜在问题，已预防）**  
DeepSeek 的 API 接口（`api.deepseek.com/v1`）不提供 Embedding 服务，无法直接使用 `OpenAIEmbeddings`。

**解决方案**  
采用**三级自动降级**的 Embedding 获取策略：

| 优先级 | 方案 | 条件 | 费用 |
|--------|------|------|------|
| 第1级 | 本地 `BAAI/bge-small-zh-v1.5` | `pip install sentence-transformers` | 免费 |
| 第2级 | 在线 Embedding API | 在 `.env` 配置 `EMBEDDING_API_KEY` | 按量 |
| 第3级 | 报错 + 解决方案提示 | 前两级都不可用 | — |

代码实现（`config/settings.py`）：

```python
def get_embedding():
    # 第1级：本地 HuggingFace 模型
    local_instance, local_err = _try_get_local_embedding()
    if local_instance is not None:
        return local_instance

    # 第2级：OpenAI 兼容 Embedding API
    if EMBEDDING_API_KEY:
        return _create_api_embedding()

    # 第3级：给出解决方案提示
    raise RuntimeError(
        "❌ 无可用的 Embedding 方案\n"
        "A. pip install sentence-transformers（推荐，零费用）\n"
        "B. 在 .env 中配置 EMBEDDING_API_KEY"
    )
```

**涉及文件**: `config/settings.py`

---

## 最终可用配置总结

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `OPENAI_API_KEY` | DeepSeek Key | LLM 调用 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek 接口 |
| `LLM_MODEL_NAME` | `deepseek-chat` | DeepSeek 模型 |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地中文向量模型 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内 HuggingFace 镜像 |
| `numpy` 版本 | `>=1.26,<2.0` | 兼容 faiss-cpu / langchain |

### 启动步骤（最终版）

```bash
cd jobsearch-agent
pip install -r requirements.txt
pip install sentence-transformers
pip install "numpy>=1.26,<2.0"
streamlit run main.py
```

浏览器打开后：侧边栏 API 已预填 DeepSeek Key → 点击「构建知识库」等待 1-2 分钟（首次下载模型）→ 即可使用全部功能。
