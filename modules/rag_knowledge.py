"""
RAG 知识库模块
核心功能：文档加载 → 文本分割 → 向量化存储 → 语义检索

架构设计：
1. 支持 .txt / .md / .pdf 三种格式的文档
2. 使用 FAISS 轻量向量库，无需额外服务部署
3. 向量库持久化到本地，重启后可直接加载复用
4. 全链路异常处理，检索失败返回明确错误信息
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.logger import get_logger
from config.settings import get_embedding


# ── 1. 文档加载 ──────────────────────────────────────────────

def load_documents(data_dir: str) -> List[Document]:
    """
    递归加载指定目录下所有 .txt / .md / .pdf 文件。
    
    实现思路：
    - 遍历目录树，按扩展名分发到对应的 LangChain 官方加载器
    - TextLoader 处理 .txt / .md（设置 autodetect_encoding=True 避免编码问题）
    - PyPDFLoader 处理 .pdf
    - 每个文档的 metadata 中记录来源文件路径，便于溯源
    
    Args:
        data_dir: 包含文档的目录路径
    
    Returns:
        Document 对象列表
    
    Raises:
        FileNotFoundError: 目录不存在
        ValueError: 未找到任何可加载文档
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ 目录不存在: {data_dir}")

    all_documents: List[Document] = []
    supported_extensions = {".txt", ".md", ".pdf"}

    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            if file_path.suffix.lower() in (".txt", ".md"):
                loader = TextLoader(
                    str(file_path), encoding="utf-8"
                )
            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                continue

            docs = loader.load()
            # 为每个文档片段标记来源文件
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(data_path))
            all_documents.extend(docs)
        except Exception as e:
            # 单个文件加载失败不阻塞整体流程，但收集错误信息
            import traceback
            error_detail = traceback.format_exc()
            logger = get_logger(__name__)
            logger.warning(f"加载文件失败 [{file_path.name}]: {e}")

    if not all_documents:
        raise ValueError(
            f"❌ 在 {data_dir} 下未找到任何可加载的 .txt/.md/.pdf 文件。\n"
            "   请确保已准备示例数据或上传文档。"
        )

    return all_documents


# ── 2. 文本分割 ──────────────────────────────────────────────

def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 进行语义级文本分割。
    
    参数设计：
    - chunk_size=500：每个片段约500字符，适配短文本检索场景（JD、面经）
    - chunk_overlap=50：10% 重叠率，保证跨片段语义连续性
    - 分隔符优先级：双换行 > 单换行 > 句号 > 空格 > 字符
    
    Args:
        documents: 原始文档列表
        chunk_size: 每个片段的最大字符数
        chunk_overlap: 相邻片段的重叠字符数
    
    Returns:
        分割后的文档片段列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


# ── 3. 向量库构建 ────────────────────────────────────────────

def build_vector_store(
    documents: List[Document],
    save_path: str,
) -> FAISS:
    """
    基于 FAISS 构建向量库并持久化到本地。
    
    实现思路：
    - 调用 Embedding 模型将文档片段转为向量
    - 使用 FAISS.from_documents 构建索引
    - save_local 持久化，后续可通过 load_vector_store 直接加载
    
    Args:
        documents: 已分割的文档片段列表
        save_path: 向量库保存路径
    
    Returns:
        FAISS 向量库实例
    """
    try:
        embedding = get_embedding()
        vector_store = FAISS.from_documents(documents, embedding)
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"❌ 向量库构建失败: {e}")


# ── 4. 向量库加载 ────────────────────────────────────────────

def load_vector_store(save_path: str) -> FAISS:
    """
    加载本地已持久化的 FAISS 向量库。
    
    Args:
        save_path: 向量库保存目录
    
    Returns:
        FAISS 向量库实例
    
    Raises:
        FileNotFoundError: 向量库路径不存在
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"❌ 向量库不存在: {save_path}\n"
            "   请先在侧边栏「知识库管理」中点击「构建知识库」。"
        )

    try:
        embedding = get_embedding()
        vector_store = FAISS.load_local(
            save_path, embedding, allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"❌ 向量库加载失败: {e}")


# ── 5. 语义检索 ──────────────────────────────────────────────

def search_knowledge(
    vector_store: FAISS,
    query: str,
    top_k: int = 3,
) -> List[dict]:
    """
    执行相似度检索，返回格式化结果。
    
    工作流程：
    1. 将查询文本转为向量
    2. 在 FAISS 索引中检索 top_k 个最相似片段
    3. 格式化输出，包含内容摘要和来源文件信息
    
    Args:
        vector_store: FAISS 向量库实例
        query: 检索查询文本
        top_k: 返回的最相似结果数量
    
    Returns:
        格式化检索结果列表，每项包含 content、source、score（距离越小越相似）
    """
    try:
        # FAISS.similarity_search_with_score 返回 (Document, score) 元组
        results_with_scores = vector_store.similarity_search_with_score(
            query, k=top_k
        )
    except Exception as e:
        return [
            {
                "content": f"检索异常: {e}",
                "source": "N/A",
                "score": 0.0,
            }
        ]

    formatted = []
    for doc, score in results_with_scores:
        formatted.append({
            "content": doc.page_content.strip(),
            "source": doc.metadata.get("source", "未知来源"),
            "score": round(score, 4),  # L2 距离，越小越相似
        })

    return formatted


# ── 6. 一站式知识库构建入口 ──────────────────────────────────

def build_knowledge_base(
    data_dir: str,
    vector_store_path: str,
) -> FAISS:
    """
    一站式知识库构建流程：加载 → 分割 → 向量化 → 持久化。
    
    供 Streamlit UI 侧边栏按钮调用。
    
    Args:
        data_dir: 文档源目录
        vector_store_path: 向量库保存路径
    
    Returns:
        FAISS 向量库实例
    
    Raises:
        RuntimeError: 任何步骤失败时抛出明确错误
    """
    import sys
    
    # 第一步：加载文档
    try:
        logger = get_logger(__name__)
        logger.info(f"正在加载文档: {data_dir}")
        docs = load_documents(data_dir)
        logger.info(f"加载完成，共 {len(docs)} 个文档片段（分割前）")
    except Exception as e:
        raise RuntimeError(f"文档加载失败: {e}")

    # 第二步：分割文档
    try:
        logger.info("正在分割文档...")
        chunks = split_documents(docs)
        logger.info(f"分割完成，共 {len(chunks)} 个文本片段")
    except Exception as e:
        raise RuntimeError(f"文档分割失败: {e}")

    # 第三步：构建向量库（这里会调用 get_embedding()）
    try:
        logger.info("正在构建向量库（首次需加载本地 Embedding 模型）...")
        vector_store = build_vector_store(chunks, vector_store_path)
        logger.info(f"向量库已保存至: {vector_store_path}")
    except Exception as e:
        raise RuntimeError(f"向量库构建失败: {e}")

    return vector_store



