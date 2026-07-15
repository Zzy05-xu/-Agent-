"""
RAG 知识库模块（优化版）
核心功能：文档加载 → 文本分割 → 向量化存储 → 语义检索

优化点：
- chunk_size 区分 JD(1500) 和面经(800)，保持信息完整性
- 支持增量更新知识库
- FAISS 粗筛扩大召回（供 Re-Ranking 使用）
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


def load_documents(data_dir: str) -> List[Document]:
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
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(data_path))
            all_documents.extend(docs)
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"加载文件失败 [{file_path.name}]: {e}")
    if not all_documents:
        raise ValueError(f"❌ 在 {data_dir} 下未找到任何可加载的 .txt/.md/.pdf 文件。")
    return all_documents


def _detect_doc_type(doc: Document) -> str:
    source = doc.metadata.get("source", "").lower()
    if "interview" in source:
        return "interview"
    return "jd"


def split_documents(documents: List[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[Document]:
    jd_docs = []
    interview_docs = []
    for doc in documents:
        if _detect_doc_type(doc) == "interview":
            interview_docs.append(doc)
        else:
            jd_docs.append(doc)
    all_chunks = []
    if jd_docs:
        jd_size = chunk_size if chunk_size is not None else 1500
        jd_overlap = chunk_overlap if chunk_overlap is not None else 200
        jd_splitter = RecursiveCharacterTextSplitter(chunk_size=jd_size, chunk_overlap=jd_overlap, separators=["\n\n", "\n", "。", ".", " ", ""], length_function=len)
        all_chunks.extend(jd_splitter.split_documents(jd_docs))
    if interview_docs:
        iv_size = chunk_size if chunk_size is not None else 800
        iv_overlap = chunk_overlap if chunk_overlap is not None else 100
        iv_splitter = RecursiveCharacterTextSplitter(chunk_size=iv_size, chunk_overlap=iv_overlap, separators=["\n\n", "\n", "。", ".", " ", ""], length_function=len)
        all_chunks.extend(iv_splitter.split_documents(interview_docs))
    logger = get_logger(__name__)
    logger.info(f"分割完成: JD={len(jd_docs)}块, 面经={len(interview_docs)}块, 总计={len(all_chunks)}块")
    return all_chunks


def build_vector_store(documents: List[Document], save_path: str) -> FAISS:
    try:
        embedding = get_embedding()
        vector_store = FAISS.from_documents(documents, embedding)
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"❌ 向量库构建失败: {e}")


def load_vector_store(save_path: str) -> FAISS:
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"❌ 向量库不存在: {save_path}\n   请先在侧边栏「知识库管理」中点击「构建知识库」。")
    try:
        embedding = get_embedding()
        vector_store = FAISS.load_local(save_path, embedding, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"❌ 向量库加载失败: {e}")


def search_knowledge(vector_store: FAISS, query: str, top_k: int = 3) -> List[dict]:
    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        return [{"content": f"检索异常: {e}", "source": "N/A", "score": 0.0}]
    formatted = []
    for doc, score in results_with_scores:
        formatted.append({"content": doc.page_content.strip(), "source": doc.metadata.get("source", "未知来源"), "score": round(score, 4)})
    return formatted


def add_documents_to_store(data_dir: str, vector_store_path: str) -> FAISS:
    logger = get_logger(__name__)
    vector_store = load_vector_store(vector_store_path)
    try:
        logger.info(f"正在加载新增文档: {data_dir}")
        new_docs = load_documents(data_dir)
        logger.info(f"加载完成，共 {len(new_docs)} 个新文档片段")
    except Exception as e:
        raise RuntimeError(f"新增文档加载失败: {e}")
    try:
        chunks = split_documents(new_docs)
        logger.info(f"分割完成，共 {len(chunks)} 个新文本片段")
    except Exception as e:
        raise RuntimeError(f"文档分割失败: {e}")
    try:
        vector_store.add_documents(chunks)
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(vector_store_path)
        logger.info(f"向量库已更新并保存至: {vector_store_path}")
    except Exception as e:
        raise RuntimeError(f"增量更新失败: {e}")
    return vector_store


def build_knowledge_base(data_dir: str, vector_store_path: str) -> FAISS:
    logger = get_logger(__name__)
    try:
        logger.info(f"正在加载文档: {data_dir}")
        docs = load_documents(data_dir)
        logger.info(f"加载完成，共 {len(docs)} 个文档片段（分割前）")
    except Exception as e:
        raise RuntimeError(f"文档加载失败: {e}")
    try:
        logger.info("正在分割文档（JD: chunk_size=1500, 面经: chunk_size=800）...")
        chunks = split_documents(docs)
        logger.info(f"分割完成，共 {len(chunks)} 个文本片段")
    except Exception as e:
        raise RuntimeError(f"文档分割失败: {e}")
    try:
        logger.info("正在构建向量库（首次需加载本地 Embedding 模型）...")
        vector_store = build_vector_store(chunks, vector_store_path)
        logger.info(f"向量库已保存至: {vector_store_path}")
    except Exception as e:
        raise RuntimeError(f"向量库构建失败: {e}")
    return vector_store
