"""
全局配置模块（优化版）
- 加载 .env 环境变量
- 提供 LLM（同步/异步）+ Embedding 实例
- 优先使用本地 BAAI/bge-small-zh-v1.5（需 sentence-transformers）
- 自动降级为 OpenAI 兼容 Embedding API
- LLM 调用内置重试机制
- Streamlit session_state 配置覆盖支持
"""
import os
from pathlib import Path
from typing import Optional, Tuple

from config.logger import get_logger

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── tenacity 重试 ──
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False

# ── 1. 环境变量加载 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# ── 2. 配置项读取 ──────────────────────────────────────────

def _get_config(key: str, default: str = "", session_state_key: Optional[str] = None) -> str:
    """优先从 streamlit session_state 读取，否则从环境变量读取"""
    if session_state_key:
        try:
            import streamlit as st
            if session_state_key in st.session_state:
                val = st.session_state.get(session_state_key)
                if val:
                    return val
        except (ImportError, RuntimeError):
            pass
    return os.getenv(key, default)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

# Embedding 配置（三选一自动检测）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

# 重试配置
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_MIN_WAIT = int(os.getenv("LLM_RETRY_MIN_WAIT", "1"))
LLM_RETRY_MAX_WAIT = int(os.getenv("LLM_RETRY_MAX_WAIT", "10"))

# ── 3. 数据目录路径 ──────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
JD_SAMPLES_DIR = DATA_DIR / "jd_samples"
INTERVIEW_DIR = DATA_DIR / "interview"
RESUME_DIR = DATA_DIR / "resume"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
APPLICATIONS_CSV = DATA_DIR / "applications.csv"


def ensure_data_dirs() -> None:
    """自动创建 data 下的所有子目录。"""
    logger = get_logger(__name__)
    dirs = [JD_SAMPLES_DIR, INTERVIEW_DIR, RESUME_DIR, VECTOR_STORE_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    if not APPLICATIONS_CSV.exists():
        APPLICATIONS_CSV.write_text(
            "公司,岗位,投递日期,状态,备注\n", encoding="utf-8"
        )
    logger.debug("数据目录初始化完成")


ensure_data_dirs()


# ── .env 安全检查 ──
def _check_env_safety() -> None:
    """启动时检查 .env 是否被意外提交到 git"""
    git_dir = PROJECT_ROOT.parent / ".git"
    if not git_dir.exists():
        return

    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger = get_logger(__name__)
            logger.warning(
                "⚠️ 安全警告: .env 文件已被 git 追踪！"
                " 请立即执行: git rm --cached .env 然后检查 .gitignore 是否包含 .env"
            )
    except Exception:
        pass


_check_env_safety()


# ── 4. 实例缓存 ─────────────────────────────────────────────
_embedding_instance = None
_embedding_mode = None  # "local" | "api" | None
_llm_cache = {}  # 按 (temperature, max_tokens) 参数化缓存


def _try_get_local_embedding():
    """尝试加载本地 HuggingFace Embedding 模型"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        return None, "langchain_community 版本过低，不支持 HuggingFaceEmbeddings"

    try:
        instance = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return instance, None
    except Exception as e:
        return None, str(e)


def _create_api_embedding():
    """创建 OpenAI 兼容的 Embedding 实例"""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
    )


def _get_active_api_key() -> str:
    """获取当前有效的 API Key（优先 session_state，其次环境变量）"""
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.session_state and st.session_state["OPENAI_API_KEY"]:
            return st.session_state["OPENAI_API_KEY"]
    except (ImportError, RuntimeError):
        pass
    return OPENAI_API_KEY


def _get_active_base_url() -> str:
    """获取当前有效的 Base URL"""
    try:
        import streamlit as st
        if "OPENAI_BASE_URL" in st.session_state and st.session_state["OPENAI_BASE_URL"]:
            return st.session_state["OPENAI_BASE_URL"]
    except (ImportError, RuntimeError):
        pass
    return OPENAI_BASE_URL


def _get_active_llm_model() -> str:
    """获取当前有效的 LLM 模型名"""
    try:
        import streamlit as st
        if "LLM_MODEL_NAME" in st.session_state and st.session_state["LLM_MODEL_NAME"]:
            return st.session_state["LLM_MODEL_NAME"]
    except (ImportError, RuntimeError):
        pass
    return LLM_MODEL_NAME


def get_llm(temperature: float = 0.2, max_tokens: int = 2048):
    """
    获取 LLM 实例（参数化缓存 + session_state 感知）。
    
    不同场景不同 temperature：
    - Agent 主调用: 0.2
    - 简历匹配: 0.1
    - 简历优化: 0.4
    - 面试题生成: 0.5
    """
    global _llm_cache
    
    active_key = _get_active_api_key()
    active_url = _get_active_base_url()
    active_model = _get_active_llm_model()
    
    # key 包含 API 配置，确保切换 API 后缓存失效
    cache_key = (temperature, max_tokens, active_key[:8] if active_key else "", active_url, active_model)

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if not active_key:
        raise RuntimeError(
            "❌ 未检测到 API Key！\n"
            "请在侧边栏「API 配置」中填入 Key，或复制 .env.example 为 .env 并填入有效的 API Key。"
        )

    try:
        instance = ChatOpenAI(
            model=active_model,
            openai_api_key=active_key,
            base_url=active_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=2,
        )
        _llm_cache[cache_key] = instance
        return instance
    except Exception as e:
        raise RuntimeError(
            f"❌ LLM 初始化失败：{e}\n"
            f"   当前配置：model={active_model}, base_url={active_url}\n"
            "   请检查：(1) API Key 是否有效 (2) Base URL 是否正确 (3) 网络是否连通"
        )


def invoke_llm_with_retry(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str:
    """
    LLM 调用封装（内置自动重试）。
    
    使用 tenacity 库实现指数退避重试：
    - 第1次失败：等 1 秒
    - 第2次失败：等 2 秒
    - 第3次失败：等 4 秒
    - 第4次失败：抛出异常
    
    如果 tenacity 未安装，退化为简单循环重试。
    """
    if _TENACITY_AVAILABLE:
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        
        @retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=1, min=LLM_RETRY_MIN_WAIT, max=LLM_RETRY_MAX_WAIT),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _invoke_with_retry():
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        
        return _invoke_with_retry()
    else:
        # 降级：简单循环重试
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        last_error = None
        import time
        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, LLM_RETRY_MAX_WAIT)
                    time.sleep(wait_time)
        raise last_error


def get_embedding():
    """
    获取 Embedding 单例实例，三级自动降级：
    1. 本地 BAAI/bge-small-zh-v1.5（最优，零费用）
    2. OpenAI 兼容 Embedding API
    3. 报错提示用户安装依赖或配置 API
    """
    global _embedding_instance, _embedding_mode
    if _embedding_instance is not None:
        return _embedding_instance

    # 第1级：本地 HuggingFace 模型
    local_instance, local_err = _try_get_local_embedding()
    if local_instance is not None:
        _embedding_instance = local_instance
        _embedding_mode = "local"
        return _embedding_instance

    local_warning = f"本地模型不可用: {local_err}" if local_err else "本地模型不可用"

    # 第2级：OpenAI 兼容 Embedding API
    if EMBEDDING_API_KEY:
        try:
            _embedding_instance = _create_api_embedding()
            _embedding_mode = "api"
            return _embedding_instance
        except Exception as e:
            raise RuntimeError(
                f"❌ Embedding 初始化失败（两级均失败）：\n"
                f"   第1级（本地模型）: {local_warning}\n"
                f"   第2级（API）: {e}\n\n"
                f"💡 解决方案（任选其一）：\n"
                f"   A. 安装本地模型依赖：pip install sentence-transformers\n"
                f"      首次运行会自动下载 {LOCAL_EMBEDDING_MODEL}（约100MB）\n"
                f"   B. 在 .env 中配置 EMBEDDING_API_KEY 使用在线 Embedding API"
            )

    raise RuntimeError(
        f"❌ 无可用的 Embedding 方案：\n"
        f"   本地模型: {local_warning}\n"
        f"   Embedding API: 未配置 EMBEDDING_API_KEY\n\n"
        f"💡 解决方案（任选其一）：\n"
        f"   A. pip install sentence-transformers（推荐，零费用）\n"
        f"   B. 在 .env 中配置 EMBEDDING_API_KEY 和 EMBEDDING_API_BASE"
    )


def get_embedding_mode() -> str:
    """返回当前 Embedding 模式：'local' | 'api' | None"""
    return _embedding_mode


def reset_instances() -> None:
    """重置缓存的单例/参数化实例"""
    global _embedding_instance, _embedding_mode, _llm_cache
    _embedding_instance = None
    _embedding_mode = None
    _llm_cache.clear()


def update_api_config(api_key: str, base_url: str, llm_model: str, emb_model: str = "") -> None:
    """
    动态更新 API 配置（用于 Streamlit 侧边栏临时覆盖）。
    同时更新全局变量和 session_state。
    """
    global OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL_NAME
    OPENAI_API_KEY = api_key
    OPENAI_BASE_URL = base_url
    LLM_MODEL_NAME = llm_model
    
    # 同步到 session_state
    try:
        import streamlit as st
        st.session_state["OPENAI_API_KEY"] = api_key
        st.session_state["OPENAI_BASE_URL"] = base_url
        st.session_state["LLM_MODEL_NAME"] = llm_model
    except (ImportError, RuntimeError):
        pass
    
    reset_instances()
