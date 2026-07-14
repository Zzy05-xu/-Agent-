"""
全局配置模块
- 加载 .env 环境变量
- 提供单例 LLM（DeepSeek）与 Embedding 实例
- 优先使用本地 BAAI/bge-small-zh-v1.5（需 sentence-transformers）
- 自动降级为 OpenAI 兼容 Embedding API
- 自动创建数据子目录
"""
import os
from pathlib import Path

from config.logger import get_logger

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── 1. 环境变量加载 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# ── 2. 配置项读取 ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

# Embedding 配置（三选一自动检测，优先级从高到低）
# HuggingFace 镜像站（国内访问加速，.env 中配置 HF_ENDPOINT）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# 当本地模型不可用时，使用 OpenAI 兼容的 Embedding API
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

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
    """
    启动时检查 .env 是否被意外提交到 git。
    
    如果检测到 .env 已被 git track，输出警告（不会阻塞运行）。
    """
    git_dir = PROJECT_ROOT.parent / ".git"
    if not git_dir.exists():
        return  # 非 git 仓库，无需检查

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
        pass  # git 不可用时静默跳过


_check_env_safety()


# ── 4. 单例实例 ─────────────────────────────────────────────
_llm_instance = None
_embedding_instance = None
_embedding_mode = None  # "local" | "api" | None
_llm_cache = {}  # 按 (temperature, max_tokens) 参数化缓存


def _try_get_local_embedding():
    """
    尝试加载本地 HuggingFace Embedding 模型。
    需要 sentence-transformers 已安装。首次运行自动下载模型。
    """
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
    """创建 OpenAI 兼容的 Embedding 实例。"""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
    )


def get_llm(temperature: float = 0.2, max_tokens: int = 2048):
    """
    获取 LLM 实例（按 temperature + max_tokens 参数化缓存）。
    
    不同场景需要不同的 temperature：
    - Agent 主调用: 0.2（平衡创造性和准确性）
    - 简历匹配: 0.1（追求评分一致性）
    - 简历优化: 0.4（需要改写创意）
    - 面试题生成: 0.5（需要多样性）
    
    参数化缓存确保每个场景的参数被正确应用。
    """
    global _llm_cache
    key = (temperature, max_tokens)

    if key in _llm_cache:
        return _llm_cache[key]

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "❌ 未检测到 OPENAI_API_KEY 环境变量！\n"
            "请复制 .env.example 为 .env 并填入有效的 API Key。"
        )

    try:
        instance = ChatOpenAI(
            model=LLM_MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=2,
        )
        _llm_cache[key] = instance
        return instance
    except Exception as e:
        raise RuntimeError(
            f"❌ LLM 初始化失败：{e}\n"
            f"   当前配置：model={LLM_MODEL_NAME}, base_url={OPENAI_BASE_URL}\n"
            "   请检查：(1) API Key 是否有效 (2) Base URL 是否正确 (3) 网络是否连通"
        )


def get_embedding():
    """
    获取 Embedding 单例实例，三级自动降级：
    1. 本地 BAAI/bge-small-zh-v1.5（最优，零费用，需 pip install sentence-transformers）
    2. OpenAI 兼容 Embedding API（需配置 EMBEDDING_API_KEY/EMBEDDING_API_BASE）
    3. 报错提示用户安装依赖或配置 API
    """
    global _embedding_instance, _embedding_mode
    if _embedding_instance is not None:
        return _embedding_instance

    # ── 第1级：本地 HuggingFace 模型 ──
    local_instance, local_err = _try_get_local_embedding()
    if local_instance is not None:
        _embedding_instance = local_instance
        _embedding_mode = "local"
        return _embedding_instance

    local_warning = f"本地模型不可用: {local_err}" if local_err else "本地模型不可用"

    # ── 第2级：OpenAI 兼容 Embedding API ──
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

    # ── 两级都不可用 ──
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
    """重置缓存的单例/参数化实例。"""
    global _llm_instance, _embedding_instance, _embedding_mode, _llm_cache
    _llm_instance = None
    _embedding_instance = None
    _embedding_mode = None
    _llm_cache.clear()


def update_api_config(api_key: str, base_url: str, llm_model: str, emb_model: str = "") -> None:
    """动态更新 API 配置（用于 Streamlit 侧边栏临时覆盖）。"""
    global OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL_NAME
    OPENAI_API_KEY = api_key
    OPENAI_BASE_URL = base_url
    LLM_MODEL_NAME = llm_model
    reset_instances()

