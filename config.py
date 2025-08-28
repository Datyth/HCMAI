config_code = """
Configuration settings for Vietnamese Video Knowledge Graph System
Cấu hình hệ thống cho Vietnamese Video Knowledge Graph
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class with all system settings"""
    
    # === Neo4j Database Configuration ===
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Connection settings
    NEO4J_MAX_CONNECTION_LIFETIME = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600"))
    NEO4J_MAX_CONNECTION_POOL_SIZE = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
    NEO4J_CONNECTION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30"))
    
    # # === OpenAI Configuration ===
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    # OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    # OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    # OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))
    
    # === Embedding Configuration ===
    VIETNAMESE_EMBEDDING_MODEL = os.getenv(
        "VIETNAMESE_EMBEDDING_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))  # MiniLM-L12-v2 dimension
    
    # Alternative models for different use cases
    EMBEDDING_MODELS = {
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim
        "vietnamese": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",    # 768 dim
        "fast": "sentence-transformers/all-MiniLM-L6-v2",                              # 384 dim
        "accurate": "sentence-transformers/all-mpnet-base-v2"                          # 768 dim
    }
    
    # === Graph Configuration ===
    VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "video_embeddings")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
    DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
    
    # === File Paths ===
    BASE_DIR = Path(__file__).parent
    OUTPUT_FILES_DIR = os.getenv("OUTPUT_FILES_DIR", str(BASE_DIR / "output_files"))
    LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
    CACHE_DIR = os.getenv("CACHE_DIR", str(BASE_DIR / "cache"))
    
    # === Vietnamese NLP Configuration ===
    VIETNAMESE_STOP_WORDS = {
        'là', 'của', 'và', 'có', 'trong', 'được', 'với', 'từ', 'theo',
        'để', 'về', 'này', 'đó', 'những', 'các', 'một', 'hai', 'ba',
        'rất', 'nhiều', 'ít', 'lớn', 'nhỏ', 'tốt', 'xấu', 'mới', 'cũ',
        'cho', 'khi', 'nếu', 'thì', 'sẽ', 'đã', 'bị', 'bởi', 'vì',
        'nên', 'mà', 'tại', 'trên', 'dưới', 'giữa', 'sau', 'trước'
    }
    
    # Vietnamese entity types for knowledge graph
    VIETNAMESE_ENTITY_TYPES = [
        "người",           # Person
        "địa_điểm",       # Location
        "tổ_chức",        # Organization
        "sự_kiện",        # Event
        "khái_niệm",      # Concept
        "sản_phẩm",       # Product
        "dịch_vụ",        # Service
        "công_nghệ",      # Technology
        "phương_pháp",    # Method
        "video",          # Video
        "phần",           # Section
        "thời_gian",      # Time
        "số_liệu",        # Data/Number
        "tài_liệu"        # Document
    ]
    
    # Vietnamese relationship types
    VIETNAMESE_RELATIONSHIP_TYPES = [
        ("video", "có_phần", "phần"),
        ("phần", "đề_cập", "khái_niệm"),
        ("phần", "giới_thiệu", "người"),
        ("phần", "thảo_luận", "sự_kiện"),
        ("người", "làm_việc_tại", "tổ_chức"),
        ("người", "sống_tại", "địa_điểm"),
        ("người", "tạo_ra", "sản_phẩm"),
        ("sự_kiện", "xảy_ra_tại", "địa_điểm"),
        ("sự_kiện", "xảy_ra_vào", "thời_gian"),
        ("khái_niệm", "liên_quan", "khái_niệm"),
        ("khái_niệm", "là_một_phần_của", "khái_niệm"),
        ("sản_phẩm", "thuộc_về", "tổ_chức"),
        ("sản_phẩm", "sử_dụng", "công_nghệ"),
        ("công_nghệ", "được_sử_dụng_trong", "sản_phẩm"),
        ("công_nghệ", "phát_triển_bởi", "tổ_chức"),
        ("phương_pháp", "áp_dụng_cho", "khái_niệm"),
        ("phương_pháp", "được_sử_dụng_trong", "dịch_vụ"),
        ("dịch_vụ", "cung_cấp_bởi", "tổ_chức"),
        ("tài_liệu", "mô_tả", "khái_niệm"),
        ("số_liệu", "thể_hiện", "khái_niệm")
    ]
    
    # === Processing Configuration ===
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    MAX_KEYWORDS = int(os.getenv("MAX_KEYWORDS", "15"))
    MAX_ENTITIES = int(os.getenv("MAX_ENTITIES", "20"))
    
    # === Search Configuration ===
    SEARCH_WEIGHTS = {
        "vector": float(os.getenv("VECTOR_WEIGHT", "0.6")),
        "knowledge_graph": float(os.getenv("KG_WEIGHT", "0.4"))
    }
    
    # Search method configurations
    SEARCH_METHODS = ["vector", "knowledge_graph", "hybrid"]
    DEFAULT_SEARCH_METHOD = os.getenv("DEFAULT_SEARCH_METHOD", "hybrid")
    
    # === Logging Configuration ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG_FILE = os.getenv("LOG_FILE", "video_kg.log")
    
    # === Performance Configuration ===
    CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
    PARALLEL_PROCESSING = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    # === Validation and Error Handling ===
    STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required API keys
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # Check Neo4j configuration
        if not all([cls.NEO4J_URI, cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD]):
            errors.append("Neo4j configuration (URI, USERNAME, PASSWORD) is required")
        
        # Check paths
        try:
            Path(cls.OUTPUT_FILES_DIR).mkdir(parents=True, exist_ok=True)
            Path(cls.LOG_DIR).mkdir(parents=True, exist_ok=True)
            Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directories: {e}")
        
        # Check vector dimension
        if cls.VECTOR_DIMENSION not in [384, 768, 1024]:
            errors.append("VECTOR_DIMENSION should be 384, 768, or 1024")
        
        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
        
        return True
    
    @classmethod
    def get_embedding_model_config(cls, model_type: str = "multilingual") -> dict:
        """Get embedding model configuration"""
        return {
            "model_name": cls.EMBEDDING_MODELS.get(model_type, cls.VIETNAMESE_EMBEDDING_MODEL),
            "dimension": 768 if "mpnet" in cls.EMBEDDING_MODELS.get(model_type, "") else 384,
            "cache": cls.CACHE_EMBEDDINGS
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("=== Vietnamese Video Knowledge Graph Configuration ===")
        print(f"Neo4j URI: {cls.NEO4J_URI}")
        print(f"Database: {cls.NEO4J_DATABASE}")
        print(f"OpenAI Model: {cls.OPENAI_MODEL}")
        print(f"Embedding Model: {cls.VIETNAMESE_EMBEDDING_MODEL}")
        print(f"Vector Dimension: {cls.VECTOR_DIMENSION}")
        print(f"Output Directory: {cls.OUTPUT_FILES_DIR}")
        print(f"Default Search Method: {cls.DEFAULT_SEARCH_METHOD}")
        print(f"Search Weights: Vector={cls.SEARCH_WEIGHTS['vector']}, KG={cls.SEARCH_WEIGHTS['knowledge_graph']}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Cache Embeddings: {cls.CACHE_EMBEDDINGS}")
        print("=" * 55)


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    LOG_LEVEL = "DEBUG"
    CACHE_EMBEDDINGS = True
    STRICT_MODE = False


class ProductionConfig(Config):
    """Production environment configuration"""
    LOG_LEVEL = "INFO"
    CACHE_EMBEDDINGS = True
    STRICT_MODE = True
    PARALLEL_PROCESSING = True


class TestingConfig(Config):
    """Testing environment configuration"""
    NEO4J_DATABASE = "test_neo4j"
    LOG_LEVEL = "WARNING"
    CACHE_EMBEDDINGS = False
    STRICT_MODE = True


# Configuration factory
def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    env = env or os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)


print("config.py created successfully")
print("=" * 60)
print(config_code[:2000] + "...")