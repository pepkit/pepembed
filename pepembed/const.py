from logging import CRITICAL, DEBUG, ERROR, INFO, WARN
from platform import python_version

PKG_NAME = "pepembed"

LEVEL_BY_VERBOSITY = [ERROR, CRITICAL, WARN, INFO, DEBUG]
LOGGING_LEVEL = "INFO"

QDRANT_DEFAULT_HOST = "localhost"
QDRANT_DEFAULT_PORT = 6333
QDRANT_DEFAULT_COLLECTION = "pephub"

VERSIONS = {
    "python_version": python_version(),
}

DEFAULT_KEYWORDS = [
    "summary",
    "title",
    "cell",
    "protocol",
    "processing",
    "source",
    "design",
    "organism",
]

DENSE_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_ENCODER_MODEL = "prithivida/Splade_PP_en_v1"
MIN_DESCRIPTION_LENGTH = 5

DEFAULT_BATCH_SIZE = 800

REQUIRED_ENV_VARS = [
    "POSTGRES_HOST",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_PORT",
    "QDRANT_HOST",
    "QDRANT_API_KEY",
    "HF_TOKEN",
]
