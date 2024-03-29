from platform import python_version
from logging import CRITICAL, DEBUG, ERROR, INFO, WARN

PKG_NAME = "pepembed"

LEVEL_BY_VERBOSITY = [ERROR, CRITICAL, WARN, INFO, DEBUG]
LOGGING_LEVEL = "INFO"

QDRANT_DEFAULT_HOST = "localhost"
QDRANT_DEFAULT_PORT = 6333
QDRANT_DEFAULT_COLLECTION = "pephub"

VERSIONS = {
    "python_version": python_version(),
}

DEFAULT_KEYWORDS = ["cell", "protocol", "description", "processing", "source"]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

MIN_DESCRIPTION_LENGTH = 5

PROJECT_TABLE = "projects"
CONFIG_COLUMN = "config"
PROJECT_NAME_COLUMN = "name"
CONFIG_COLUMN = "config"
NAMESPACE_COLUMN = "namespace"
DESCRIPTION_COLUNM = "description"
TAG_COLUMN = "tag"
ROW_ID_COLUMN = "id"

DEFAULT_BATCH_SIZE = 100
DEFAULT_UPSERT_BATCH_SIZE = 1000
