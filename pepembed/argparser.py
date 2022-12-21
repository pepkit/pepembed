from ubiquerg import VersionInHelpParser

from . import __version__
from .const import *
from ._version import __version__ as pepembed_version


def build_argparser():
    banner = "%(prog)s - Run embedding on PEPs"
    additional_description = "pephub.databio.org"

    parser = VersionInHelpParser(
        prog=PKG_NAME,
        description=banner,
        epilog=additional_description,
        version=pepembed_version,
    )

    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        choices=range(len(LEVEL_BY_VERBOSITY)),
        help="Choose level of verbosity (default: %(default)s)",
    )

    parser.add_argument(
        "--dbg",
        dest="dbg",
        action="store_true",
        help="Enable debug mode (default: %(default)s)",
    )

    parser.add_argument(
        "-m",
        "--hf-model",
        dest="hf_model",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help="Huggingface model registry (default: %(default)s)",
    )

    parser.add_argument(
        "--keywords-file",
        dest="keywords_file",
        default=None,
        help="File containing keywords to search for (default: %(default)s)",
    )

    parser.add_argument(
        "--postgres-host",
        dest="postgres_host",
        default=None,
        help="Postgres host (default: %(default)s)",
    )

    parser.add_argument(
        "--postgres-port",
        dest="postgres_port",
        default=5432,
        help="Postgres port (default: %(default)s)",
    )

    parser.add_argument(
        "--postgres-user",
        dest="postgres_user",
        default=None,
        help="Postgres user (default: %(default)s)",
    )

    parser.add_argument(
        "--postgres-password",
        dest="postgres_password",
        default=None,
        help="Postgres password (default: %(default)s)",
    )

    parser.add_argument(
        "--postgres-db",
        dest="postgres_db",
        default=None,
        help="Postgres database (default: %(default)s)",
    )

    parser.add_argument(
        "--qdrant-host",
        dest="qdrant_host",
        default=None,
        help="Qdrant host (default: %(default)s)",
    )

    parser.add_argument(
        "--qdrant-port",
        dest="qdrant_port",
        default=None,
        help="Qdrant port (default: %(default)s)",
    )

    parser.add_argument(
        "--qdrant-collection",
        dest="qdrant_collection",
        default=None,
        help="Qdrant collection name (default: %(default)s)",
    )

    parser.add_argument(
        "--recreate-collection",
        dest="recreate_collection",
        action="store_true",
        help="Recreate collection if it exists (default: %(default)s)",
    )

    parser.add_argument(
        "--qdrant-api-key",
        dest="qdrant_api_key",
        default=None,
        help="Qdrant API key (default: %(default)s)",
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=100,
        help="Batch size for embedding (default: %(default)s)",
    )

    parser.add_argument(
        "--upsert-batch-size",
        dest="upsert_batch_size",
        default=1000,
        help="Batch size for upserting embeddings into qdrant (default: %(default)s)",
    )

    return parser
