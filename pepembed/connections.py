import logging
import os
from typing import Union

from fastembed import TextEmbedding
from pepdbagent import PEPDatabaseAgent
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SparseEncoder

from .const import (
    PKG_NAME,
    QDRANT_DEFAULT_COLLECTION,
    QDRANT_DEFAULT_HOST,
    QDRANT_DEFAULT_PORT,
)

_LOGGER = logging.getLogger(PKG_NAME)


def get_qdrant(
    collection_name=QDRANT_DEFAULT_COLLECTION,
    recreate_collection: bool = False,
    embedding_dim: Union[None, int] = None,
) -> QdrantClient:
    """
    Get a Qdrant client.

    :Args:
        recreate_collection: Whether to recreate the collection if it does not exist. [default: False]
        embedding_dim: The embedding dimension to use, for recreation of the collection [default: None]
    :Returns:
        QdrantClient: The Qdrant client.
    """
    _LOGGER.info("Connecting to Qdrant.")

    q_host = os.environ.get("QDRANT_HOST", QDRANT_DEFAULT_HOST)
    q_port = os.environ.get("QDRANT_PORT", QDRANT_DEFAULT_PORT)
    q_api_key = os.environ.get("QDRANT_API_KEY", None)

    qdrant = QdrantClient(
        url=q_host,
        port=q_port,
        api_key=q_api_key,
    )

    collection_exist = qdrant.collection_exists(collection_name=collection_name)

    if not collection_exist and not recreate_collection:
        _LOGGER.error(
            f"Collection {collection_name} does not exist, and recreate_collection is False."
            f" Please set recreate_collection to True to create the collection."
        )
        exit(1)
    elif not collection_exist and recreate_collection:
        _LOGGER.info(
            f"Collection {collection_name} does not exist. Creating collection."
        )

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedding_dim, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                ),
            ),
            on_disk_payload=True,
        )
    qdrant.create_payload_index(
        collection_name=collection_name,
        field_name="name",
        field_type=models.PayloadSchemaType.KEYWORD,
    )

    collection_info = qdrant.get_collection(collection_name=collection_name)

    _LOGGER.info(f"Collection status: {collection_info.status}")
    return qdrant


def get_db_agent() -> PEPDatabaseAgent:
    """
    Get the database connection string from environment variables.

    :Returns:
        str: The database connection string.
    """

    agent = PEPDatabaseAgent(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "password"),
        database=os.environ.get("POSTGRES_DB", "pepdb"),
    )

    return agent


def get_sparce_model(sparce_model: str) -> Union[None, SparseEncoder]:
    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        return None
    sparse_model = SparseEncoder(sparce_model, token=token)
    return sparse_model


def get_dense_model(dense_model: str) -> Union[None, TextEmbedding]:
    return TextEmbedding(dense_model)
