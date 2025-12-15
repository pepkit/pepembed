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
    """Get a Qdrant client.

    Args:
        collection_name: Name of the Qdrant collection.
        recreate_collection: Whether to recreate the collection if it does not exist.
        embedding_dim: The embedding dimension to use for recreation of the collection.

    Returns:
        The Qdrant client instance.
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
    """Get the database connection string from environment variables.

    Returns:
        The PEP database agent instance.
    """

    agent = PEPDatabaseAgent(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "password"),
        database=os.environ.get("POSTGRES_DB", "pepdb"),
    )

    return agent


def get_sparse_model(sparse_model: str) -> Union[None, SparseEncoder]:
    """Get a sparse encoder model.

    Args:
        sparse_model: Name of the sparse encoder model.

    Returns:
        Sparse encoder instance, or None if HF_TOKEN is not set.
    """
    # token = os.environ.get("HF_TOKEN", None)
    # if token is None:
    #     return None
    _LOGGER.info(f"Initializing sparse model: {sparse_model}")
    sparse_model = SparseEncoder(sparse_model)
    return sparse_model


def get_dense_model(dense_model: str) -> Union[None, TextEmbedding]:
    """Get a dense encoder model.

    Args:
        dense_model: Name of the dense encoder model.

    Returns:
        Text embedding instance.
    """
    _LOGGER.info(f"Initializing dense model: {dense_model}")
    return TextEmbedding(dense_model)
