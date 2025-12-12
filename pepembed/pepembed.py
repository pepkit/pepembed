# %%
import sys
from logging import getLogger

from dotenv import load_dotenv
from pepdbagent.db_utils import Projects
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from sqlalchemy import and_, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from .connections import get_db_agent, get_dense_model, get_qdrant, get_sparce_model
from .const import (
    DEFAULT_BATCH_SIZE,
    DENSE_ENCODER_MODEL,
    PKG_NAME,
    QDRANT_DEFAULT_COLLECTION,
    REQUIRED_ENV_VARS,
    SPARSE_ENCODER_MODEL,
)
from .id_tracker import IDTracker
from .utils import (
    batch_generator,
    check_env_variable,
    markdown_to_text,
    mine_metadata_from_dict,
)

_LOGGER = getLogger(name=PKG_NAME)
_LOGGER.setLevel("INFO")


# %%
def pepembed(
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate_collection: bool = True,
    collection_name: str = QDRANT_DEFAULT_COLLECTION,
    hf_model_dense: str = DENSE_ENCODER_MODEL,
    hf_model_sparse: str = SPARSE_ENCODER_MODEL,
) -> None:
    """
    Main function to embed PEPs and store them in Qdrant.

    :Args:
        batch_size: The batch size for embedding. [default: 800]
        recreate_collection: Whether to recreate the Qdrant collection. [default: True]
        collection_name: The name of the Qdrant collection. [default: pephub]
        hf_model_dense: The HuggingFace model to use for dense embeddings. [default: sentence-transformers/all-MiniLM-L6-v2]
        hf_model_sparse: The HuggingFace model to use for sparse embeddings. [default: naver/splade-v3]
    :Returns:
        None
    """
    load_dotenv()

    if all([check_env_variable(var) for var in REQUIRED_ENV_VARS]):
        _LOGGER.error("Any of required environment variables are not set. Exiting...")
        sys.exit(1)

    _LOGGER.info("Connecting to database.")
    agent = get_db_agent()

    dense_encoder = get_dense_model(hf_model_dense)
    sparce_encoder = get_sparce_model(hf_model_sparse)

    embedding_dimensions = int(dense_encoder.get_embedding_size(hf_model_dense))

    _LOGGER.info("Connecting to qdrant.")
    qdrant = get_qdrant(
        collection_name=collection_name,
        recreate_collection=recreate_collection,
        embedding_dim=embedding_dimensions,
    )

    # Initialize ID tracker
    id_tracker = IDTracker()
    tracker_stats = id_tracker.get_stats()
    _LOGGER.info(
        f"ID Tracker initialized: {tracker_stats['total_processed']} IDs already processed"
    )

    _LOGGER.info("Fetching PEPs from database.")

    with Session(agent.pep_db_engine.engine) as session:
        statement = select(
            Projects.namespace,
            Projects.name,
            Projects.tag,
            Projects.config,
            Projects.id,
            Projects.description,
            Projects.private,
        )
        # statement = statement.where(Projects.namespace == "geo").limit(10000)
        projects = session.execute(statement).all()

    _LOGGER.info(f"Found {len(projects)} PEPs from database.")

    # Filter out already processed projects
    projects = id_tracker.filter_unprocessed(projects)
    _LOGGER.info(f"After filtering: {len(projects)} PEPs to process.")

    _LOGGER.info("Starting indexing process....")
    # we need to work in batches since its much faster
    for i, batch in enumerate(
        tqdm(batch_generator(projects, batch_size), total=len(projects) // batch_size)
    ):
        # First pass: collect all texts and metadata from batch
        batch_data = []
        dense_texts = []
        sparse_texts = []

        for p in batch:
            try:
                description = markdown_to_text(p.description)
                dense_text = mine_metadata_from_dict(
                    p.config, name=p.name, description=description
                )
                sparse_text = f"{p.name}. {description}"
                batch_info = {
                    "id": p.id,
                    "sparse_text": sparse_text,
                    "namespace": p.namespace,
                    "name": p.name,
                    "tag": p.tag,
                    "private": p.private,
                }

                dense_texts.append(dense_text)
                sparse_texts.append(sparse_text)
                batch_data.append(batch_info)
            except Exception as e:
                _LOGGER.error(
                    f"Error processing PEP {p.namespace}/{p.name}:{p.tag}: {e}"
                )
                continue

        # Batch encode all dense texts at once
        dense_embeddings_list = list(dense_encoder.embed(dense_texts, parallel=4))

        # Batch encode all sparse texts at once
        sparse_results = sparce_encoder.encode(
            sparse_texts, batch_size=64, convert_to_tensor=False
        )

        # Second pass: create points from batch results
        points = []
        for data, dense, sparse in zip(
            batch_data, dense_embeddings_list, sparse_results
        ):

            sparse_col = sparse.coalesce()

            sparse_embeddings = models.SparseVector(
                indices=sparse_col.indices().tolist()[0],
                values=sparse_col.values().tolist(),
            )

            points.append(
                PointStruct(
                    id=data["id"],
                    vector={
                        "dense": list(dense),
                        "sparse": sparse_embeddings,
                    },
                    payload={
                        "description": data["sparse_text"],
                        "registry": f"{data['namespace']}/{data['name']}:{data['tag']}",
                        "private": data["private"],
                        "name": data["name"],
                    },
                )
            )
        if len(points) == 0:
            _LOGGER.info(f"No valid points to upsert in batch {i}, skipping.")
            continue
        operation_info = qdrant.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
        _LOGGER.info(f"Qdrant operation: {operation_info}")

        # Mark batch as processed after successful upsert
        processed_ids = [data["id"] for data in batch_data]
        id_tracker.mark_batch_processed(processed_ids)

    _LOGGER.info("Indexing process completed.")


if __name__ == "__main__":
    try:
        sys.exit(pepembed())
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")
        sys.exit(1)
