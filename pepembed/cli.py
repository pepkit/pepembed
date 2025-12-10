# %%
import sys
import logging

import os
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from logmuse import init_logger
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from pepdbagent.db_utils import Projects
from dotenv import load_dotenv

from .const import (
    LOGGING_LEVEL,
    PKG_NAME,
    DEFAULT_BATCH_SIZE,
    QDRANT_DEFAULT_COLLECTION,
    DENSE_ENCODER_MODEL,
    SPARSE_ENCODER_MODEL,
)
from .argparser import build_argparser

from .utils import batch_generator, markdown_to_text, mine_metadata_from_dict
from .connections import get_db_agent, get_qdrant, get_sparce_model, get_dense_model
from .id_tracker import IDTracker


_LOGGER = init_logger(name=PKG_NAME, level=LOGGING_LEVEL)


# %%
def main():
    """Entry point for the CLI."""
    load_dotenv()
    # parser = build_argparser()
    # args, _ = parser.parse_known_args()
    #
    # batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    # recreate_collection = args.recreate_collection or False

    batch_size = DEFAULT_BATCH_SIZE
    recreate_collection = True
    collection_name = os.environ.get("QDRANT_COLLECTION", QDRANT_DEFAULT_COLLECTION)

    agent = get_db_agent()
    hf_model_dense = os.environ.get("HF_MODEL_DENSE", DENSE_ENCODER_MODEL)
    hf_model_sparse = os.environ.get("HF_MODEL_SPARSE", SPARSE_ENCODER_MODEL)

    dense_encoder = get_dense_model(hf_model_dense)
    sparce_encoder = get_sparce_model(hf_model_sparse)

    embedding_dimensions = int(dense_encoder.get_embedding_size(hf_model_dense))

    qdrant = get_qdrant(
        recreate_collection=recreate_collection, embedding_dim=embedding_dimensions
    )

    # Initialize ID tracker
    id_tracker = IDTracker()
    tracker_stats = id_tracker.get_stats()
    _LOGGER.info(f"ID Tracker initialized: {tracker_stats['total_processed']} IDs already processed")

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
                _LOGGER.error(f"Error processing PEP {p.namespace}/{p.name}:{p.tag}: {e}")
                continue

        # Batch encode all dense texts at once
        dense_embeddings_list = list(dense_encoder.embed(dense_texts, parallel=4))

        # Batch encode all sparse texts at once
        sparse_results = sparce_encoder.encode(sparse_texts, batch_size=64, convert_to_tensor=False)

        # Second pass: create points from batch results
        points = []
        for data, dense, sparse in zip(batch_data, dense_embeddings_list, sparse_results):

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

        operation_info = qdrant.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(operation_info)

        # Mark batch as processed after successful upsert
        processed_ids = [data["id"] for data in batch_data]
        id_tracker.mark_batch_processed(processed_ids)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")
        sys.exit(1)
