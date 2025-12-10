# %%
import sys
import logging

import os
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

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

    _LOGGER.info(f"Found {len(projects)} PEPs.")

    # we need to work in batches since its much faster
    projects_encoded = []
    for i, batch in enumerate(
        tqdm(batch_generator(projects, batch_size), total=len(projects) // batch_size)
    ):
        # build list of descriptions for batch
        points = []
        for p in batch:
            description = markdown_to_text(p.description)
            dense_text = mine_metadata_from_dict(
                p.config, name=p.name, description=description
            )
            sparse_text = f"{p.name}. {description}"

            embeddings_list = list(dense_encoder.embed(dense_text))
            sparse_result = sparce_encoder.encode(sparse_text).coalesce()

            sparse_embeddings = models.SparseVector(
                indices=sparse_result.indices().tolist()[0],
                values=sparse_result.values().tolist(),
            )

            points.append(
                PointStruct(
                    id=p.id,
                    vector={
                        "dense": list(embeddings_list[0]),
                        "sparse": sparse_embeddings,
                    },
                    payload={
                        "description": sparse_text,
                        "registry": f"{p.namespace}/{p.name}:{p.tag}",
                        "private": p.private,
                        "name": p.name,
                    },
                )
            )

        operation_info = qdrant.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(operation_info)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")
        sys.exit(1)
