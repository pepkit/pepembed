import sys
import logging
import os
import psycopg2

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
from dotenv import load_dotenv
from logmuse import init_logger
from argparse import Namespace

from pepembed.const import (
    LEVEL_BY_VERBOSITY,
    LOGGING_LEVEL,
    PKG_NAME,
    PROJECT_TABLE,
    CONFIG_COLUMN,
    PROJECT_NAME_COLUMN,
    NAMESPACE_COLUMN,
    TAG_COLUMN,
    ROW_ID_COLUMN,
    DEFAULT_BATCH_SIZE,
    QDRANT_DEFAULT_COLLECTION,
    DEFAULT_UPSERT_BATCH_SIZE,
)
from pepembed.pepembed import PEPEncoder
from pepembed.utils import batch_generator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

load_dotenv()
args = Namespace(
    postgres_user=os.environ.get("POSTGRES_USER"),
    postgres_password=os.environ.get("POSTGRES_PASSWORD"),
    postgres_host=os.environ.get("POSTGRES_HOST"),
    postgres_db=os.environ.get("POSTGRES_DB"),
    postgres_port=os.environ.get("POSTGRES_PORT"),
    qdrant_host=os.environ.get("QDRANT_HOST"),
    qdrant_port=os.environ.get("QDRANT_PORT"),
    qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
    qdrant_collection=os.environ.get("QDRANT_COLLECTION"),
    dbg=False,
    verbosity=None,
    logging_level=None,
    recreate_collection=True,
    hf_model=os.environ.get("HF_MODEL"),
    keywords_file="keywords.txt",
    batch_size=DEFAULT_BATCH_SIZE,
    upsert_batch_size=DEFAULT_UPSERT_BATCH_SIZE,
)

# Set the logging level.
if args.dbg:
    # Debug mode takes precedence and will listen for all messages.
    level = args.logging_level or logging.DEBUG
elif args.verbosity is not None:
    # Verbosity-framed specification trumps logging_level.
    level = LEVEL_BY_VERBOSITY[args.verbosity]
else:
    # Normally, we're not in debug mode, and there's not verbosity.
    level = LOGGING_LEVEL

# initialize the logger
logger_kwargs = {"level": level, "devmode": args.dbg}
init_logger(name="peppy", **logger_kwargs)
global _LOGGER
_LOGGER = init_logger(name=PKG_NAME, **logger_kwargs)

# pull list of peps
_LOGGER.info("Establishing connection to database.")
conn = psycopg2.connect(
    user=(args.postgres_user or os.environ.get("POSTGRES_USER")),
    password=(args.postgres_password or os.environ.get("POSTGRES_PASSWORD")),
    host=(args.postgres_host or os.environ.get("POSTGRES_HOST")),
    database=(args.postgres_db or os.environ.get("POSTGRES_DB")),
    port=(args.postgres_port or 5432),
)
curs = conn.cursor()

# test connection
_LOGGER.info("Testing connection to database.")
curs.execute("SELECT 1")
res = curs.fetchone()
if not res == (1,):
    _LOGGER.error("Connection to database failed.")
    sys.exit(1)

# get list of peps
_LOGGER.info("Pulling PEPs from database.")
curs.execute(
    f"SELECT {NAMESPACE_COLUMN}, {PROJECT_NAME_COLUMN}, {TAG_COLUMN}, {CONFIG_COLUMN}, {ROW_ID_COLUMN} FROM {PROJECT_TABLE}"
)
projects = curs.fetchall()

# map list of tuples to list of dicts
_LOGGER.info(f"Found {len(projects)} PEPs.")


# initialize encoder
_LOGGER.info("Initializing encoder.")
encoder = PEPEncoder(args.hf_model, keywords_file=args.keywords_file)
EMBEDDING_DIM = 384 # hardcoded for sentence-transformers/all-MiniLM-L12-v2 and BAAI/bge-small-en-v1.5
_LOGGER.info(f"Computing embeddings of {EMBEDDING_DIM} dimensions.")

# encode PEPs in batches
_LOGGER.info("Encoding PEPs.")
BATCH_SIZE = args.batch_size or DEFAULT_BATCH_SIZE

# we need to work in batches since its much faster
projects_encoded = []
i = 0
for batch in tqdm(
    batch_generator(projects, BATCH_SIZE), total=(len(projects) // BATCH_SIZE)
):
    # build list of descriptions for batch
    descs = []
    for p in batch:
        d = encoder.mine_metadata_from_dict(p[3], min_desc_length=20)
        if d != "" or d is None:
            descs.append(d)
        else:
            descs.append(f"{p[0]} {p[1]} {p[2]}")
    # every 100th batch, print out the first description
    if i % 100 == 0:
        _LOGGER.info(f"First description: {descs[0]}")
    # encode descriptions
    try:
        embeddings = encoder.encode(descs)
        projects_encoded.extend(
            [
                dict(
                    id=p[4],
                    registry=f"{p[0]}/{p[1]}:{p[2]}",
                    description=desc,
                    embedding=embd,
                )
                for p, desc, embd in zip(batch, descs, embeddings)
            ]
        )
    except Exception as e:
        _LOGGER.error(f"Error encoding batch: {e}")
    i += 1

_LOGGER.info("Encoding complete.")
_LOGGER.info("Connecting to Qdrant.")

# get the qdrant connection info
QDRANT_HOST = args.qdrant_host or os.environ.get("QDRANT_HOST")
QDRANT_PORT = args.qdrant_port or os.environ.get("QDRANT_PORT")
QDRANT_API_KEY = args.qdrant_api_key or os.environ.get("QDRANT_API_KEY")

# connect to qdrant
qdrant = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
)

# get the collection info
COLLECTION = (
    args.qdrant_collection
    or os.environ.get("QDRANT_COLLECTION")
    or QDRANT_DEFAULT_COLLECTION
)

# recreate the collection if necessary
if args.recreate_collection:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
        on_disk_payload=True,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
    )
    collection_info = qdrant.get_collection(collection_name=COLLECTION)
else:
    try:
        collection_info = qdrant.get_collection(collection_name=COLLECTION)
    except Exception as e:
        _LOGGER.error(
            f"Error getting collection info. Collection {COLLECTION} might not exist."
        )
        _LOGGER.info("Recreating collection.")
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
            on_disk_payload=True,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        collection_info = qdrant.get_collection(collection_name=COLLECTION)

# verify status of collection after getting or creating
_LOGGER.info(f"Collection status: {collection_info.status}")

# insert embeddings into qdrant
_LOGGER.info("Inserting embeddings into Qdrant.")
_LOGGER.info("Building point strcutures.")

# build up point structs
all_points = [
    PointStruct(
        id=p["id"],
        vector=p["embedding"].tolist(),
        payload={"registry": p["registry"], "description": p["description"]},
    )
    for p in tqdm(projects_encoded, total=len(projects_encoded))
]

# determine upsert batch size
UPSERT_BATCH_SIZE = args.upsert_batch_size or DEFAULT_UPSERT_BATCH_SIZE

# upsert in batches, it will timeout if we do not
# a good batch size is ~1000 vectors. Running locally, this is super quick.
for batch in tqdm(
    batch_generator(all_points, UPSERT_BATCH_SIZE),
    total=len(all_points) // UPSERT_BATCH_SIZE,
):
    operation_info = qdrant.upsert(collection_name=COLLECTION, wait=True, points=batch)

    assert operation_info.status == "completed"

conn.close()

_LOGGER.info("Done.")
_LOGGER.info(
    f"View the collection at https://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}"
)
_LOGGER.info(
    f"""View some points and their paylods with the following curl command:
    curl -H "Content-type: application/json" -d '{{
        "ids": [0, 3, 100]
    }}' 'http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points'
"""
)
