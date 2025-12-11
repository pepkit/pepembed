import logging
import os
from typing import Optional

import typer
from dotenv import load_dotenv

from ._version import __version__ as pepembed_version
from .const import (
    DEFAULT_BATCH_SIZE,
    DENSE_ENCODER_MODEL,
    PKG_NAME,
    QDRANT_DEFAULT_COLLECTION,
    SPARSE_ENCODER_MODEL,
)

_LOGGER = logging.getLogger(PKG_NAME)

app = typer.Typer(
    name=PKG_NAME,
    help="Run embedding on PEPs",
    epilog="pephub.databio.org",
    add_completion=False,
)


def build_argparser():
    """
    Build and return the typer app for CLI argument parsing.
    This function maintains compatibility with the original argparse interface.
    """
    return app


def version_callback(value: bool):
    if value:
        typer.echo(f"pepembed version: {pepembed_version}")
        raise typer.Exit()


@app.command()
def main(
    qdrant_collection: Optional[str] = typer.Option(
        None,
        help="Qdrant collection name",
    ),
    recreate_collection: bool = typer.Option(
        True,
        help="Recreate collection if it exists",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        help="Batch size for embedding",
    ),
    dense_model: Optional[str] = typer.Option(
        None,
        help="HuggingFace dense encoder model",
    ),
    sparse_model: Optional[str] = typer.Option(
        None,
        help="HuggingFace sparse encoder model",
    ),
    env_var: Optional[str] = typer.Option(
        None,
        help="Path to .env file, if not set, will not load any .env file",
    ),
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, help="App version"
    ),
):
    """Run embedding on PEPs"""
    # Import here to avoid circular imports
    from .pepembed import pepembed

    if env_var:
        load_dotenv(dotenv_path=env_var)

    collection_name = qdrant_collection or os.environ.get(
        "QDRANT_COLLECTION", QDRANT_DEFAULT_COLLECTION
    )
    hf_model_dense = dense_model or os.environ.get(
        "HF_MODEL_DENSE", DENSE_ENCODER_MODEL
    )
    hf_model_sparse = sparse_model or os.environ.get(
        "HF_MODEL_SPARSE", SPARSE_ENCODER_MODEL
    )

    pepembed(
        batch_size=batch_size,
        recreate_collection=recreate_collection,
        collection_name=collection_name,
        hf_model_dense=hf_model_dense,
        hf_model_sparse=hf_model_sparse,
    )


if __name__ == "__main__":
    app()
