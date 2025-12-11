import logging
import sys

from .argparser import app
from .const import PKG_NAME

_LOGGER = logging.getLogger(name=PKG_NAME)
_LOGGER.setLevel(logging.INFO)


def main():
    app(prog_name=PKG_NAME)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
