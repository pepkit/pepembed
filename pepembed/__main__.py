import logging
import sys
import coloredlogs

from .argparser import app
from .const import LOGGING_LEVEL, PKG_NAME

_LOGGER = logging.getLogger(name=PKG_NAME)
_LOGGER.propagate = False
coloredlogs.install(
    logger=_LOGGER,
    level=LOGGING_LEVEL,
    datefmt="%H:%M:%S",
    fmt="[%(levelname)s] [%(asctime)s] [PEPEMBED] %(message)s",
)


def main():
    app(prog_name=PKG_NAME)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
