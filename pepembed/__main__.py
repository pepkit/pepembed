import logging
import sys
import coloredlogs

from .argparser import app
from .const import PKG_NAME

_LOGGER = logging.getLogger(name=PKG_NAME)
_LOGGER.propagate = False
coloredlogs.install(
    logger=_LOGGER,
    datefmt="%H:%M:%S",
    fmt="[%(levelname)s] [%(asctime)s] [PEPEMBED] %(message)s",
)


# Add console handler to output logs
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# _LOGGER.addHandler(handler)


def main():
    app(prog_name=PKG_NAME)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
