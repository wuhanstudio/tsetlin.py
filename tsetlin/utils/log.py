import sys

# Choose whether to use log or print
USE_LOGGER = True

if sys.implementation.name == 'micropython':
    USE_LOGGER = False

if USE_LOGGER:
    from loguru import logger
    log = logger.info
else:
    log = print
