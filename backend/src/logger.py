import os
import logging

log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_level_value = os.environ.get("LOG_LEVEL", logging.INFO)

logging.basicConfig(
    level=log_level_value,
    format=log_format,
)

logger = logging.getLogger("imdb")
