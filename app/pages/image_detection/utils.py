import sys
import logging
import pprint



def configure_logger(name=__name__):
    """Configure logging for proper output to DataDog
    """
    logging.getLogger().handlers = []
    logger = logging.getLogger(__name__)
    logger.handlers = []
    log_handler = logging.StreamHandler()
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
    logger.propogate = False
    return logger

def check_is_subset(expected_subset:list, superset:bool) -> bool:
    return set(expected_subset).issubset(superset)