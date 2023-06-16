import sys
import logging
import pprint

# from pythonjsonlogger import jsonlogger
# import click


def configure_logger(name=__name__):
    """Configure logging for proper output to DataDog
    """
    logging.getLogger().handlers = []
    logger = logging.getLogger(__name__)
    logger.handlers = []
    log_handler = logging.StreamHandler()
    # formatter = jsonlogger.JsonFormatter(
    #     '%(asctime)s %(name)s %(levelname)s %(message)s %(lineno)s %(filename)s %(funcName)s'
    # )
    # log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
    logger.propogate = False
    return logger


# def message(msg,logger,**kwargs):
#     """Print output without formatting if interactive CLI use. Otherwise log in JSON format.
#     """
#     click.secho(msg,**kwargs) if sys.stdin.isatty() else logger.info(pprint.pformat(msg))

def check_is_subset(expected_subset:list, superset:bool) -> bool:
    return set(expected_subset).issubset(superset)