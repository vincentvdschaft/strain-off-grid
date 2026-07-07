import logging
from contextlib import contextmanager


@contextmanager
def suppress_logs(level=logging.CRITICAL):
    previous = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(previous)
