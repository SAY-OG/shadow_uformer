import logging


def get_logger():
    logger = logging.getLogger("ShadowUformer")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
