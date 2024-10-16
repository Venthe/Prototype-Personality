import logging


def setup_logging(log_level=logging.INFO, log_path=None, logger=logging.getLogger()):

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)-5s %(process)d --- [%(name)s][%(threadName)s] : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path != None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
