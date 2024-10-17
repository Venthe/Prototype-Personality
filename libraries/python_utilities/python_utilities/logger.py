import logging


def setup_logging(log_level=logging.INFO, log_path=None, logger=logging.getLogger()):
    datefmt = "%Y-%m-%dT%H:%M:%S"
    format = "%(asctime)s.%(msecs)03dZ %(levelname)-5s %(process)d --- [%(name)s][%(threadName)s] : %(message)s"

    logging.basicConfig(level=log_level, datefmt=datefmt, format=format)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(log_level)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    formatter = logging.Formatter(
        format,
        datefmt=datefmt,
    )

    if log_path != None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
