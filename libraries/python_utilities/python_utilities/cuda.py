import logging
import torch

def detect_cuda():
    logger = logging.getLogger(__name__)
    if torch == None:
        return False

    if torch.cuda.is_available():
        logger.debug("CUDA is available.")
        logger.debug(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logger.debug("CUDA is not available.")
        return False