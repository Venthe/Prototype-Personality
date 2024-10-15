import time
import logging

from communication import process_communication
from knowledge import process_knowledge
from senses import process_senses

def initialize_logging():
    LOGGING_FORMAT = (
        "%(asctime)s.%(msecs)03dZ %(levelname)-5s %(process)d --- "
        "[%(name)s][%(threadName)s] : %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(
        # Set the minimum log level to DEBUG
        level=logging.INFO,
        format=LOGGING_FORMAT,
        datefmt=DATE_FORMAT,
    )
    global logger
    logger = logging.getLogger(__name__)

def run():
    logger.info("Initializing prototype assistant")
    knowledge_message_callback, text_to_speech_get = process_knowledge()
    process_senses(knowledge_message_callback)
    process_communication(text_to_speech_get)

    # Main thread should do nothing, except the communication with the UI
    try:
        logger.info("Working...")
        logger.info("Press Ctrl+C to interrupt...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted! Exiting gracefully...")

if __name__ == "__main__":
    initialize_logging()
    run()