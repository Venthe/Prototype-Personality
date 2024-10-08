import logging
import threading
import queue
import time

logger = logging.getLogger(__name__)

def process_knowledge():
    logger.info("Starting knowledge processing thread")
    knowledge_message_queue = queue.Queue()
    text_to_speech_queue = queue.Queue()

    def setup_personality():
        personality = Personality()
        while True:
            message = knowledge_message_queue.get()
            personality.listen(message)

    personality_thread = threading.Thread(target=setup_personality, name="Personality")
    personality_thread.daemon = True
    personality_thread.start()

    def trigger_tts():
        i = 0
        while True:
            text_to_speech_queue.put(f"This is a random text {i}")
            i = i + 1
            time.sleep(4)

    personality_thread = threading.Thread(target=trigger_tts, name="Personality-TTS")
    personality_thread.daemon = True
    personality_thread.start()
    
    return knowledge_message_queue.put, text_to_speech_queue.get

class Personality:
    def __init__(self):
        """"""
     
    def listen(self, message):
        if message.type == "speech":
            filtered_texts = [item.text for item in message.data if item.probability < 0.15]    
            print(filtered_texts)