import logging
import threading
import io
import time
import httpx
import pygame

logger = logging.getLogger(__name__)

def process_communication(text_to_speech_get):
    logger.info("Starting communication processing thread")
    setup_play_thread(text_to_speech_get)

def setup_play_thread(text_to_speech_get):
    print("Starting TTS listener")

    def play():
        print("Starting TTS loop")
        dlq = None
        dlq2 = None
        while True:
            data = None
            if dlq is not None:
                data = dlq
                dlq = None
            else:
                data = text_to_speech_get()
            try:
                response = httpx.post("http://localhost:5001/text-to-speech", data=data)
                response.raise_for_status()
                mp3_file = io.BytesIO(response.content)

                # Initialize the mixer
                pygame.mixer.init()
                
                # Load the MP3 file from memory
                pygame.mixer.music.load(mp3_file, 'mp3')
                
                # Play the MP3 file
                pygame.mixer.music.play()
                
                # Wait until the music finishes playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
            except Exception as e:
                print(e)
                if dlq2 == None and dlq != None:
                    dlq2 = data
                    dlq = data
                    time.sleep(10)
                elif dlq2 != None and dlq != None:
                    dlq2 = None
                    dlq = None
                    logger.warn("Dropping")
                else:
                    dlq = data
                    time.sleep(5)

    thread = threading.Thread(target=play, name="TTS")
    thread.daemon = True
    thread.start()