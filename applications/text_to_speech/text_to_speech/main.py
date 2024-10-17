import os
from datetime import datetime
import time
from flask import Flask, request, send_file


app = Flask(__name__)
speaker_id, color_convert, tts_model = prepare_model()

def convert(text_to_voice):
    print(f"TTS: {text_to_voice}")

    start_time = time.time()
    current_timestamp = datetime.now().timestamp()
    base_output_file = f"{config['temp_dir']}/{current_timestamp}.base.wav"

    tts_model.tts_to_file(text_to_voice, speaker_id, base_output_file, speed=config['speed'])
    tone_mapped_output_file = f"{config['temp_dir']}/{current_timestamp}.tone-mapped.wav"
    color_convert(input=base_output_file, output=tone_mapped_output_file)

    print(f"Text converted to sound in {time.time() - start_time}s")
    mp3 = convert_wav_to_mp3_memory(tone_mapped_output_file)
    os.remove(tone_mapped_output_file)
    os.remove(base_output_file)
    return mp3

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    app.logger.info("Received request for text-to-speech")
    data = request.get_data(as_text=True)

    result = convert(data)

    return send_file(result, mimetype='audio/mpeg', as_attachment=True, download_name='output.mp3')

if __name__ == '__main__':
    print("Starting server at 0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)