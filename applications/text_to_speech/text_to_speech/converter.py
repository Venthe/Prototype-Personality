import io
import ffmpeg


def convert_wav_to_mp3_memory(wav_file_path):
    """Convert a WAV file to MP3 and output to a memory buffer."""
    mem_file = io.BytesIO()
    process = (
        ffmpeg
        .input(wav_file_path)
        .output('pipe:', format='mp3')  # Output to pipe with MP3 format
        .run(capture_stdout=True, capture_stderr=True)
    )
    mem_file.write(process[0])
    mem_file.seek(0)

    return mem_file