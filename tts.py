import wave
from piper import PiperVoice

voice = PiperVoice.load("hi_speakers/hi_IN-pratham-medium.onnx")

def tts(text_hi):
    with wave.open("output/tts_output.wav", "wb") as wav_file:
        voice.synthesize_wav(f"{text_hi}", wav_file)
    return
    