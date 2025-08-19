import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import librosa
import numpy as np


model_path = "models/indicconformer_stt_hi_hybrid_rnnt_large.nemo"
lang_id = "hi"

device = "cpu"
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval() # inference mode
model = model.to(device)
model.cur_decoder = "rnnt"

def stt(audio_path):
    wav, rate = librosa.load(audio_path, sr=16000, mono=True)
    wav = wav.astype(np.float32)

    # Give it as a list of numpy arrays
    return model.transcribe([wav], batch_size=1, logprobs=False, language_id=lang_id)[0][0]