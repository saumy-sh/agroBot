from espnet2.bin.asr_inference import Speech2Text
import torch
import time
import librosa
import numpy as np

device = 'cpu'
print(device)

speech2text = Speech2Text(
    asr_train_config="/kaggle/input/shrutam-model/shrutam_model/Branchformer1024/config.yaml",
    asr_model_file="/kaggle/input/shrutam-model/shrutam_model/Branchformer1024/model.pth",
    device=device,
    beam_size=5
)




def shrutam_transcriber(audio_path):
    wav, sr = librosa.load(audio_path, sr=None, mono=True)      
    if sr != 16000:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
    # Ensure it's 2D [1, samples] for most STT models
    # wav = np.expand_dims(wav, axis=0)
    text = speech2text(wav)[0][0]
    # text = conformer_model.transcribe([wav], batch_size=1,logprobs=False, language_id=lang_id)
    return text