import os
import torch
import librosa
import soundfile as sf
from dataset import load_ljspeech_dataset, load_speechcommand_dataset
from model import WaveNet_Mel2Raw
import matplotlib.pyplot as plt



def get_melspectrum(dataset, idx):
    filename = dataset.files[idx]
    y, sr = librosa.load(filename)
    y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate)
    mels = torch.tensor(dataset._compute_mel_spec(y))
    return mels

def inference_speech(model, mels, device):
    mels = mels.to(device)
    model.eval()
    return model.inference(mels, device)


#
from config import Config
config = Config()

result_dir = config.gen_result_folder
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

    
##
# Target Audio
_, testset = load_ljspeech_dataset(config)

# Target audio
import IPython.display as ipd
import librosa

y, sr = librosa.load(testset.files[config.gen_test_idx]); print(len(y)/sr)
y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate)
ipd.Audio(y, rate=config.sample_rate)


##
# Model
model = WaveNet_Mel2Raw.create_with(config)
model.load_checkpoint(config.gen_model_cp)
model.to(device)
model.eval()


##
# Inference
mels = get_melspectrum(testset, config.gen_test_idx)
y_hat = inference_speech(model, mels, device)


##
# result
plt.plot(y_hat.cpu()[0,:])
ipd.Audio(y_hat.cpu()[0,:], rate= config.sample_rate)

##
# save to file
import time
sf.write(os.path.join(config.gen_result_folder, f'TEST_{config.gen_test_idx}_{int(time.time())}.wav'), y_hat.T.cpu().numpy(), config.sample_rate, format='WAV')
