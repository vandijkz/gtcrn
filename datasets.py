import torch
from torch.utils import data
import os
import librosa
import numpy as np
import random as random
import soundfile as sf

class VoiceDataset(data.Dataset):
    """Dataset class (custom) for VCTK+DEMAND speech enhancement tasks.
    
    Args:
        clean_folder : The folder containing the clean wav files.
        noisy_folder : The folder containing the noisy wav files.
        task : Please ignore this arg. Reamaning this arg may be for compatibility with other scripts.
        sample_rate (int) : The sample rate of the sources and mixtures.
        segment : The time length (s) of input speech.
        n_src (int) : The number of sources in the mixture.

    Returns:
        noisy_spec, clean_spec
        
    References
        [1] Valentini-Botinhao, Cassia. (2017). 
        Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. 
        University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR).
    """

    dataset_name = "VCTK_DEMAND"
    
    def __init__(self,  clean_folder, noisy_folder, task="enh_single", sample_rate=16000, segment=3, n_src=1):
        self.clean_folder = clean_folder
        self.noisy_folder = noisy_folder
        self.file_list = os.listdir(clean_folder)
        self.task = task
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        if self.segment is not None:
            self.seg_len = int(self.segment * self.sample_rate)
        else:
            self.seg_len = None

    def __getitem__(self, index):
        clean_file = os.path.join(self.clean_folder, self.file_list[index])
        noisy_file = os.path.join(self.noisy_folder, self.file_list[index])
        self.clean_file = clean_file
        self.noisy_file = noisy_file

        clean_data, _ = librosa.load(clean_file, sr=self.sample_rate)
        
        if self.seg_len is not None:
            if len(clean_data) >= self.seg_len:
                start = random.randint(0, len(clean_data) - self.seg_len)
                stop = start + self.seg_len
                clean_data, _ = sf.read(clean_file, dtype="float32", start=start, stop=stop)
                noisy_data, _ = sf.read(noisy_file, dtype="float32", start=start, stop=stop)
            else:
                noisy_data, _ = librosa.load(noisy_file, sr=self.sample_rate)
                padding_length = self.seg_len - len(clean_data)
                padding = np.zeros(padding_length, dtype=np.float32)
                clean_data = np.concatenate((clean_data, padding))
                noisy_data = np.concatenate((noisy_data, padding))
        else:
            noisy_data, _ = librosa.load(noisy_file, sr=self.sample_rate)

        # 将音频数据转换为频谱图，并分离实部和虚部
        def transform_waveform_to_spectrogram(waveform):
            stft = librosa.stft(waveform, n_fft=512, hop_length=256)
            spectrogram = np.stack((stft.real, stft.imag), axis=-1)
            return torch.from_numpy(spectrogram).permute(2, 1, 0)

        noisy_spec = transform_waveform_to_spectrogram(noisy_data)
        clean_spec = transform_waveform_to_spectrogram(clean_data)

        return noisy_spec, clean_spec

    def __len__(self):
        return len(self.file_list)
    
    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self._dataset_name()
        infos["task"] = self.task
        return infos

    def _dataset_name(self):
        return "VCTK_DEMAND"


if __name__ == '__main__':
    ds = VoiceDataset('/data0/zhanghaoyi/gtcrn/train/data/clean_trainset_28spk_wav_16k/',
                      '/data0/zhanghaoyi/gtcrn/train/data/noisy_trainset_28spk_wav_16k/')
    len_list = []

    for i in range(ds.__len__()):
        clean, noisy = ds.__getitem__(i)
        length = len(clean)
        len_list.append(length)
    
    print(np.mean(len_list))
