from collections import defaultdict
from os.path import join

import librosa
import torch

from pyannote.audio import Pipeline

from txtgrid_master import TextGrid_Master as tm



class SpeakerDiarization:
    def __init__(self, model_path, device = torch.device('cpu')):
        self.device = device
        self.pipeline = Pipeline.from_pretrained(join(model_path, 'config.yaml'))
        self.pipeline.to(device)
        
    def process_speech(self,
                       speech,
                       sr,
                       n_exact_speakers = 0,
                       n_min_speakers = 0,
                       n_max_speakers = 0):
        self.duration = librosa.get_duration(y=speech, sr=sr)
        speech = torch.from_numpy(speech)
        speech = speech.unsqueeze(0).to(self.device)
        audio = {"waveform": speech, "sample_rate": sr}
        if n_exact_speakers > 0: 
            self.diarization = self.pipeline(audio, num_speakers=n_exact_speakers)
        elif n_min_speakers > 0 or n_max_speakers > 0:
            self.diarization = self.pipeline(audio, min_speakers=n_min_speakers, max_speakers=n_max_speakers)
        else:
            self.diarization = self.pipeline(audio)
    
    def write_rttm(self, rttm_file):
        with open(rttm_file, "w") as rttm:
            self.diarization.write_rttm(rttm)
    
    def write_textgrid(self, textgrid_file, speech_label = 'speech'):
        dTiers = defaultdict(lambda : [[],[],[]])
        for s, t, l in self.diarization.itertracks(yield_label=True):
            dTiers[l][0].append(round(s.start,2))
            dTiers[l][1].append(round(s.end,2))
            dTiers[l][2].append(speech_label)
        
        dTiers = tm.FillGapsInTxtGridDict(dTiers)
        dTiers = tm.Merge_labels(dTiers)
        merged_tiers = [t for t in dTiers.keys() if t[0:2]=='m-']
        tm.WriteTxtGrdFromDict(textgrid_file,dTiers,0,self.duration,sFilGab='sil',lSlctdTiers=merged_tiers)