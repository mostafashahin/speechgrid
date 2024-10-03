from collections import defaultdict
from txtgrid_master import TextGrid_Master as tm
import librosa
from pyannote.audio import Pipeline
import torch



class speaker_diar:
    def __init__(self, device = torch.device('cpu')):
        self.device = device
        self.pipeline = Pipeline.from_pretrained("Models/speaker_diar/pyannote3.1/speaker-diarization-3.1/config.yaml")
        self.pipeline.to(device)
        
    def diarize(self, speech, sr):
        self.duration = librosa.get_duration(y=speech, sr=sr)
        speech = torch.from_numpy(speech)
        speech = speech.unsqueeze(0).to(self.device)
        audio = {"waveform": speech, "sample_rate": sr}
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