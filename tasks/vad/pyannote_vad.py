import librosa
import torch
from collections import defaultdict

from txtgrid_master import TextGrid_Master as tm
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model




class speech_detection:
    def __init__(self,Model_path, params = None):
        model = Model.from_pretrained(Model_path)
        self.pipeline = VoiceActivityDetection(segmentation=model)
        self.pipeline.instantiate(params)
        
    def DoVAD(self, speech, sr):
        self.duration = librosa.get_duration(y=speech, sr=sr)
        speech = torch.from_numpy(speech)
        speech = speech.unsqueeze(0)
        audio = {"waveform": speech, "sample_rate": sr}
        self.vad = self.pipeline(audio)
    
    def write_rttm(self, rttm_file):
        with open(rttm_file, "w") as rttm:
            self.vad.write_rttm(rttm)
    
    def write_textgrid(self, textgrif_file, speech_label='speech'):
        dTiers = defaultdict(lambda : [[],[],[]])
        for s, t, l in self.vad.itertracks(yield_label=True):
            dTiers[l][0].append(round(s.start,2))
            dTiers[l][1].append(round(s.end,2))
            dTiers[l][2].append(speech_label)
        
        dTiers = tm.FillGapsInTxtGridDict(dTiers)
        dTiers = tm.Merge_labels(dTiers)
        merged_tiers = [t for t in dTiers.keys() if t[0:2]=='m-']
        tm.WriteTxtGrdFromDict(textgrif_file,dTiers,0,self.duration,sFilGab='sil',lSlctdTiers=merged_tiers)