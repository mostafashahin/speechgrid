
import txtgrid_master.TextGrid_Master as tm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa


class SpeechRecognition:
    def __init__(self,model_id, device = torch.device('cpu'), lm_model_path=None, lang='eng'):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        
        if 'mms' in model_id:
            self.model.load_adapter(target_lang=lang, local_files_only=True)
            self.processor.tokenizer.set_target_lang(lang)
        
        self.model.to(self.device)
        self.lm_decode = None
        if lm_model_path:
            try:
                from tasks.asr import lm_decoder
                vocab_dict = self.processor.tokenizer.get_vocab()
                self.lm_decode = lm_decoder.NgramDecoder(vocab_dict, lm_model_path)
            except Exception as e:
                print(f"Error in loading language model {lm_model_path} language not applied")
                print(e)
        
    
    def recognize(self,speech):
        inputs = self.processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        if self.lm_decode:
            try:
                predicted_sentences = self.lm_decode.decode(logits)
            except Exception as e:
                print('Error in decoding with LM',e)
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_sentences = self.processor.batch_decode(predicted_ids)[0]
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = self.processor.batch_decode(predicted_ids)[0]
        return predicted_sentences
    
    def process_speech(self, speech, sr=16000):
        self.duration = librosa.get_duration(y=speech, sr=sr)
        self.dTiers_asr = {'words':([0],[len(speech)/16000],[self.recognize(speech)])}
        
    
    def process_intervals(self, speech, textgrid_file, sr=16000, offset_sec=0, speech_label = 'speech'):
        self.duration = librosa.get_duration(y=speech, sr=sr)
        dTiers = tm.ParseTxtGrd(textgrid_file)
        dTiers_asr = {}
        for tier_name in dTiers.keys():
            llabel = []
            for start_time, end_time, label in zip(*dTiers[tier_name]):
                if speech_label in label:
                    sample = speech[int((start_time-offset_sec)*sr):int((end_time+offset_sec)*sr)]
                    try:
                        label = self.recognize(sample)
                    except RuntimeError:
                        pass
                        #print('Error in', start_time, end_time, label)
                    #print(start_time, end_time, label)       
                llabel.append(label)
            dTiers_asr[tier_name] = (dTiers[tier_name][0],dTiers[tier_name][1], llabel)
        self.dTiers_asr = dTiers_asr


    def write_textgrid(self, textgrid_file):
        tm.WriteTxtGrdFromDict(textgrid_file,self.dTiers_asr,0,self.duration)
        