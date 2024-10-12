import yaml
import logging
import os

from config import output_dir, device

from tasks.sd import pyannote_sd
from tasks.vad import pyannote_vad
from tasks.asr import wav2vec_asr

logger = logging.getLogger(__name__)
class SpeechGrid:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.available_tasks = self.config['speechgrid']['tasks']
        self.speech_label = self.config['speechgrid']['speech_label']
        self.max_dur = self.config['speechgrid']['max_dur']
        self.loaded_tasks = {}
        

    def load_asr(self):
        
        logger.info("Loading ASR Model...")
        asr_model_path = self.config['speechgrid']['speech_recognition']['asr_model_path']
        asr_model_path = os.path.join(os.getcwd(), asr_model_path)

        lm_enable = self.config['speechgrid']['speech_recognition']['lang_model']

        if lm_enable:
            lm_model_path = self.config['speechgrid']['speech_recognition']['lm_model_path']
        else:
            lm_model_path = None

        asr_lang = self.config['speechgrid']['speech_recognition']['lang']
        
        try:
            asr_engine = wav2vec_asr.SpeechRecognition(asr_model_path, 
                                                       device= device, 
                                                       lm_model_path=lm_model_path, 
                                                       lang= asr_lang)
            self.loaded_tasks['ASR'] = asr_engine
        except:
            logger.exception(f'Error loading asr model {asr_model_path}')
            raise "Error in loading asr model"


    def load_sd(self):
        logger.info("Loading SD Model...")

        sd_model_path = self.config['speechgrid']['speaker_diarization']['model_path']
        
        try:
            sd_engine = pyannote_sd.SpeakerDiarization(model_path= sd_model_path, device= device)
            self.loaded_tasks['SD'] = sd_engine
        except:
            logger.exception(f'Error loading speaker diarization model {sd_model_path}')
            raise f"Error in loading sd model"


    def load_vad(self):
        logger.info("Loading VAD Model...")
    
        vad_model_path = self.config['speechgrid']['voice_activity_detection']['model_path']
        params = {
                      'min_duration_off': self.config['speechgrid']['voice_activity_detection']['min_duration_off'],
                      'min_duration_on': self.config['speechgrid']['voice_activity_detection']['min_duration_on']
                     }
        
        try:
            vad_engine = pyannote_vad.VoiceActivityDetection(vad_model_path, device= device, params=params)
            self.loaded_tasks['VAD'] = vad_engine
        except:
            logger.exception(f'Error loading voice activity detection model {vad_model_path}')
            raise "Error in loading voice activity detection model"   

    def load_tasks(self, task_list):
        if 'ASR' in task_list:
            self.load_asr()
        if 'SD' in task_list:
            self.load_sd()
        if 'VAD' in task_list:
            self.load_vad()

    def create_task_pipeline(self, selected_tasks, duration): #Per speech file
        if len(selected_tasks) == 1:
            if 'ASR' in selected_tasks and duration > self.max_dur: #Add VAD task to split the speech file by sil
                task_pipeline = ['VAD', 'ASR']
            else:
                task_pipeline = list(selected_tasks)
        elif set(selected_tasks) == set(['SD', 'ASR']):
            task_pipeline = ['SD', 'ASR'] #Apply SD first and then ASR on the SD intervals
        elif set(selected_tasks) == set(['VAD', 'ASR']):
            task_pipeline = ['VAD', 'ASR'] #Apply VAD first and then ASR on the SD intervals
        elif set(selected_tasks) == set(['VAD', 'ASR', 'SD']): #If both SD, VAD and ASR then ASR will be applied on SD output
            task_pipeline = ['VAD', 'SD', 'ASR']
        else:
            task_pipeline = tasks #Only 'SD' and 'VAD' each one will be applied separately

        return task_pipeline