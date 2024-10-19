import yaml
import logging
import os
import gradio as gr
import soundfile as sf

from config import output_dir, device
from core.utils import generate_file_basename, load_speech_file, zip_files

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
        

    def get_asr_available_lang(self):
        return self.config['speechgrid']['speech_recognition']['avail_lang']
    
    def get_asr_lang(self):
        return self.config['speechgrid']['speech_recognition']['lang']
        
    def set_asr_lang(self, lang):
        self.config['speechgrid']['speech_recognition']['lang'] = lang

    def get_lm_enable(self):
        return self.config['speechgrid']['speech_recognition']['lang_model']

    def set_lm_enable(self, is_enable):
        self.config['speechgrid']['speech_recognition']['lang_model'] = is_enable

    def set_speaker_numbers(self, n_exact, n_min, n_max):
        self.config['speechgrid']['speaker_diarization']['num_speakers'] = n_exact
        self.config['speechgrid']['speaker_diarization']['min_num_speakers'] = n_min
        self.config['speechgrid']['speaker_diarization']['max_num_speakers'] = n_max
    
    def get_speaker_numbers(self):
        n_exact = self.config['speechgrid']['speaker_diarization']['num_speakers'] 
        n_min = self.config['speechgrid']['speaker_diarization']['min_num_speakers']
        n_max = self.config['speechgrid']['speaker_diarization']['max_num_speakers']

        return n_exact, n_min, n_max
    
    def load_asr(self):
        
        logger.info("Loading ASR Model...")
        asr_model_path = self.config['speechgrid']['speech_recognition']['asr_model_path']
        asr_model_path = os.path.join(os.getcwd(), asr_model_path)

        lm_enable = self.get_lm_enable()

        if lm_enable:
            lm_model_path = self.config['speechgrid']['speech_recognition']['lm_model_path']
        else:
            lm_model_path = None

        asr_lang = self.get_asr_lang()
        
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

    def load_tasks(self, task_list, force_load=False):
        if 'ASR' in task_list and ('ASR' not in self.loaded_tasks or force_load):
            self.load_asr()
        if 'SD' in task_list and ('SD' not in self.loaded_tasks or force_load):
            self.load_sd()
        if 'VAD' in task_list and ('VAD' not in self.loaded_tasks or force_load):
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






class SpeechGridInterface(SpeechGrid):
    def __init__(self, config_file):
        super().__init__(config_file=config_file)
        self.tasks = []
        #Options
        self.recorded_speech = False #If user record file on the fly
    
        

    def add_task(self, task):
        if task not in self.tasks:
            self.tasks.append(task)

    def remove_task(self, task):
        if task in self.tasks:
            self.tasks.remove(task)

    def set_tasks(self, tasks):
        self.tasks = tasks
        
    def get_tasks(self):
        return self.tasks


    def process(self, path, mode='single', progress=gr.Progress()): #input either a speech file o
        self.load_tasks(self.get_tasks())

        if mode=='single':
            speech_file, output_zip_file, wav_file_created, out_file_created = self.process_file(path, progress = progress)
            if wav_file_created:
                download_speech_enable = True
                download_speech_value = speech_file
                download_speech_label = f"Download speech file"
            else:
                download_speech_enable = False
                download_speech_value = None
                download_speech_label = "Error in saving speech file"
                
            if out_file_created:
                download_data_enable = True
                download_data_value = output_zip_file
                download_data_label = f"Download output file"
            else:
                download_data_enable = False
                download_data_value = None
                download_data_label = "Error in archiving data"

            return ["Processing completed..", 
                gr.DownloadButton(label=download_data_label,
                                  value=download_data_value,
                                  interactive=download_data_enable,
                                  visible=True),
                
                gr.DownloadButton(label=download_speech_label,
                                  value=download_speech_value,
                                  interactive=download_speech_enable,
                                  visible=True)]
    
    def apply_tasks_to_speech(self, task_pipeline, speech, basename, sr=16000, progress = gr.Progress()):
        out_textgrid = []

        num_processes = len(task_pipeline)+1
        i = 1
        for task in task_pipeline:
            logger.info(f'Applying {task}...')
            progress(i/(num_processes+1), desc=f"Applying {task}")
            i += 1
            if task == 'ASR':
                asr_engine = self.loaded_tasks['ASR']
                textgrid_file = os.path.join(output_dir,f'{basename}_ASR.TextGrid')
                if not out_textgrid:
                    asr_engine.process_speech(speech)
                else:
                    input_textgrid = out_textgrid[-1]
                    asr_engine.process_intervals(speech, input_textgrid, sr = sr, offset_sec=0, 
                                                 speech_label = self.speech_label)
                
                asr_engine.write_textgrid(textgrid_file)
                out_textgrid.append(textgrid_file)
            
            elif task == 'VAD':
                vad_engine = self.loaded_tasks['VAD']
                rttm_file = os.path.join(output_dir,f'{basename}_VAD.rttm')
                textgrid_file = os.path.join(output_dir,f'{basename}_VAD.TextGrid')
                vad_engine.process_speech(speech,sr)
                vad_engine.write_rttm(rttm_file)
                vad_engine.write_textgrid(textgrid_file, speech_label=self.speech_label)
                out_textgrid.append(textgrid_file)
            
            elif task == 'SD':
                sd_engine = self.loaded_tasks['SD']
                rttm_file = os.path.join(output_dir,f'{basename}_SD.rttm')
                textgrid_file = os.path.join(output_dir,f'{basename}_SD.TextGrid')
                n_exact_speakers, n_min_speakers, n_max_speakers = self.get_speaker_numbers()
                sd_engine.process_speech(speech=speech,
                                         sr=sr,
                                         n_exact_speakers = n_exact_speakers,
                                         n_min_speakers = n_min_speakers,
                                         n_max_speakers = n_max_speakers)
                sd_engine.write_rttm(rttm_file)
                sd_engine.write_textgrid(textgrid_file, speech_label=self.speech_label)
                out_textgrid.append(textgrid_file)
        
        return out_textgrid
    
    def process_file(self, speech_file, progress=gr.Progress()):
        if self.recorded_speech:
            basename = generate_file_basename() #Generate random name
        else:
            basename = os.path.splitext(os.path.basename(speech_file))[0]

        progress(0, desc=f"Loading Speech File...")
        
        logger.info('Loading Speech File...')
    
        try:
            speech, sr, duration = load_speech_file(speech_file)
        except Exception as e:
            logger.exception(f'Failed to load the speech file {speech_file}, {e}')
            raise
        
        tasks = set(self.get_tasks())
        
        task_pipeline = self.create_task_pipeline(tasks, duration)
    
        logger.info(f'Start processing, following tasks will be performed on {basename} {','.join(task_pipeline)}')
    
        #Loading task engines
        self.load_tasks(task_pipeline)
        
        out_textgrid = self.apply_tasks_to_speech(task_pipeline, speech, basename, sr, progress)
        
        output_zip_file = os.path.join(output_dir,f'{basename}_output.zip')
        
        progress(1, desc=f"Generate output files")
    
    
        logger.info(f'Saving output files in {output_dir}, {basename}')
    
        #This save a version of the speech file with 16k, mono, 16bit
        wav_file_created = True
        out_file_created = True
        try:
            speech_file = os.path.join(output_dir,f'{basename}.wav')
            sf.write(speech_file, speech, sr)

        except Exception as e:
            logger.exception(f'Failed to save the speech file {speech_file}, {e}')
            wav_file_created = False
        
        try:
            p = zip_files(out_textgrid, output_zip_file)
       
        except Exception as e:
            logger.exception(f'Failed to create output archive in {output_zip_file}, {e}')
            out_file_created = False
      
         
        progress(1, desc=f"Processing {basename} completed..")
    
        logger.info(f'Processing {basename} is completed')

        return (speech_file, output_zip_file, wav_file_created, out_file_created)