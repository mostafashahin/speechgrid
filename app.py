import sys
import os
from os.path import splitext, join
from collections import defaultdict

import librosa
import soundfile as sf
import torch


import gradio as gr
from gradio.components import Audio, Dropdown, Textbox, Image




import txtgrid_master.TextGrid_Master as tm
from core.utils import generate_file_basename, load_speech_file, zip_files
from config import output_dir, MAX_DUR, ASR_MODEL, SPEECH_LABEL, LM_PATH, device

from tasks.sd import pyannote_sd
from tasks.vad import pyannote_vad
from tasks.asr import wav2vec_asr
    
    
def load_asr(params=None):
    #TODO add to the asr class to read parameters and load the correct model? or a separate function

    try:
        asr_engine = wav2vec_asr.speech_recognition(ASR_MODEL, device= device, lm_model_path=LM_PATH) #This from parameters and has default one #If language not determine use language id
    except:
        print(f'Error loading asr model {ASR_MODEL}')
        raise "Error in loading asr model"
    
    return asr_engine



def load_sd(params=None):
    
    try:
        diarizer = pyannote_sd.speaker_diar()
    except Exception as e:
        print(f'Error loading speaker diarization model')
        raise f"Error in loading speaker diarization model {e}"
    
    return diarizer



vad_params = {
             'min_duration_off': 0.09791355693027545,
             'min_duration_on': 0.05537587440407595
             }

def load_vad(params=None):
    model_path = 'Models/VAD/pytorch_model.bin'
    try:
        vad_pipeline = pyannote_vad.speech_detection(model_path, params)
    except:
        print(f'Error loading voice activity detection model {model_path}')
        raise "Error in loading voice activity detection model"   
    return vad_pipeline


def process_file(speech_file, tasks=['SD', 'ASR'], parameters=None, progress=gr.Progress()):
    basename = generate_file_basename()
    
    speech, sr, duration = load_speech_file(speech_file)
    
    #This save a version of the speech file with 16k, mono, 16bit
    speech_file = join(output_dir,f'{basename}.wav')
    sf.write(speech_file, speech, sr)
    
    tasks = set(tasks)
    
    task_pipeline = []
    if len(tasks) == 1:
        if 'ASR' in tasks and duration > MAX_DUR: #Add VAD task to split the speech file by sil
            task_pipeline = ['VAD', 'ASR']
        else:
            task_pipeline = list(tasks)
    elif set(tasks) == set(['SD', 'ASR']):
        task_pipeline = ['SD', 'ASR']
    elif set(tasks) == set(['VAD', 'ASR']):
        task_pipeline = ['VAD', 'ASR']
    elif set(tasks) == set(['VAD', 'ASR', 'SD']): #If both SD, VAD and ASR then ASR will be applied on SD output
        task_pipeline = ['VAD', 'SD', 'ASR']
    else:
        task_pipeline = tasks #Only 'SD' and 'VAD' each one will be applied separetly

    #print(speech_file, tasks, duration)
    if 'ASR' in task_pipeline:
        asr_engine = load_asr()
    
    if 'SD' in task_pipeline:
        diarizer = load_sd()
        
    if 'VAD' in task_pipeline:
        vad_engine = load_vad(params=vad_params)
    
    out_textgrid = []
    
    i = 0
    for task in task_pipeline:
        progress(i/(len(task_pipeline)+1), desc=f"Applying {task}")
        i = i+1
        if task == 'ASR':
            texgrid_file = join(output_dir,f'{basename}_ASR.TextGrid')
            if not out_textgrid:
                dTiers_asr = asr_engine.process_speech(speech)
            else:
                input_textgrid = out_textgrid[-1]
                dTiers_asr = asr_engine.process_intervals(speech, input_textgrid, sr = sr, offset_sec=0, speech_label = SPEECH_LABEL)
            
            tm.WriteTxtGrdFromDict(texgrid_file,dTiers_asr,0,duration)
            out_textgrid.append(texgrid_file)
        
        elif task == 'VAD':
            rttm_file = join(output_dir,f'{basename}_VAD.rttm')
            texgrid_file = join(output_dir,f'{basename}_VAD.TextGrid')
            vad_engine.DoVAD(speech,sr)
            vad_engine.write_rttm(rttm_file)
            vad_engine.write_textgrid(texgrid_file, speech_label=SPEECH_LABEL)
            out_textgrid.append(texgrid_file)
        
        elif task == 'SD':
            rttm_file = join(output_dir,f'{basename}_SD.rttm')
            texgrid_file = join(output_dir,f'{basename}_SD.TextGrid')
            diarizer.diarize(speech=speech, sr=sr)
            diarizer.write_rttm(rttm_file)
            diarizer.write_textgrid(texgrid_file, speech_label=SPEECH_LABEL)
            out_textgrid.append(texgrid_file)
    
    
    output_zip_file = join(output_dir,f'{basename}_output.zip')
    
    progress(len(task_pipeline)/(len(task_pipeline)+1), desc=f"Create output archive")
    p = zip_files(out_textgrid, output_zip_file)
    
    progress(1, desc=p)
    
    return [p, gr.DownloadButton(label=f"Download output file", value=output_zip_file, visible=True), 
           gr.DownloadButton(label=f"Download speech file", value=speech_file, visible=True)]

    
    
#TODO: Progress bar
#TODO: VAD ########DONE
#TODO: Number of speakers
#TODO: Word alignment
#TODO: Download button  ######DONE
#TODO: Wrape in a docker #####DONE
#TODO: Test pyannote offline  ########DONE
#TODO: Logging of errors and info
#TODO: ASR with LM
#TODO: Process batch
#TODO: Kaldi ASR
#TODO: MMS ASR



with gr.Blocks(theme=gr.themes.Soft()) as gui:

    record_audio = gr.Audio(sources=["microphone","upload"], type="filepath")

    tasks = gr.CheckboxGroup(choices=[("Speech to Text","ASR"), ("Speaker Separation","SD"),("Speech Detection","VAD")], label="Tasks", info="Apply the following tasks:")

    process = gr.Button("Process Audio")

    output_text = gr.Textbox(label='Progress', interactive=False)
    with gr.Row():
        with gr.Column():
            d1 = gr.DownloadButton("Download output", visible=False)
        with gr.Column():
            d2 = gr.DownloadButton("Download speech file", visible=False)

    process.click(process_file, inputs=[record_audio, tasks], outputs=[output_text, d1, d2])
    
    
     
gui.launch()