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
from config import setup_logger
import logging

from tasks.sd import pyannote_sd
from tasks.vad import pyannote_vad
from tasks.asr import wav2vec_asr


if len(sys.argv) == 2:
    shareable=bool(int(sys.argv[1]))
else:
    shareable = False

    

setup_logger()
logger = logging.getLogger(__name__)

def load_asr(params=None):
    #TODO add to the asr class to read parameters and load the correct model? or a separate function
    logger.info("Loading ASR Model...")

    try:
        asr_engine = wav2vec_asr.speech_recognition(ASR_MODEL, device= device, lm_model_path=LM_PATH) #This from parameters and has default one #If language not determine use language id
    except:
        logger.exception(f'Error loading asr model {ASR_MODEL}')
        raise "Error in loading asr model"
    
    return asr_engine



def load_sd(params=None):
    logger.info("Loading SD Model...")
    
    try:
        diarizer = pyannote_sd.speaker_diar(device= device)
    except Exception as e:
        print(f'Error loading speaker diarization model')
        raise f"Error in loading speaker diarization model {e}"
    
    return diarizer



vad_params = {
             'min_duration_off': 0.09791355693027545,
             'min_duration_on': 0.05537587440407595
             }

def load_vad(params=None):
    logger.info("Loading VAD Model...")
    
    model_path = 'Models/VAD/pytorch_model.bin'
    try:
        vad_pipeline = pyannote_vad.speech_detection(model_path, device= device, params=params)
    except:
        print(f'Error loading voice activity detection model {model_path}')
        raise "Error in loading voice activity detection model"   
    return vad_pipeline


def process_file(speech_file, tasks=['SD', 'ASR'], parameters=None, progress=gr.Progress()):
    basename = generate_file_basename()
    

    progress(0, desc=f"Loading Speech File...")
    
    logger.info('Loading Speech File...')

    try:
        speech, sr, duration = load_speech_file(speech_file)
    except Exception as e:
        logger.exception(f'Failed to load the speech file {speech_file}, {e}')
        raise
    
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

    logger.info('Start processing, following tasks will be performed', ','.join(task_pipeline))

    #Loading task engines
    if 'ASR' in task_pipeline:
        asr_engine = load_asr()
    
    if 'SD' in task_pipeline:
        diarizer = load_sd()
        
    if 'VAD' in task_pipeline:
        vad_engine = load_vad(params=vad_params)
    
    out_textgrid = []


    num_processes = len(task_pipeline)+1
    
    
    i = 1
    for task in task_pipeline:
        logger.info(f'Applying {task}')
        progress(i/(num_processes+1), desc=f"Applying {task}")
        i += 1
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
    
    progress(num_processes/(num_processes+1), desc=f"Generate output files")


    logger.info(f'Saving output files in {output_dir}, {basename}')

    #This save a version of the speech file with 16k, mono, 16bit
    try:
        speech_file = join(output_dir,f'{basename}.wav')
        sf.write(speech_file, speech, sr)
        download_speech_enable = True
        download_speech_value = speech_file
        download_speech_label = f"Download speech file"
    except Exception as e:
        logger.exception(f'Failed to save the speech file {speech_file}, {e}')
        download_speech_enable = False
        download_speech_value = None
        download_speech_label = "Error in saving speech file"
    
    try:
        p = zip_files(out_textgrid, output_zip_file)
        download_data_enable = True
        download_data_value = output_zip_file
        download_data_label = f"Download output file"
   
    except Exception as e:
        logger.exception(f'Failed to create output archive in {output_zip_file}, {e}')
        #Disable Download Data Button
        download_data_enable = False
        download_data_value = None
        download_data_label = "Error in archiving data"
        
        
     
    progress(1, desc="Processing completed..")

    logger.info('Processing is completed')
    
    return ["Processing completed..", 
            gr.DownloadButton(label=download_data_label,
                              value=download_data_value,
                              interactive=download_data_enable,
                              visible=True),
            
            gr.DownloadButton(label=download_speech_label,
                              value=download_speech_value,
                              interactive=download_speech_enable,
                              visible=True)]

    
    
#TODO: Progress bar
#TODO: VAD ########DONE
#TODO: Number of speakers
#TODO: Word alignment
#TODO: Download button  ######DONE
#TODO: Wrape in a docker #####DONE
#TODO: Test pyannote offline  ########DONE
#TODO: Logging of errors and info    #########IN PROGRESS
#TODO: ASR with LM  ########DONE
#TODO: Process batch
#TODO: Kaldi ASR
#TODO: MMS ASR
#TODO: Add parameters selection for ASR, SD, VAD
#TODO: Rewrite the textgrid
#TODO: TextGrid code to get the logger and use logging instead of print.
#TODO: Add logging to other packages
#TODO: Use .bin instead of ARPA in LM
#TODO: ASR add the expected words



with gr.Blocks(title="SpeechGrid", theme=gr.themes.Soft()) as gui:

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
    
    
     
gui.launch(share=shareable)
