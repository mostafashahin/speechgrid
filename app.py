import os
import sys
import logging

import gradio as gr

from speechgrid import SpeechGrid, SpeechGridInterface
from config import setup_logger, output_dir



if len(sys.argv) == 2:
    shareable=bool(int(sys.argv[1]))
else:
    shareable = False

    

setup_logger()
logger = logging.getLogger(__name__)


speech_grid_interface = SpeechGridInterface(config_file='config.yaml')

    
    
#TODO: Progress bar
#TODO: VAD ########DONE
#TODO: Number of speakers  ######DONE
#TODO: Word alignment
#TODO: Download button  ######DONE
#TODO: Wrape in a docker #####DONE
#TODO: Test pyannote offline  ########DONE
#TODO: Logging of errors and info    #########IN PROGRESS
#TODO: ASR with LM  ########DONE
#TODO: Process batch
#TODO: Kaldi ASR
#TODO: MMS ASR   ######DONE
#TODO: Add parameters selection for ASR, SD, VAD
#TODO: Rewrite the textgrid
#TODO: TextGrid code to get the logger and use logging instead of print.
#TODO: Add logging to other packages
#TODO: Use .bin instead of ARPA in LM
#TODO: ASR add the expected words
#TODO: Consider control the offset in interval ASR
#TODO: Add parameters of min silence duration #NEED TO WELL UNDERSTAND THESE PARAMETERS
#TODO: Create set, get for speaker number
#TODO: Name of file as the uploaded file
#TODO: Make process enable after loading or recording ####DONE####
#TODO: Use import tempfile to access the temp dir if need to do so
#TODO: Review the use of get and set, use @property instead or direct access
#TODO: Add nemo diarization

#TODO: May specify the min but max 0?!

def reset_min_max(exact,min_v,max_v):
    if exact > 0:
        min_v = 0
        max_v = 0
    speech_grid_interface.set_speaker_numbers(exact, min_v, max_v)
    return exact, min_v, max_v

def validate_min_max(exact, min_v, max_v):
    if min_v > 0 or max_v > 0:
        exact = 0
        if min_v > max_v:
            max_v = min_v
    speech_grid_interface.set_speaker_numbers(exact, min_v, max_v)
    return exact, min_v, max_v

def audio_stat(audio_value):
    if audio_value:
        return gr.Button("Process Audio", interactive=True)
    else:
        return gr.Button("Process Audio", interactive=False)

def audio_record():
    speech_grid_interface.recorded_speech = True

def audio_upload():
    speech_grid_interface.recorded_speech = False

with gr.Blocks(title="SpeechGrid", theme=gr.themes.Soft()) as gui:

    with gr.Tab('Main'):

        record_audio = gr.Audio(sources=["microphone","upload"], type="filepath")
        
        
    
        tasks = gr.CheckboxGroup(choices=[("Speech to Text","ASR"),
                                          ("Speaker Separation","SD"),
                                          ("Speech Detection","VAD")],
                                 value=speech_grid_interface.get_tasks(),
                                 label="Tasks",
                                 info="Apply the following tasks:")
        tasks.input(speech_grid_interface.set_tasks, inputs=tasks)
            
    
        process = gr.Button("Process Audio", interactive=False)

        record_audio.input(audio_stat, inputs=record_audio, outputs=process)
        record_audio.stop_recording(audio_record)
        record_audio.upload(audio_upload)
    
        output_text = gr.Textbox(label='Progress', interactive=False)
        with gr.Row():
            with gr.Column():
                d1 = gr.DownloadButton("Download output", visible=False)
            with gr.Column():
                d2 = gr.DownloadButton("Download speech file", visible=False)
    
        
        
    with gr.Tab('Advanced Options'):
        gr.Markdown(
            """
            ### Speaker Separation
            Number of speakers
            """)
        with gr.Row():
            n_exact = gr.Number(label='Exact')
            n_min = gr.Number(label='Minimum')
            n_max = gr.Number(label='Maximum')

            n_exact.input(reset_min_max,
                           inputs=[n_exact, n_min, n_max],
                          outputs=[n_exact, n_min, n_max])
            n_min.input(validate_min_max,
                        inputs=[n_exact, n_min, n_max],
                        outputs=[n_exact, n_min, n_max])
            n_max.input(validate_min_max,
                        inputs=[n_exact, n_min, n_max],
                        outputs=[n_exact, n_min, n_max])
        gr.Markdown(
            """
            ### Speech to Text
            """
        )
        with gr.Row():
            avail_lang = [(k,v) for k,v in speech_grid_interface.get_asr_available_lang().items()]
            lang_drop = gr.Dropdown(label='Language', choices=avail_lang, value=speech_grid_interface.get_asr_lang())
            lang_drop.change(speech_grid_interface.set_asr_lang,
                            inputs=lang_drop)

            is_lm_enabled = speech_grid_interface.get_lm_enable()
            lm_enable = gr.Checkbox(label='Enable Language Model', value=is_lm_enabled, interactive=True)
            lm_enable.change(speech_grid_interface.set_lm_enable,
                            inputs=lm_enable)
    
    process.click(speech_grid_interface.process, 
                      inputs=record_audio,
                      outputs=[output_text, d1, d2])
     
gui.queue().launch(share=shareable)
