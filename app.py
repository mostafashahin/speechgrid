import os
import sys
import logging
import soundfile as sf
import gradio as gr

from speechgrid import SpeechGrid
from config import setup_logger, output_dir
from core.utils import generate_file_basename, load_speech_file, zip_files


if len(sys.argv) == 2:
    shareable=bool(int(sys.argv[1]))
else:
    shareable = False

    

setup_logger()
logger = logging.getLogger(__name__)


speech_grid = SpeechGrid(config_file='config.yaml')

#Consider use dynamic arguments *args and **kwargs
def process_file(speech_file, tasks=['SD', 'ASR'],
                 n_exact_speakers = 0,
                 n_min_speakers = 0,
                 n_max_speakers = 0,
                 progress=gr.Progress()):
    
    basename = generate_file_basename()
    

    progress(0, desc=f"Loading Speech File...")
    
    logger.info('Loading Speech File...')

    try:
        speech, sr, duration = load_speech_file(speech_file)
    except Exception as e:
        logger.exception(f'Failed to load the speech file {speech_file}, {e}')
        raise
    
    tasks = set(tasks)

    
    
    task_pipeline = speech_grid.create_task_pipeline(tasks, duration)

    logger.info('Start processing, following tasks will be performed', ','.join(task_pipeline))

    #Loading task engines
    speech_grid.load_tasks(task_pipeline)
    
    out_textgrid = []


    num_processes = len(task_pipeline)+1
    
    
    i = 1
    for task in task_pipeline:
        logger.info(f'Applying {task}')
        progress(i/(num_processes+1), desc=f"Applying {task}")
        i += 1
        if task == 'ASR':
            asr_engine = speech_grid.loaded_tasks['ASR']
            textgrid_file = os.path.join(output_dir,f'{basename}_ASR.TextGrid')
            if not out_textgrid:
                asr_engine.process_speech(speech)
            else:
                input_textgrid = out_textgrid[-1]
                asr_engine.process_intervals(speech, input_textgrid, sr = sr, offset_sec=0, speech_label = speech_grid.speech_label)
            
            asr_engine.write_textgrid(textgrid_file)
            out_textgrid.append(textgrid_file)
        
        elif task == 'VAD':
            vad_engine = speech_grid.loaded_tasks['VAD']
            rttm_file = os.path.join(output_dir,f'{basename}_VAD.rttm')
            textgrid_file = os.path.join(output_dir,f'{basename}_VAD.TextGrid')
            vad_engine.process_speech(speech,sr)
            vad_engine.write_rttm(rttm_file)
            vad_engine.write_textgrid(textgrid_file, speech_label=speech_grid.speech_label)
            out_textgrid.append(textgrid_file)
        
        elif task == 'SD':
            sd_engine = speech_grid.loaded_tasks['SD']
            rttm_file = os.path.join(output_dir,f'{basename}_SD.rttm')
            textgrid_file = os.path.join(output_dir,f'{basename}_SD.TextGrid')
            sd_engine.process_speech(speech=speech,
                                     sr=sr,
                                     n_exact_speakers = n_exact_speakers,
                                     n_min_speakers = n_min_speakers,
                                     n_max_speakers = n_max_speakers)
            sd_engine.write_rttm(rttm_file)
            sd_engine.write_textgrid(textgrid_file, speech_label=speech_grid.speech_label)
            out_textgrid.append(textgrid_file)
    
    
    output_zip_file = os.path.join(output_dir,f'{basename}_output.zip')
    
    progress(num_processes/(num_processes+1), desc=f"Generate output files")


    logger.info(f'Saving output files in {output_dir}, {basename}')

    #This save a version of the speech file with 16k, mono, 16bit
    try:
        speech_file = os.path.join(output_dir,f'{basename}.wav')
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
#TODO: Consider control the offset in interval ASR
#TODO: Add parameters of min silence duration #NEED TO WELL UNDERSTAND THESE PARAMETERS
#TODO: Create set, get for speaker number


def reset_min_max(exact,min_v,max_v):
  if exact > 0:
      min_v = 0
      max_v = 0
  return exact, min_v, max_v

def validate_min_max(exact, min_v, max_v):
    if min_v > 0 or max_v > 0:
        exact = 0
        if min_v > max_v:
            max_v = min_v
    return exact, min_v, max_v



with gr.Blocks(title="SpeechGrid", theme=gr.themes.Soft()) as gui:

    with gr.Tab('Main'):

        record_audio = gr.Audio(sources=["microphone","upload"], type="filepath")
    
        tasks = gr.CheckboxGroup(choices=[("Speech to Text","ASR"),
                                          ("Speaker Separation","SD"),
                                          ("Speech Detection","VAD")], 
                                 label="Tasks",
                                 info="Apply the following tasks:")
            
    
        process = gr.Button("Process Audio")
    
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

            n_exact.change(reset_min_max,
                           inputs=[n_exact, n_min, n_max],
                          outputs=[n_exact, n_min, n_max])
            n_min.change(validate_min_max,
                        inputs=[n_exact, n_min, n_max],
                        outputs=[n_exact, n_min, n_max])
            n_max.change(validate_min_max,
                        inputs=[n_exact, n_min, n_max],
                        outputs=[n_exact, n_min, n_max])
        gr.Markdown(
            """
            ### Speech to Text
            """
        )
        with gr.Row():
            avail_lang = [(k,v) for k,v in speech_grid.get_asr_available_lang().items()]
            lang_drop = gr.Dropdown(label='Language', choices=avail_lang, value=speech_grid.get_asr_lang())
            lang_drop.change(speech_grid.set_asr_lang,
                            inputs=lang_drop)

            is_lm_enabled = speech_grid.get_lm_enable()
            lm_enable = gr.Checkbox(label='Enable Language Model', value=is_lm_enabled, interactive=True)
            lm_enable.change(speech_grid.set_lm_enable,
                            inputs=lm_enable)
    
    process.click(process_file, 
                      inputs=[record_audio, 
                              tasks,
                              n_exact,
                              n_min,
                              n_max],
                      outputs=[output_text, d1, d2])
     
gui.launch(share=shareable)
