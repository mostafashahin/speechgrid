import logging
from logging import handlers
import os
import torch

accelerate = False  # Corrected from 'Fasle' to 'False'
device = torch.device('cpu')

if torch.cuda.is_available():
    accelerate = True
    device = torch.device('cuda')

# Define the output directory path
output_dir = os.path.join(os.getcwd(), 'output')

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Define constants and configurations
MAX_DUR = 120  # 2 minutes
#ASR_MODEL = os.path.join(os.getcwd(), 'Models/ASR/wav2vec2-large-xlsr-53-english/')
ASR_MODEL = os.path.join(os.getcwd(), 'Models/ASR/mms-1b-all')
LM_PATH = os.path.join(os.getcwd(), 'Models/ASR/LM/ngram/4gram_big.arpa')
SPEECH_LABEL = 'speech'


def setup_logger():
    log_file = 'logs/log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    formatter = logging.Formatter("{asctime} - {name} - {levelname} - {message}", style='{')

    file_handler = logging.handlers.RotatingFileHandler(filename=log_file, backupCount=10, maxBytes=100*1024*1024)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    