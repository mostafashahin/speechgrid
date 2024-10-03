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
ASR_MODEL = os.path.join(os.getcwd(), 'Models/ASR/wav2vec2-large-xlsr-53-english/')
LM_PATH = os.path.join(os.getcwd(), 'Models/ASR/LM/ngram/4gram_big.arpa')
SPEECH_LABEL = 'speech'
