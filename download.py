# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import WhisperForConditionalGeneration, WhisperConfig, AutoProcessor
from utils import custom_save


def download_model():
    # do a dry run of loading the huggingface model, which will download weights & config objects
    
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    config = WhisperConfig.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    custom_save(model)

if __name__ == "__main__":
    download_model()