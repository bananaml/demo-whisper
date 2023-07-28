from potassium import Potassium, Request, Response
import torch
import boto3
import os

from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torchaudio

# create a new Potassium app
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():

    config = WhisperConfig.from_pretrained("openai/whisper-base")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    
    with init_empty_weights():
        model = WhisperForConditionalGeneration(config)
    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model, "model.safetensors", device_map="auto"
    )

    # set up boto3 client with credentials from environment variables
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    # get bucket from environment variable
    bucket = os.environ.get("AWS_BUCKET")
   
    context = {
        "model": model,
        "s3": s3,
        "bucket": bucket,
        "processor": processor,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    device = get_device()

    # get file path from request.json dict
    path = request.json.get("path")
    processor = context.get("processor")

    # download file from bucket
    context.get("s3").download_file(context.get("bucket"), path, "sample.wav")

    # open the stored file and convert to tensors
    input_features = processor(load_audio("sample.wav"), sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # run inference on the sample
    model = context.get("model")
    generated_ids = model.generate(inputs=input_features)
    
    # convert the generated ids back to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # return output JSON to the client
    return Response(
        json = {"outputs": transcription}, 
        status=200
    )

# Note that since this function doesn't have a decorator, it's not a handler
def load_audio(audio_path):
    """Loads audio file into tensor and resamples to 16kHz"""
    speech, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    
    return speech.squeeze()

def get_device():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on CUDA")
    
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on MPS")
    
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    return device

if __name__ == "__main__":
    app.serve()
