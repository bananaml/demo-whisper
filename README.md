![](https://www.banana.dev/lib_zOkYpJoyYVcAamDf/x2p804nk9qvjb1vg.svg?w=340 "Banana.dev")

# Banana.dev Whisper transcription starter template

This is an `openai/whisper-base` starter template from [Banana.dev](https://www.banana.dev) that allows on-demand serverless GPU inference.

You can fork this repository and deploy it on Banana as is, or customize it based on your own needs.

Rather than accepting files in requests, it uses AWS S3 to read stored audio files at runtime and return the transcribed text.

# Running this app

## Deploying on Banana.dev

1. [Fork this](https://github.com/bananaml/demo-whisper/fork) repository to your own Github account.
2. Connect your Github account on Banana.
3. [Create a new model](https://app.banana.dev/deploy) on Banana from the forked Github repository.

## Running after deploying

1. Wait for the model to build after creating it.
2. Make an API request to it using one of the provided snippets in your Banana dashboard.

For more info, check out the [Banana.dev docs](https://docs.banana.dev/banana-docs/).

## Testing locally

### Using Docker

Build the model as a Docker image. You can change the `banana-whisper` part to anything.

Make sure to change the three AWS variables to your own.

```sh
docker build --build-arg AWS_ACCESS_KEY_ID=your_access_key_id --build-arg AWS_SECRET_ACCESS_KEY=your_secret_key --build-arg AWS_BUCKET=your_bucket -t banana-whisper .
```

Run the Potassium server

```sh
docker run --publish 8000:8000 -it banana-whisper
```

Run inference after the above is built and running.
This assumes you have a "hello_world.wav" file in your S3 bucket.

```sh
curl -X POST -H 'Content-Type: application/json' -d '{"path": "hello_world.wav"}' http://localhost:8000
```

### Without Docker

You could also install and run it without Docker.

Just make sure that the pip dependencies in the Docker file (and torch) are installed in your Python virtual environment.

Run the Potassium app in one terminal window.

```sh
AWS_ACCESS_KEY_ID=your_access_key_id AWS_SECRET_ACCESS_KEY=your_secret_key AWS_BUCKET=your_bucket python3 app.py
```

Call the model in another terminal window with the Potassium app still running from the previous step.

```sh
curl -X POST -H 'Content-Type: application/json' -d '{"path": "hello_world.wav"}' http://localhost:8000
```

# Requirements

## ffmpeg

The `ffmpeg` system dependency is required.

## S3

S3 read credentials (access key id and secret) and an S3 bucket are required to read files uploaded to S3.

You should add these to your model's settings using the same keys as in the `Dockerfile`
