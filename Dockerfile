# Base image for Banana model builds
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install additional python packages
# torch is already installed in this image
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add your model weight files 
# (in this case we have a python script)
ADD utils.py .
ADD download.py .
RUN python3 download.py

# Add the rest of your code
ADD . .

# Set environment variables passed in during build
ARG AWS_BUCKET
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_BUCKET=$AWS_BUCKET
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

EXPOSE 8000

# Start the app in the container
CMD python3 -u app.py
