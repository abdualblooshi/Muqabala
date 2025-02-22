# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-pip \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    libasound2-dev \
    pulseaudio \
    alsa-utils \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configure PulseAudio for WSL
RUN mkdir -p /root/.config/pulse && \
    echo "load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1" > /root/.config/pulse/default.pa && \
    echo "load-module module-esound-protocol-tcp" >> /root/.config/pulse/default.pa && \
    echo "load-module module-null-sink sink_name=MySink" >> /root/.config/pulse/default.pa && \
    echo "load-module module-virtual-source source_name=MySource" >> /root/.config/pulse/default.pa

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/tts_models /app/temp_audio /app/reports && \
    chmod -R 777 /app/tts_models /app/temp_audio /app/reports && \
    chown -R root:root /app/tts_models /app/temp_audio /app/reports

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in stages
# 1. Install numpy first with specific version
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy==1.22.0"

# 2. Install torch and pandas
RUN pip install --no-cache-dir torch "pandas>=1.4.0,<2.0.0"

# 3. Install audio-related packages
RUN pip install --no-cache-dir wheel \
    && pip install --no-cache-dir SpeechRecognition pyttsx3 wave \
    && PORTAUDIO_INC_DIR=/usr/include/portaudio.h pip install --no-cache-dir pyaudio

# 4. Install Piper TTS and download models
RUN pip install --no-cache-dir piper-tts>=1.2.0 \
    && cd /app/tts_models \
    && wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx \
    && wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json \
    && chmod 644 *.onnx *.json \
    && chown -R root:root /app/tts_models

# 5. Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure proper permissions after copying
RUN chmod -R 777 /app/temp_audio /app/reports && \
    chown -R root:root /app/temp_audio /app/reports

# Set environment variables for audio
ENV PYTHONUNBUFFERED=1
ENV PULSE_SERVER=host.docker.internal
ENV ALSA_CARD=MySink
ENV PIPER_SHARE_DIR=/app/tts_models

# Create ALSA config
RUN echo 'pcm.!default { type plug slave.pcm "null" }' > /etc/asound.conf

# Expose Streamlit port
EXPOSE 8501

# Start PulseAudio in the background and run the application
CMD pulseaudio --start && streamlit run app.py --server.address=0.0.0.0  

