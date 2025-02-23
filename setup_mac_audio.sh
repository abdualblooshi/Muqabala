#!/bin/bash

echo "Setting up speech-to-text environment for macOS..."

# Check for Homebrew
if ! command -v brew &>/dev/null; then
  echo "Homebrew is not installed. Please install Homebrew from https://brew.sh and re-run the script."
  exit 1
fi

echo "Updating Homebrew..."
brew update

echo "Installing required packages..."
# Install PortAudio for audio I/O and SoX for recording/testing
brew install portaudio sox

echo "Installing Python packages..."
# Ensure pip is up-to-date and install PyAudio and SpeechRecognition
pip3 install --upgrade pip
pip3 install pyaudio SpeechRecognition

echo "Testing microphone input..."
echo "Recording 3 seconds of audio. Please speak now..."
# Use SoX's 'rec' command to record a 3-second audio clip (mono channel)
rec -c 1 test.wav trim 0 3

echo "Playing back the recorded audio..."
play test.wav

rm test.wav

echo "Speech-to-text setup complete!"
