#!/bin/bash

# Install required audio dependencies
pip install pyaudio

# Install portaudio (required for pyaudio on macOS)
brew install portaudio

# Test audio setup
echo "Testing audio setup..."
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print('Audio devices:', p.get_device_info_by_index(0)['name'])"

echo "Audio setup completed successfully!"
