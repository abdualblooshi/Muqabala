#!/bin/bash

echo "Setting up audio for WSL..."

# Install required packages
echo "Installing audio packages..."
sudo apt-get update
sudo apt-get install -y \
    pulseaudio \
    alsa-utils \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    libportaudio2 \
    libsndfile1 \
    espeak \
    espeak-ng

# Create PulseAudio config directory
echo "Configuring PulseAudio..."
mkdir -p ~/.config/pulse

# Configure PulseAudio for WSL
cat << EOF > ~/.config/pulse/default.pa
#!/usr/bin/pulseaudio -nF
load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1
load-module module-esound-protocol-tcp
load-module module-waveout
load-module module-null-sink
load-module module-rescue-streams
load-module module-always-sink
load-module module-intended-roles
load-module module-suspend-on-idle
load-module module-position-event-sounds
load-module module-role-cork
load-module module-filter-heuristics
load-module module-filter-apply
load-module module-switch-on-connect
EOF

# Create PulseAudio client config
cat << EOF > ~/.config/pulse/client.conf
default-server = unix:/run/user/1000/pulse/native
# Prevent a server running in the container
autospawn = no
daemon-binary = /bin/true
# Enable shared memory
enable-shm = true
EOF

# Kill any existing PulseAudio process
echo "Restarting PulseAudio..."
pulseaudio --kill || true
sleep 2

# Start PulseAudio in daemon mode
pulseaudio --start --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1" --exit-idle-time=-1 --daemon

# Wait for PulseAudio to start
sleep 2

# Test audio setup
echo "Testing audio setup..."
echo "You should hear a test sound..."
speaker-test -c2 -t wav -l1

# Test microphone
echo "Testing microphone..."
arecord -d 3 test.wav
aplay test.wav
rm test.wav

echo "Audio setup complete!"
echo "If you don't hear any sound or experience issues:"
echo "1. Make sure your Windows host has audio enabled"
echo "2. Try restarting the WSL terminal"
echo "3. Run 'pulseaudio --start' if audio stops working"
echo "4. Check microphone permissions in Windows" 