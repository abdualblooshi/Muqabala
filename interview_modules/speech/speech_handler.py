import speech_recognition as sr
from googletrans import Translator
import os
import wave
import pyaudio
from typing import Optional, Tuple, Callable
import logging
import subprocess
import platform
import sys
import requests
import json
import queue
import threading
from .speech_learner import SpeechLearner
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechHandler:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "deepseek-r1:7b"):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # DeepSeek configuration
        self.ollama_url = ollama_url
        self.model = model
        
        # Set up directories with absolute paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.temp_audio_dir = os.path.join(self.base_dir, "temp_audio")
        self.models_dir = os.path.join(self.base_dir, "tts_models")
        
        # Create directories with proper permissions
        for directory in [self.temp_audio_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
            try:
                os.chmod(directory, 0o777)  # Full permissions
            except Exception as e:
                logger.warning(f"Could not set permissions for {directory}: {e}")
        
        # Try to initialize microphone
        self._initialize_microphone()
        
        # Initialize text-to-speech
        self.tts_engine = None
        self.tts = None
        self.tts_ar = None
        self.use_piper = False
        
        # Try to initialize TTS systems
        self._initialize_tts()
        
        # Initialize translator
        self.translator = Translator()
        
        # Audio recording settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Initialize speech learner
        self.speech_learner = SpeechLearner()

    def _initialize_microphone(self):
        """Initialize microphone with proper error handling."""
        try:
            # List available microphones
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                raise RuntimeError("No microphones found")
            
            logger.info(f"Available microphones: {mics}")
            
            # Try to initialize microphone with specific device index
            for idx, mic_name in enumerate(mics):
                try:
                    self.microphone = sr.Microphone(device_index=idx)
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        logger.info(f"Successfully initialized microphone: {mic_name}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to initialize microphone {mic_name}: {e}")
            
            if not self.microphone:
                raise RuntimeError("Failed to initialize any microphone")
                
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {str(e)}")
            self.microphone = None

    def record_and_transcribe(self, language: str = 'en-US', duration: int = 10) -> Tuple[bool, str]:
        """Record audio for a fixed duration and transcribe it."""
        try:
            # Create a temporary file for recording
            temp_file = os.path.join(self.temp_audio_dir, "temp_recording.wav")
            
            # Record audio
            if not self.start_recording(temp_file, duration):
                return False, "Failed to record audio"
            
            # Transcribe the recorded audio
            try:
                with sr.AudioFile(temp_file) as source:
                    audio = self.recognizer.record(source)
                    initial_text = self.recognizer.recognize_google(audio, language=language)
                    
                    # Translate if Arabic
                    if language.startswith('ar'):
                        initial_text = self.translator.translate(initial_text, dest='en').text
                    
                    # Get current context from session state if available
                    import streamlit as st
                    context = ""
                    if "questions" in st.session_state and "chat_history" in st.session_state:
                        current_q_index = len(st.session_state.chat_history) // 2
                        if current_q_index < len(st.session_state.questions):
                            context = st.session_state.questions[current_q_index]
                    
                    # Refine with DeepSeek
                    refined_text = self._refine_with_deepseek(initial_text, context)
                    
                    return True, refined_text
                    
            except sr.UnknownValueError:
                return False, "Could not understand audio. Please speak clearly and try again."
            except sr.RequestError as e:
                return False, f"Speech recognition service error: {str(e)}"
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")
                
        except Exception as e:
            logger.error(f"Error in record_and_transcribe: {str(e)}")
            return False, str(e)

    def start_recording(self, output_file: str, duration: int = 5) -> bool:
        """Record audio for specified duration."""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
            
            frames = []
            for i in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            wf = wave.open(output_file, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in audio recording: {str(e)}")
            return False

    def _play_audio_file(self, file_path: str):
        """Play an audio file using pyaudio."""
        try:
            wf = wave.open(file_path, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                          channels=wf.getnchannels(),
                          rate=wf.getframerate(),
                          output=True)
            
            data = wf.readframes(self.CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(self.CHUNK)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")

    def _refine_with_deepseek(self, text: str, context: str = "") -> str:
        """Refine the recognized text using DeepSeek model and learned corrections."""
        try:
            # First, check for learned corrections
            suggested_text, confidence = self.speech_learner.suggest_correction(text, context)
            
            # If we have a high-confidence correction, use it
            if confidence > 0.8:
                logger.info(f"Using learned correction with confidence {confidence}")
                return suggested_text
            
            # Otherwise, use DeepSeek for refinement
            prompt = f"""As an AI language model, please help refine and improve this speech-to-text output
            while maintaining its original meaning. Make it more natural and grammatically correct.
            
            Context: {context}
            Original text: {text}
            
            Refined text:"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                refined_text = response.json().get("response", "").strip()
                # Store the correction for learning
                self.speech_learner.add_correction(text, refined_text, context)
                logger.info(f"DeepSeek refinement: {refined_text}")
                return refined_text
            return text
            
        except Exception as e:
            logger.error(f"Text refinement failed: {e}")
            return text

    def _initialize_tts(self):
        """Initialize text-to-speech systems."""
        try:
            # Try to initialize pyttsx3 as fallback
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                logger.info("pyttsx3 initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.tts_engine = None
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_engine = None 