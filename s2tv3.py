import whisper
import sounddevice as sd
import numpy as np
import wave
from gtts import gTTS
import os
import pygame
from pathlib import Path
import tempfile

class AudioTranscriptionSystem:
    def __init__(self, 
                 sample_rate=16000,
                 channels=1,
                 model_size="base",
                 language="en"):
        """
        Initialize the audio transcription system.
        
        Args:
            sample_rate (int): Audio sampling rate in Hz
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
            model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large")
            language (str): Language code for speech synthesis
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.model_size = model_size
        self.language = language
        self.model = None
        self.tempfile = tempfile
        self.Path = Path
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
    def load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            print(f"Loading {self.model_size} Whisper model...")
            self.model = whisper.load_model(self.model_size)
            print("Model loaded successfully.")
    
    def record_audio(self, duration):
        """
        Record audio from the microphone.
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        try:
            print(f"Recording for {duration} seconds...")
            audio_data = sd.rec(
                int(self.sample_rate * duration),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()
            print("Recording complete.")
            return audio_data
        except Exception as e:
            raise RuntimeError(f"Error recording audio: {str(e)}")
    
    def save_audio(self, audio_data, file_path):
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data (numpy.ndarray): Audio data to save
            file_path (str): Output file path
        """
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        self.load_model()
        try:
            result = self.model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Error transcribing audio: {str(e)}")
    
    def text_to_speech(self, text, output_file):
        """
        Convert text to speech.
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Output audio file path
        """
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_file)
        except Exception as e:
            raise RuntimeError(f"Error converting text to speech: {str(e)}")
    
    def play_audio(self, audio_file):
        """
        Play audio file using pygame mixer.
        
        Args:
            audio_file (str): Path to audio file to play
        """
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            raise RuntimeError(f"Error playing audio: {str(e)}")
    
    def process(self, duration=5):
        """
        Complete process: record, transcribe, and play back.
        
        Args:
            duration (float): Recording duration in seconds
        
        Returns:
            str: Transcribed text
        """
        with self.tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary file paths
            temp_dir = self.Path(temp_dir)
            input_file = temp_dir / "input.wav"
            output_file = temp_dir / "output.mp3"
            
            try:
                # Record and save audio
                audio_data = self.record_audio(duration)
                self.save_audio(audio_data, str(input_file))
                
                # Transcribe audio
                transcribed_text = self.transcribe_audio(str(input_file))
                print(f"Transcription: {transcribed_text}")
                
                # Convert transcription to speech and play
                self.text_to_speech(transcribed_text, str(output_file))
                print("Playing transcribed text...")
                self.play_audio(str(output_file))
                
                return transcribed_text
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                return None

def main():
    # Example usage
    system = AudioTranscriptionSystem(
        sample_rate=16000,
        channels=1,
        model_size="base",
        language="en"
    )
    
    try:
        transcribed_text = system.process(duration=5)
        if transcribed_text:
            print("\nProcess completed successfully!")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()