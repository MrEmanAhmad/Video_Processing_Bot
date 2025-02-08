"""
Step 5: Audio generation module
Generates audio from commentary using Google Cloud Text-to-Speech
"""

import os
import logging
from pathlib import Path
from typing import Optional
from google.cloud import texttospeech
import json
import re

logger = logging.getLogger(__name__)

class AudioGenerator:
    """Handles audio generation using Google Cloud Text-to-Speech."""
    
    def __init__(self, google_credentials_path: str):
        """
        Initialize the AudioGenerator with Google Cloud credentials.
        
        Args:
            google_credentials_path: Path to Google Cloud credentials JSON file
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
        self.client = texttospeech.TextToSpeechClient()
        self.voice_params = {
            'name': 'en-GB-Journey-O',
            'language_code': 'en-GB'
        }
        
    def generate_audio(self, text: str, output_path: Path, target_duration: float) -> Optional[Path]:
        """
        Generate audio from text using specified voice parameters.
        
        Args:
            text: Text to convert to speech
            output_path: Path where the audio file should be saved
            target_duration: Target duration in seconds
            
        Returns:
            Path to the generated audio file if successful, None otherwise
        """
        try:
            # Create the parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove SSML tags for plain text
            plain_text = re.sub(r'<[^>]+>', '', text)  # Remove all XML/SSML tags
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()  # Normalize whitespace
            
            # Configure the synthesis input with plain text
            synthesis_input = texttospeech.SynthesisInput(text=plain_text)
            
            # Configure the voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.voice_params['language_code'],
                name=self.voice_params['name']
            )
            
            # Configure the audio output with basic settings only
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            
            # Perform the text-to-speech request
            logger.info(f"Generating audio for text: {plain_text[:100]}...")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Write the audio content to file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                
            logger.info(f"Successfully generated audio file: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return None
            
    def list_available_voices(self):
        """
        List all available English voices.
        Useful for debugging and checking available options.
        """
        try:
            voices = self.client.list_voices().voices
            english_voices = []
            
            for voice in voices:
                if any(language_code.startswith('en-') for language_code in voice.language_codes):
                    english_voices.append({
                        'name': voice.name,
                        'language_codes': voice.language_codes,
                        'ssml_gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                        'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
                    })
                    
            return english_voices
        except Exception as e:
            logger.error(f"Error listing voices: {str(e)}")
            return []

def execute_step(audio_script: str, output_dir: Path, style_name: str) -> Optional[Path]:
    """
    Execute audio generation step.
    
    Args:
        audio_script: Text to convert to speech
        output_dir: Directory to save generated audio
        style_name: Name of the commentary style used
        
    Returns:
        Path to the generated audio file if successful, None otherwise
    """
    logger.debug("Step 5: Generating audio...")
    
    # Load video metadata to get duration
    try:
        with open(output_dir / "video_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            video_duration = float(metadata.get('duration', 0))
    except Exception as e:
        logger.error(f"Could not load video duration: {e}")
        video_duration = 0
    
    # Initialize audio generator with Google Cloud credentials from credentials directory
    credentials_path = Path("credentials/google_credentials.json")
    if not credentials_path.exists():
        logger.error("Google credentials file not found in credentials directory")
        return None
        
    generator = AudioGenerator(str(credentials_path))
    
    # Generate audio file
    audio_file = output_dir / f"commentary_{style_name}.wav"
    result = generator.generate_audio(audio_script, audio_file, video_duration)
    
    if result:
        logger.debug(f"Audio generated successfully: {audio_file}")
        return audio_file
    else:
        logger.error("Audio generation failed")
        return None 