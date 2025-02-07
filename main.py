import os
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
import ffmpeg
import cloudinary
import cloudinary.uploader
import cloudinary.api
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from google.cloud import texttospeech
from google.oauth2 import service_account
import openai
import json
import subprocess
import re
import emoji
import asyncio
import time
import shutil
from datetime import datetime
import requests
import atexit
import sys
import logging.handlers

# Load environment variables
load_dotenv()

# Configure custom logging format with colors and better structure
class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logging with the custom formatter
logger = logging.getLogger("TwitterVideoBot")
logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))

# Create console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler for persistent logging with rotation
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'bot_{datetime.now().strftime("%Y%m%d")}.log')

# Use RotatingFileHandler instead of FileHandler
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'  # Explicitly set encoding
)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

console_handler.setFormatter(ColoredFormatter(console_format))
file_handler.setFormatter(logging.Formatter(file_format))

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Suppress other loggers' output
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Register cleanup function
def cleanup_logs():
    try:
        # Remove handlers first
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing log handler: {e}", file=sys.stderr)
                
        # Now try to remove old log files
        if os.path.exists(log_dir):
            current_log = os.path.basename(log_file)
            for old_log in os.listdir(log_dir):
                if old_log != current_log and old_log.endswith('.log'):
                    try:
                        os.remove(os.path.join(log_dir, old_log))
                    except Exception as e:
                        print(f"Error removing old log {old_log}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error in cleanup_logs: {e}", file=sys.stderr)

atexit.register(cleanup_logs)

# Log startup information
logger.info("=== Twitter Video Processing Bot Starting ===")
logger.info(f"Log file created at: {log_file}")

# Initialize API clients and configurations
try:
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET')
    )
    logger.info("✓ Cloudinary configuration initialized")
except Exception as e:
    logger.error("✗ Failed to initialize Cloudinary configuration", exc_info=True)
    raise

# Initialize OpenAI client
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    logger.info("✓ OpenAI API key configured")
except Exception as e:
    logger.error("✗ Failed to configure OpenAI API key", exc_info=True)
    raise

# Initialize Google Cloud TTS client with explicit credentials
try:
    # Try using environment variable directly first
    creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if creds_json:
        try:
            logger.info("🔄 Initializing Google Cloud credentials from environment variable")
            
            # Try parsing with different methods
            try:
                # First try normal parsing
                creds_dict = json.loads(creds_json)
            except json.JSONDecodeError:
                try:
                    # Try removing quotes and escapes
                    cleaned_json = creds_json.strip('"\'').replace('\\"', '"')
                    creds_dict = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    try:
                        # Try with strict=False
                        creds_dict = json.loads(creds_json, strict=False)
                    except json.JSONDecodeError as je:
                        # If still failing, try more aggressive cleaning
                        cleaned_json = creds_json.encode().decode('unicode_escape')
                        cleaned_json = cleaned_json.strip('"\'')
                        # Remove any duplicate backslashes
                        cleaned_json = re.sub(r'\\+', r'\\', cleaned_json)
                        try:
                            creds_dict = json.loads(cleaned_json, strict=False)
                        except json.JSONDecodeError as final_je:
                            logger.error("✗ All JSON parsing attempts failed")
                            logger.error(f"Original JSON string: {creds_json[:100]}...")
                            logger.error(f"Cleaned JSON string: {cleaned_json[:100]}...")
                            raise final_je
            
            logger.info("✓ JSON credentials parsed successfully")
            
            # Verify required fields
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in creds_dict]
            if missing_fields:
                raise ValueError(f"Missing required fields in credentials: {missing_fields}")
            
            # Create credentials directory if it doesn't exist
            credentials_dir = '/app/credentials'
            os.makedirs(credentials_dir, exist_ok=True)
            
            # Write credentials to file for persistence
            credentials_path = os.path.join(credentials_dir, 'google_credentials.json')
            with open(credentials_path, 'w') as f:
                json.dump(creds_dict, f, indent=2)
            logger.info(f"Credentials file created successfully at {credentials_path}")
            
            # Set environment variable to point to the file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            # Create credentials object directly from dictionary
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            
            # Initialize client with credentials
            tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            
            # Test the credentials with a simple API call
            voices = tts_client.list_voices()
            logger.info("✓ Google Cloud TTS client initialized and verified successfully")
            
        except json.JSONDecodeError as je:
            logger.error("✗ Failed to parse JSON credentials")
            logger.error(f"JSON parsing error at position {je.pos}: {je.msg}")
            logger.error(f"Invalid JSON string: {creds_json[:100]}...")  # Log first 100 chars
            raise
                
    else:
        # Fallback to file-based credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', "/app/credentials/google_credentials.json")
        if os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
                # Verify credentials
                voices = tts_client.list_voices()
                logger.info("✓ Google Cloud TTS client initialized from credentials file")
            except Exception as e:
                logger.error(f"✗ Failed to initialize from credentials file: {str(e)}")
                raise
        else:
            logger.error("✗ No Google Cloud credentials found in environment or file system")
            raise Exception("No Google Cloud credentials found")
            
except Exception as e:
    logger.error("✗ Failed to initialize Google Cloud TTS client", exc_info=True)
    raise

class TwitterVideoProcessor:
    def __init__(self):
        self.temp_dir = os.path.join('/tmp', f'twitter_processor_{int(time.time())}')
        os.makedirs(self.temp_dir, exist_ok=True)
        # Initialize OpenAI client with stricter timeouts
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=15.0,  # Reduced from 30
            max_retries=2  # Reduced from 3
        )
        # Initialize DeepSeek client with stricter timeouts
        self.deepseek_client = openai.OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1",
            timeout=15.0,  # Reduced from 30
            max_retries=1  # Reduced from 3
        )
        # Initialize TTS client
        try:
            creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if creds_json:
                try:
                    # Clean up the JSON string if needed
                    creds_json = creds_json.strip('"').replace('\\"', '"')  # Remove outer quotes and unescape inner quotes
                    
                    # Try parsing with different methods
                    try:
                        # First try normal parsing
                        creds_dict = json.loads(creds_json)
                    except json.JSONDecodeError:
                        try:
                            # Try with strict=False to handle control characters
                            creds_dict = json.loads(creds_json, strict=False)
                        except json.JSONDecodeError as je:
                            # If still failing, try to clean up the string more aggressively
                            creds_json = creds_json.encode().decode('unicode_escape')
                            creds_dict = json.loads(creds_json, strict=False)
                    
                    # Verify required fields
                    required_fields = ['type', 'project_id', 'private_key', 'client_email']
                    missing_fields = [field for field in required_fields if field not in creds_dict]
                    if missing_fields:
                        raise ValueError(f"Missing required fields in credentials: {missing_fields}")
                    
                    # Create credentials directory if it doesn't exist
                    credentials_dir = '/app/credentials'
                    os.makedirs(credentials_dir, exist_ok=True)
                    
                    # Write credentials to file for persistence
                    credentials_path = os.path.join(credentials_dir, 'google_credentials.json')
                    with open(credentials_path, 'w') as f:
                        json.dump(creds_dict, f, indent=2)
                    logger.info(f"Credentials file created successfully at {credentials_path}")
                    
                    # Set environment variable
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                    
                    # Create credentials and initialize client
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
                    
                    # Verify credentials
                    self.tts_client.list_voices()
                    logger.info("✓ TTS client initialized with environment credentials")
                except json.JSONDecodeError as je:
                    logger.error("✗ Failed to parse JSON credentials")
                    logger.error(f"JSON parsing error at position {je.pos}: {je.msg}")
                    logger.error(f"Invalid JSON string: {creds_json[:100]}...")  # Log first 100 chars
                    raise
                except Exception as e:
                    logger.error(f"Failed to initialize TTS client from environment: {e}")
                    raise
            else:
                # Try file-based credentials
                credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', "/app/credentials/google_credentials.json")
                if os.path.exists(credentials_path):
                    try:
                        credentials = service_account.Credentials.from_service_account_file(credentials_path)
                        self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
                        # Verify credentials
                        self.tts_client.list_voices()
                        logger.info("✓ TTS client initialized from credentials file")
                    except Exception as e:
                        logger.error(f"Failed to initialize from credentials file: {e}")
                        raise
                else:
                    raise Exception("No Google Cloud credentials found")
        except Exception as e:
            logger.error(f"Failed to initialize TTS client: {e}")
            raise
        # Initialize context dictionary for workflow tracking
        self.context = {
            "temp_dir": self.temp_dir,
            "cloudinary_resources": [],  # Track resources for cleanup
            "error": None,  # Track any errors
            "stage": "initialized",  # Track current processing stage
            "watermark": "🎥 Created by @AutomatorByMani | Share & Enjoy!",  # Default watermark text
            "cloudinary_resources_tracked": set()  # Track all Cloudinary resources with types
        }
        
        # Keep animations separate from context
        self.loading_animations = {
            'download': [
                "🎬 Fetching video ⏳",
                "🎬 Downloading content 📥",
                "🎬 Getting tweet data 🔄",
                "🎬 Almost there 📩",
            ],
            'analyze': [
                "🧠 Analyzing frame 1 🔍",
                "🧠 Processing visuals 👀",
                "🧠 Understanding context 🤔",
                "🧠 Generating insights ✨",
            ],
            'speech': [
                "🎙️ Preparing narration 🗣️",
                "🎙️ Crafting voice-over 🎵",
                "🎙️ Fine-tuning audio 🎧",
                "🎙️ Polishing sound 🔊",
            ],
            'merge': [
                "✨ Preparing video canvas...",
                "✨ Adding padding and frames...",
                "✨ Adjusting dimensions...",
                "✨ Fine-tuning format...",
            ],
            'merge_audio': [
                "🎵 Mixing audio tracks...",
                "🎵 Balancing sound levels...",
                "🎵 Syncing narration...",
                "🎵 Perfecting audio blend...",
            ],
            'final_touch': [
                "🎨 Adding final touches ✨",
                "🎨 Polishing transitions 🌟",
                "🎨 Making it perfect 💫",
                "🎨 Almost ready 🌈",
            ],
            'upload': [
                "📤 Preparing upload 📡",
                "📤 Sending your way 🚀",
                "📤 Almost ready ⭐",
                "📤 Final touches 💫",
            ]
        }
        
    async def animate_loading(self, message_obj, animation_key: str, duration: float = 2.0):
        """Animate loading message while processing"""
        try:
            start_time = time.time()
            animation_frames = self.loading_animations[animation_key]
            frame_index = 0
            
            while True:
                try:
                    # Check if we've been running longer than expected
                    if duration and (time.time() - start_time > duration):
                        break
                        
                    await message_obj.edit_text(animation_frames[frame_index])
                    frame_index = (frame_index + 1) % len(animation_frames)
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    # Clean exit on cancellation
                    break
                except Exception as e:
                    logger.debug(f"Animation frame update skipped: {e}")
                    await asyncio.sleep(0.5)
                    continue
                    
        except Exception as e:
            logger.debug(f"Animation ended: {e}")
        finally:
            try:
                # Try to set final frame on exit
                if animation_frames:
                    await message_obj.edit_text(animation_frames[-1])
            except Exception:
                pass

    async def download_tweet(self, url: str, message_obj) -> bool:
        """Download tweet video and extract metadata."""
        try:
            self.context["stage"] = "downloading"
            logger.info("Starting processing...")
            logger.info("Test 1: Testing tweet download...")

            # Create downloads directory inside temp_dir
            downloads_dir = os.path.join(self.temp_dir, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            # Convert x.com to twitter.com
            url = url.replace('x.com', 'twitter.com')

            # Extract tweet ID
            tweet_id = url.split('/')[-1].split('?')[0]
            logger.info(f"Extracted tweet ID: {tweet_id}")

            # First try with API v2 endpoint
            api_url = f"https://api.twitter.com/2/tweets/{tweet_id}?expansions=attachments.media_keys&media.fields=variants,url,type"
            headers = {
                'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs=1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'x-guest-token': None
            }

            # Get guest token
            try:
                guest_token_url = 'https://api.twitter.com/1.1/guest/activate.json'
                guest_token_response = requests.post(guest_token_url, headers={'Authorization': headers['Authorization']})
                if guest_token_response.status_code == 200:
                    headers['x-guest-token'] = guest_token_response.json().get('guest_token')
                    logger.info("Successfully obtained guest token")
            except Exception as e:
                logger.warning(f"Failed to get guest token: {e}")

            # Configure yt-dlp with updated settings
            options = {
                'outtmpl': os.path.join(downloads_dir, '%(id)s.%(ext)s'),
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {
                    'twitter': {
                        'api': ['graphql', 'syndication', 'api'],
                    }
                },
                'http_headers': headers,
                'cookiesfrombrowser': ('chrome',),  # Try to use Chrome cookies if available
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'retry_sleep_functions': {'http': lambda n: 1 + n * 2},  # Exponential backoff
            }

            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    # Try direct download first
                    try:
                        info = ydl.extract_info(url, download=True)
                    except Exception as e:
                        logger.warning(f"Direct download failed, trying with syndication API: {e}")
                        # Try with syndication API
                        options['extractor_args']['twitter']['api'] = ['syndication']
                        info = ydl.extract_info(url, download=True)

                    video_filename = ydl.prepare_filename(info)
                    video_filename = os.path.splitext(video_filename)[0] + ".mp4"

                    # Get tweet text from info or API
                    tweet_text = info.get('description', '')
                    if not tweet_text:
                        try:
                            tweet_response = requests.get(api_url, headers=headers)
                            if tweet_response.status_code == 200:
                                tweet_data = tweet_response.json()
                                tweet_text = tweet_data.get('data', {}).get('text', 'No text found')
                        except Exception as e:
                            logger.warning(f"Failed to get tweet text from API: {e}")
                            tweet_text = "No text found"

                    # Store the paths and metadata in context
                    self.context["video_path"] = video_filename
                    self.context["tweet_text"] = tweet_text
                    logger.info(f"Downloaded video: {video_filename}")
                    logger.info(f"Tweet text: {tweet_text}")

                    return True

            except Exception as e:
                error_message = str(e)
                logger.error(f"Download failed: {error_message}")
                
                # Try one last time with minimal options
                try:
                    logger.info("Attempting final download method...")
                    minimal_options = {
                        'outtmpl': os.path.join(downloads_dir, '%(id)s.%(ext)s'),
                        'format': 'best[ext=mp4]/best',
                        'quiet': True,
                    }
                    with yt_dlp.YoutubeDL(minimal_options) as ydl:
                        info = ydl.extract_info(url, download=True)
                        video_filename = ydl.prepare_filename(info)
                        video_filename = os.path.splitext(video_filename)[0] + ".mp4"
                        
                        self.context["video_path"] = video_filename
                        self.context["tweet_text"] = info.get('description', 'No text found')
                        logger.info("Final download method successful")
                        return True
                except Exception as final_e:
                    logger.error(f"Final download method failed: {final_e}")
                    if "Unable to extract uploader id" in str(final_e):
                        await message_obj.edit_text("⚠️ This tweet might be from a private account or has been deleted.")
                    else:
                        await message_obj.edit_text("⚠️ Failed to download the video. The tweet might be unavailable or protected.")
                    return False

        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            await message_obj.edit_text("❌ An error occurred while processing the video.")
            return False

    async def extract_frame(self, message_obj) -> bool:
        """Extract a representative frame from the video using FFmpeg"""
        self.context["stage"] = "extracting_frame"
        await message_obj.edit_text("📸 Extracting video frame...")
        
        frame_path = os.path.join(self.temp_dir, "frame.jpg")
        try:
            stream = (
                ffmpeg
                .input(self.context["video_path"])
                .filter('select', 'gte(n,10)')
                .output(frame_path, vframes=1)
                .overwrite_output()
            )
            
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            self.context["frame_path"] = frame_path
            return True
            
        except ffmpeg.Error as e:
            self.context["error"] = f"Frame extraction failed: {e.stderr.decode()}"
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

    async def upload_to_cloudinary(self, message_obj) -> bool:
        """Upload frame to Cloudinary and return success"""
        self.context["stage"] = "uploading_frame"
        await message_obj.edit_text("☁️ Uploading frame to cloud storage...")
        
        try:
            result = cloudinary.uploader.upload(self.context["frame_path"])
            self.context["frame_url"] = result['secure_url']
            self.context["cloudinary_resources"].append(result['public_id'])
            return True
        except Exception as e:
            self.context["error"] = f"Cloudinary upload failed: {str(e)}"
            logger.error(f"Cloudinary upload error: {e}")
            raise

    async def analyze_content(self, message_obj) -> bool:
        """Analyze content using Vision API and generate comment"""
        self.context["stage"] = "analyzing"
        animation_task = asyncio.create_task(self.animate_loading(message_obj, 'analyze'))
        
        try:
            await message_obj.edit_text("🧠 Analyzing content...")
            
            # Get video duration using ffprobe
            try:
                probe = ffmpeg.probe(self.context["video_path"])
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                video_duration = float(video_info.get('duration', 30.0))
            except Exception as e:
                logger.warning(f"Failed to get video duration: {e}, using default")
                video_duration = 30.0
            
            tweet_text = self.context.get("tweet_text", "")
            frame_url = self.context.get("frame_url")
            
            if not frame_url:
                logger.error("No frame URL available for analysis")
                self.context["error"] = "No frame URL available for analysis"
                return False
            
            max_words = max(15, min(35, int(video_duration * 2.5)))
            logger.info(f"Video duration: {video_duration}s, Maximum words allowed: {max_words}")

            # First get vision analysis
            try:
                logger.info("Getting vision analysis...")
                vision_response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Analyze this video frame and tweet text: {tweet_text}"},
                                    {"type": "image_url", "image_url": {"url": frame_url}}
                                ]
                            }
                        ]
                    ),
                    timeout=20  # 20 second timeout for vision
                )
                
                frame_analysis = vision_response.choices[0].message.content
                if not frame_analysis:
                    raise Exception("No analysis generated from Vision API")
                
                # Try DeepSeek first for text generation
                try:
                    logger.info("Using DeepSeek for comment generation...")
                    comment_response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.deepseek_client.chat.completions.create,
                            model="deepseek-chat",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"""Generate an enthusiastic, engaging comment that sounds like a friendly person sharing their genuine excitement.
                                    Key requirements:
                                    1. Use natural speech patterns but keep them positive
                                    2. Show genuine enthusiasm and warmth
                                    3. Keep it under {max_words} words
                                    4. Use friendly, conversational language
                                    5. Express sincere appreciation"""
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on this analysis:\n{frame_analysis}\nTweet text: {tweet_text}\nGenerate an enthusiastic, friendly comment."
                                }
                            ],
                            max_tokens=150,
                            temperature=0.8
                        ),
                        timeout=15  # 15 second timeout for DeepSeek
                    )
                    generated_comment = comment_response.choices[0].message.content.strip()
                    
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"DeepSeek failed: {str(e)}, falling back to OpenAI")
                    # Fall back to OpenAI for text generation
                    comment_response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.openai_client.chat.completions.create,
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"Generate a natural {max_words}-word enthusiastic comment."
                                },
                                {
                                    "role": "user",
                                    "content": f"Based on this analysis:\n{frame_analysis}\nGenerate an engaging comment."
                                }
                            ],
                            max_tokens=100,
                            temperature=0.7
                        ),
                        timeout=15  # 15 second timeout for OpenAI
                    )
                    generated_comment = comment_response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"Vision API or text generation failed: {str(e)}")
                raise Exception("Failed to generate content analysis")

            # Verify and clean up comment
            word_count = len(generated_comment.split())
            if word_count > max_words:
                generated_comment = ' '.join(generated_comment.split()[:max_words])
                word_count = max_words
            
            logger.info(f"Generated comment ({word_count} words): {generated_comment}")
            estimated_duration = word_count / 2.5
            logger.info(f"Estimated narration duration: {estimated_duration:.1f}s")
            
            self.context["comment"] = generated_comment
            return True

        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__} - {str(e)}")
            self.context["error"] = f"Analysis failed: {str(e)}"
            raise e
        finally:
            animation_task.cancel()

    def clean_text_for_narration(self, text: str) -> str:
        """Clean text by removing quotes, hashtags, emojis, and numbers for better narration"""
        # Remove quotes (both single and double)
        text = text.strip('"\'')
        text = re.sub(r'[""""]', '', text)  # Remove fancy quotes
        text = re.sub(r'[\''']', '', text)  # Remove fancy apostrophes
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove square brackets and their contents (like [pause])
        text = re.sub(r'\[.*?\]', '', text)
        
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'[\(\)]', '', text)  # Remove parentheses
        
        return text.strip()

    async def run_ffmpeg_async(self, stream):
        """Run FFmpeg command asynchronously with improved error handling"""
        try:
            process = ffmpeg.run_async(stream, pipe_stderr=True)
            loop = asyncio.get_running_loop()
            
            # Use asyncio.shield to protect the process communication from cancellation
            out, err = await asyncio.shield(
                loop.run_in_executor(None, process.communicate)
            )
            
            if process.returncode != 0:
                error_message = err.decode() if err else 'No error output'
                logger.error(f"FFmpeg error output: {error_message}")
                raise Exception(f"FFmpeg command failed with return code {process.returncode}")
            
            return out, err
            
        except asyncio.CancelledError:
            logger.warning("FFmpeg process was cancelled, cleaning up...")
            if process and process.poll() is None:
                process.kill()
                await loop.run_in_executor(None, process.wait)
            raise
        except Exception as e:
            logger.error(f"FFmpeg error: {str(e)}")
            if process and process.poll() is None:
                process.kill()
                await loop.run_in_executor(None, process.wait)
            raise

    def verify_video_format(self, video_path: str) -> dict:
        """Verify video format meets requirements"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            aspect_ratio = height / width
            
            format_info = {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "is_reel_ratio": abs(aspect_ratio - 16/9) < 0.1,  # Allow small deviation
                "has_frame": width % 40 == 0 and height % 40 == 0  # Check if dimensions include 20px frame on each side
            }
            logger.info(f"Video format verification: {format_info}")
            return format_info
        except Exception as e:
            logger.error(f"Error verifying video format: {e}")
            return None

    def save_output_video(self) -> str:
        """Save the final video to an organized output directory"""
        try:
            # Use /tmp for container storage
            output_dir = os.path.join('/tmp', 'outputs', datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "final_video.mp4")
            shutil.copy2(self.context["output_video_path"], output_path)
            logger.info(f"Saved final video to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving output video: {e}")
            return self.context["output_video_path"]

    async def verify_cleanup(self) -> bool:
        """Verify all resources have been properly cleaned up"""
        cleanup_success = True
        
        # 1. Verify Cloudinary resources
        try:
            for resource_id, resource_type in list(self.context["cloudinary_resources_tracked"]):
                try:
                    # Try to get the resource - should raise NotFound if cleaned
                    cloudinary.api.resource(resource_id, resource_type=resource_type)
                    # If we get here, resource still exists - try to clean it
                    logger.warning(f"Found uncleaned resource {resource_id}, attempting cleanup...")
                    cloudinary.uploader.destroy(resource_id, resource_type=resource_type)
                except cloudinary.api.NotFound:
                    logger.info(f"Verified cleanup of {resource_id}")
                except Exception as e:
                    logger.error(f"Error verifying/cleaning resource {resource_id}: {e}")
                    cleanup_success = False
        except Exception as e:
            logger.error(f"Error in Cloudinary cleanup verification: {e}")
            cleanup_success = False

        # 2. Verify local directories
        local_paths = [
            self.temp_dir,
            os.path.join('/tmp', 'outputs'),
            os.path.join(self.temp_dir, "downloads"),
            os.path.join(self.temp_dir, "audio_output")
        ]
        
        for path in local_paths:
            if path and os.path.exists(path):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    if os.path.exists(path):
                        logger.error(f"Failed to remove directory: {path}")
                        cleanup_success = False
                    else:
                        logger.info(f"Verified cleanup of {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up {path}: {e}")
                    cleanup_success = False

        return cleanup_success

    async def cleanup_resources(self):
        """Clean up all resources including Cloudinary and local files"""
        try:
            logger.info("Starting cleanup process...")
            
            # 1. Clean up Cloudinary resources first
            for resource_id, resource_type in list(self.context["cloudinary_resources_tracked"]):
                try:
                    cloudinary.uploader.destroy(resource_id, resource_type=resource_type)
                    logger.info(f"Cleaned up Cloudinary resource: {resource_id} ({resource_type})")
                except cloudinary.api.NotFound:
                    logger.info(f"Resource already cleaned up: {resource_id} ({resource_type})")
                except Exception as e:
                    logger.error(f"Error cleaning up Cloudinary resource {resource_id}: {e}")
            
            # 2. Clean up all local directories
            local_paths = [
                self.temp_dir,
                os.path.join('/tmp', 'outputs'),
                os.path.join(self.temp_dir, "downloads"),
                os.path.join(self.temp_dir, "audio_output")
            ]
            
            for path in local_paths:
                if path and os.path.exists(path):
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                        logger.info(f"Cleaned up directory: {path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up directory {path}: {e}")
            
            # 3. Verify cleanup
            cleanup_success = await self.verify_cleanup()
            if not cleanup_success:
                logger.warning("Some resources may not have been properly cleaned up")
            else:
                logger.info("All resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error in cleanup process: {e}")
            raise

    async def add_padding_and_frame(self, video_path: str, message_obj) -> str:
        """Add padding to make video reel-sized (9:16) and add white frame using FFmpeg directly"""
        await message_obj.edit_text("🎨 Formatting video for reels...")
        
        try:
            # Get original video dimensions and verify
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Store video info
            self.context["video_info"] = {
                "original_width": width,
                "original_height": height,
                "duration": float(video_info.get('duration', 0))
            }
            
            logger.info(f"Original video dimensions: {width}x{height}")
            
            # Calculate target dimensions
            if height > width:
                target_width = width + 40
                target_height = int((target_width * 16) / 9)
            else:
                target_height = height + 40
                target_width = int((target_height * 9) / 16)
            
            logger.info(f"Target dimensions: {target_width}x{target_height}")
            
            # Create padded video using FFmpeg directly
            padded_video = os.path.join(self.temp_dir, "padded_video.mp4")
            
            # Use FFmpeg to add padding and preserve audio
            stream = (
                ffmpeg
                .input(video_path)
                .filter('pad', 
                       width=target_width,
                       height=target_height,
                       x='(out_w-in_w)/2',
                       y='(out_h-in_h)/2',
                       color='white')
                .output(padded_video,
                       acodec='copy',  # Copy audio stream without re-encoding
                       vcodec='libx264',
                       preset='ultrafast',
                       movflags='+faststart')
                .overwrite_output()
            )
            
            # Run FFmpeg command
            await self.run_ffmpeg_async(stream)
            
            # Verify the output
            if not os.path.exists(padded_video):
                raise Exception("Failed to create padded video")
            
            # Verify format
            format_info = self.verify_video_format(padded_video)
            if format_info and not format_info["is_reel_ratio"]:
                logger.warning("Padded video does not have correct reel ratio")
                if abs(format_info["aspect_ratio"] - 16/9) > 0.1:
                    logger.info("Attempting to fix aspect ratio...")
                    fixed_video = os.path.join(self.temp_dir, "fixed_video.mp4")
                    fix_stream = (
                        ffmpeg
                        .input(padded_video)
                        .filter('scale', width='-1', height='1351')
                        .output(fixed_video,
                               acodec='copy',  # Preserve audio
                               preset='ultrafast',
                               movflags='+faststart')
                        .overwrite_output()
                    )
                    await self.run_ffmpeg_async(fix_stream)
                    padded_video = fixed_video
            
            return padded_video
                
        except Exception as e:
            logger.error(f"Error adding padding: {e}")
            raise

    async def show_progress_animation(self, message_obj, animation_key: str, duration: float = None):
        """Show progress animation with optional duration"""
        try:
            animation_task = asyncio.create_task(self.animate_loading(message_obj, animation_key))
            if duration:
                try:
                    await asyncio.wait_for(animation_task, timeout=duration)
                except asyncio.TimeoutError:
                    pass
            return animation_task
        except Exception as e:
            logger.debug(f"Progress animation creation failed: {e}")
            return None

    async def merge_audio_video(self, message_obj) -> bool:
        """Merge audio with video using FFmpeg with proper audio ducking"""
        output_path = os.path.join(self.temp_dir, "final_video.mp4")
        try:
            # Show initial merge animation
            merge_task = await self.show_progress_animation(message_obj, 'merge')
            
            try:
                # First add padding and frame
                padded_video = await self.add_padding_and_frame(self.context["video_path"], message_obj)
                merge_task.cancel()
                
                # Show audio mixing animation
                audio_task = await self.show_progress_animation(message_obj, 'merge_audio')
                
                try:
                    # Verify we have the necessary files
                    if not os.path.exists(self.context["audio_path"]):
                        logger.error("Audio file not found")
                        self.context["error"] = "Audio file not found"
                        return False
                    
                    if not os.path.exists(padded_video):
                        logger.error("Padded video file not found")
                        self.context["error"] = "Padded video file not found"
                        return False
                    
                    # Check if video has audio stream
                    video_probe = ffmpeg.probe(padded_video)
                    has_audio = any(stream['codec_type'] == 'audio' for stream in video_probe['streams'])
                    
                    # Create FFmpeg command based on whether video has audio
                    video_stream = ffmpeg.input(padded_video)
                    narration_stream = ffmpeg.input(self.context["audio_path"])
                    
                    if has_audio:
                        logger.info("Video has audio stream, mixing with narration")
                        video_audio = video_stream.audio.filter('volume', 0.3)
                        narration_audio = narration_stream.filter('volume', 1.0)
                        mixed_audio = ffmpeg.filter(
                            [video_audio, narration_audio],
                            'amix',
                            inputs=2,
                            duration='first',
                            dropout_transition=0.5
                        )
                    else:
                        logger.info("Video has no audio stream, using narration only")
                        mixed_audio = narration_stream.filter('volume', 1.0)
                    
                    # Create the output stream with optimized settings
                    stream = (
                        ffmpeg
                        .output(
                            video_stream.video,
                            mixed_audio,
                            output_path,
                            acodec='aac',
                            vcodec='libx264',
                            strict='experimental',
                            movflags='+faststart',
                            preset='ultrafast',
                            threads='auto'
                        )
                        .overwrite_output()
                    )
                    
                    # Run the FFmpeg command with timeout
                    await asyncio.wait_for(
                        self.run_ffmpeg_async(stream),
                        timeout=300  # 5 minutes timeout
                    )
                    
                    # Verify the output
                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        logger.error("Failed to create output video or file is empty")
                        self.context["error"] = "Failed to create output video"
                        return False
                    
                    # Verify audio mixing
                    output_probe = ffmpeg.probe(output_path)
                    output_audio = next((s for s in output_probe['streams'] if s['codec_type'] == 'audio'), None)
                    if not output_audio:
                        logger.error("Output video has no audio stream")
                        self.context["error"] = "Output video has no audio"
                        return False
                    
                    self.context["output_video_path"] = output_path
                    return True
                    
                except Exception as e:
                    logger.error(f"Error in merge_audio_video: {e}")
                    self.context["error"] = f"Error in audio mixing: {str(e)}"
                    return False
                finally:
                    audio_task.cancel()
                
            except Exception as e:
                logger.error(f"Error in padding/frame addition: {str(e)}")
                self.context["error"] = f"Error in video padding: {str(e)}"
                return False
                
        except Exception as e:
            logger.error(f"Error in merge_audio_video: {e}")
            self.context["error"] = f"Error in video processing: {str(e)}"
            return False

    async def generate_speech(self, message_obj) -> bool:
        """Convert text to speech using Google Cloud TTS"""
        animation_task = asyncio.create_task(self.animate_loading(message_obj, 'speech'))
        try:
            await message_obj.edit_text("🔊 Generating speech...")
            
            # Get comment from context
            comment = self.context.get("comment")
            if not comment:
                logger.error("No comment found in context")
                self.context["error"] = "No comment found in context"
                return False
            
            # Clean text for narration
            narration_text = self.clean_text_for_narration(comment)
            logger.info(f"Original text for narration: {narration_text}")
            
            if not narration_text:
                logger.error("No text available for narration")
                self.context["error"] = "No text available for narration"
                return False

            # Create output directory in temp_dir
            output_dir = os.path.join(self.temp_dir, "audio_output")
            os.makedirs(output_dir, exist_ok=True)

            # Configure voice parameters
            voice_params = {
                'name': 'en-GB-Journey-O',
                'language_codes': ['en-GB']
            }

            # Configure synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=narration_text)
            
            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_params['language_codes'][0],
                name=voice_params['name']
            )

            # Configure audio
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            # Generate speech
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save the audio file
            audio_path = os.path.join(output_dir, f"{voice_params['name']}.wav")
            with open(audio_path, "wb") as out:
                out.write(response.audio_content)

            # Store the audio path in context
            self.context["audio_path"] = audio_path
            
            # Verify the audio file was created and has content
            if not os.path.exists(audio_path):
                logger.error("Failed to generate audio file")
                self.context["error"] = "Failed to generate audio file"
                return False
                
            if os.path.getsize(audio_path) == 0:
                logger.error("Generated audio file is empty")
                self.context["error"] = "Generated audio file is empty"
                return False
                
            logger.info(f"Generated audio file: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in generate_speech: {e}")
            self.context["error"] = f"Error in speech generation: {str(e)}"
            return False
        finally:
            animation_task.cancel()

    def track_cloudinary_resource(self, resource_id: str, resource_type: str = "image"):
        """Track a Cloudinary resource for cleanup verification"""
        self.context["cloudinary_resources_tracked"].add((resource_id, resource_type))
        logger.info(f"Tracking Cloudinary resource: {resource_id} ({resource_type})")

    async def verify_cloudinary_cleanup(self) -> bool:
        """Verify all tracked Cloudinary resources were cleaned up"""
        try:
            # First attempt to clean up any remaining resources
            for resource_id, resource_type in self.context["cloudinary_resources_tracked"]:
                try:
                    cloudinary.uploader.destroy(resource_id, resource_type=resource_type)
                    logger.info(f"Cleaned up remaining resource: {resource_id} ({resource_type})")
                except cloudinary.api.NotFound:
                    logger.info(f"Verified cleanup of {resource_id} ({resource_type})")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_id}: {e}")
            
            # Verify all resources are gone
            all_cleaned = True
            for resource_id, resource_type in self.context["cloudinary_resources_tracked"]:
                try:
                    result = cloudinary.api.resource(resource_id, resource_type=resource_type)
                    logger.error(f"Resource {resource_id} ({resource_type}) still exists!")
                    all_cleaned = False
                except cloudinary.api.NotFound:
                    logger.info(f"Verified cleanup of {resource_id} ({resource_type})")
            
            return all_cleaned
        except Exception as e:
            logger.error(f"Error verifying Cloudinary cleanup: {e}")
            return False

async def process_tweet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler for processing tweet URLs"""
    processor = None
    cleanup_task = None
    try:
        message = await update.message.reply_text("✨ Let's create something amazing! Starting the magic... 🎬")
        processor = TwitterVideoProcessor()
        
        # Set custom watermark if user has one
        if context.user_data.get("watermark"):
            processor.context["watermark"] = context.user_data["watermark"]
        
        # Extract and validate tweet URL
        text_parts = update.message.text.split()
        tweet_url = text_parts[1] if len(text_parts) > 1 else text_parts[0]
        
        if not ('x.com' in tweet_url or 'twitter.com' in tweet_url):
            await message.edit_text("❌ Please provide a valid X/Twitter URL.\nExample: https://x.com/username/status/123456789")
            return
        
        logger.info("Starting processing...")
        
        # Test 1: Download Tweet
        logger.info("Test 1: Testing tweet download...")
        success = await processor.download_tweet(tweet_url, message)
        if not success:
            raise Exception("Tweet download failed")
        assert processor.context.get("video_path"), "Video path not set"
        assert processor.context.get("tweet_text"), "Tweet text not captured"
        logger.info("✓ Tweet download test passed")
        
        # Test 2: Frame Extraction
        logger.info("Test 2: Testing frame extraction...")
        success = await processor.extract_frame(message)
        if not success:
            raise Exception("Frame extraction failed")
        assert processor.context.get("frame_path"), "Frame path not set"
        assert os.path.exists(processor.context["frame_path"]), "Frame file not created"
        logger.info("✓ Frame extraction test passed")
        
        # Test 3: Cloudinary Upload
        logger.info("Test 3: Testing Cloudinary upload...")
        success = await processor.upload_to_cloudinary(message)
        if not success:
            raise Exception("Cloudinary upload failed")
        assert processor.context.get("frame_url"), "Frame URL not set"
        # Track the resource for cleanup verification
        if processor.context.get("cloudinary_resources"):
            for resource_id in processor.context["cloudinary_resources"]:
                processor.track_cloudinary_resource(resource_id)
        logger.info("✓ Cloudinary upload test passed")
        
        # Test 4: Content Analysis
        logger.info("Test 4: Testing content analysis...")
        try:
            success = await asyncio.wait_for(
                processor.analyze_content(message),
                timeout=60  # 60 seconds timeout for content analysis
            )
            assert success, "Content analysis failed"
            assert processor.context.get("comment"), "Comment not generated"
            assert len(processor.context["comment"]) > 0, "Empty comment generated"
            logger.info("✓ Content analysis test passed")
        except asyncio.TimeoutError:
            logger.error("Content analysis timed out")
            raise Exception("Content analysis timed out after 60 seconds")
        
        # Test 5: Speech Generation
        logger.info("Test 5: Testing speech generation...")
        try:
            success = await asyncio.wait_for(
                processor.generate_speech(message),
                timeout=30  # 30 seconds timeout for speech generation
            )
            assert success, "Speech generation failed"
            assert processor.context.get("audio_path"), "Audio path not set"
            assert os.path.exists(processor.context["audio_path"]), "Audio file not created"
            logger.info("✓ Speech generation test passed")
        except asyncio.TimeoutError:
            logger.error("Speech generation timed out")
            raise Exception("Speech generation timed out after 30 seconds")
        
        # Test 6: Video Processing
        logger.info("Test 6: Testing video processing...")
        try:
            success = await asyncio.wait_for(
                processor.merge_audio_video(message),
                timeout=300  # 5 minutes timeout for video processing
            )
            assert success, "Video processing failed"
            final_video = processor.context.get("output_video_path")
            assert final_video and os.path.exists(final_video), "Final video not created"
            
            # Verify video format
            format_info = processor.verify_video_format(final_video)
            assert format_info, "Video format verification failed"
            logger.info(f"Video format verification: {format_info}")
            if not format_info["is_reel_ratio"]:
                logger.warning("Video aspect ratio is not 9:16")
            if not format_info["has_frame"]:
                logger.warning("Video does not have proper white frame")
            logger.info("✓ Video processing test passed")
            
            # Save and send the final video
            saved_video_path = processor.save_output_video()
            logger.info(f"Test video saved to: {saved_video_path}")
            
            # Check file size and compress if needed
            file_size = os.path.getsize(saved_video_path) / (1024 * 1024)  # Size in MB
            if file_size > 50:  # If larger than 50MB, compress
                await message.edit_text("📦 Video is large, optimizing for Telegram...")
                compressed_path = os.path.join(os.path.dirname(saved_video_path), "compressed_video.mp4")
                compress_stream = (
                    ffmpeg
                    .input(saved_video_path)
                    .output(compressed_path, 
                           **{'c:v': 'libx264', 
                              'crf': '28',  # Adjust quality (23-28 is good range)
                              'preset': 'faster',
                              'c:a': 'aac',
                              'b:a': '128k',
                              'movflags': '+faststart'})
                    .overwrite_output()
                )
                await processor.run_ffmpeg_async(compress_stream)
                saved_video_path = compressed_path
            
            # Upload animation
            upload_task = asyncio.create_task(processor.animate_loading(message, 'upload'))
            try:
                # Increase timeout for large files
                await asyncio.wait_for(
                    update.message.reply_video(
                        video=open(saved_video_path, 'rb'),
                        caption=f"✨ Your masterpiece is ready! 🎉\n\n"
                               f"💭 Generated comment:\n{processor.context['comment']}\n\n"
                               f"🎨 Created with love by your Video Assistant 🤖",
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60,
                        pool_timeout=120
                    ),
                    timeout=300  # 5 minutes total timeout
                )
                upload_task.cancel()
                await message.edit_text("✅ Processing completed successfully! Check out your video above ⬆️")
                logger.info("All processing completed successfully!")
                
                # Clean up immediately after video is sent
                logger.info("Starting immediate cleanup after video sent...")
                if processor:
                    await processor.cleanup_resources()
                # Clean up output directory if it exists
                if 'saved_video_path' in locals():
                    output_dir = os.path.dirname(saved_video_path)
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir, ignore_errors=True)
                        logger.info(f"Cleaned up output directory: {output_dir}")
                
            except asyncio.TimeoutError:
                upload_task.cancel()
                await message.edit_text("⚠️ Video upload timed out. The file might be too large for Telegram.")
                logger.error("Video upload timed out")
                raise
            except Exception as e:
                upload_task.cancel()
                processor.context["error"] = f"Failed to send video: {str(e)}"
                raise
                
        except asyncio.TimeoutError:
            logger.error("Video processing timed out")
            raise Exception("Video processing timed out after 5 minutes")
            
    except AssertionError as e:
        error_message = str(e)
        stage = processor.context.get("stage", "unknown")
        error_context = processor.context.get("error", str(e))
        
        error_messages = {
            "downloading": "❌ Failed to download the tweet. Please check if the URL is correct and the tweet is public.",
            "uploading_frame": "❌ Error processing video. Please try again with a different video.",
            "analyzing": "❌ Error analyzing the content. Please try again.",
            "generating_speech": "❌ Error generating narration. Please try again.",
            "merging": "❌ Error during video processing. The video might be in an unsupported format."
        }
        
        await message.edit_text(error_messages.get(stage, f"❌ An error occurred: {error_context}\nPlease try again."))
        logger.error(f"Processing error: {e}", exc_info=True)
    except Exception as e:
        error_message = str(e)
        stage = processor.context.get("stage", "unknown")
        error_context = processor.context.get("error", str(e))
        
        await message.edit_text(f"❌ An error occurred: {error_context}\nPlease try again.")
        logger.error(f"Processing error: {e}", exc_info=True)
    finally:
        try:
            if processor:
                await processor.cleanup_resources()
                # Double-check cleanup success
                cleanup_success = await processor.verify_cleanup()
                if not cleanup_success:
                    logger.warning("Some resources may not have been properly cleaned up")
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command"""
    welcome_message = (
        "👋 Welcome to the Video Processing Assistant! 🎬\n\n"
        "I can help you create amazing videos from tweets. Here's what I can do:\n"
        "✨ Add professional narration\n"
        "🎨 Optimize video format\n"
        "🎵 Balance audio perfectly\n"
        "🔍 Generate engaging comments\n\n"
        "Available commands:\n"
        "1️⃣ Send me a tweet URL directly\n"
        "2️⃣ /process <tweet_url> - Process a tweet\n"
        "3️⃣ /watermark <text> - Set your custom watermark\n"
        "   (default: 🎥 Created by @AutomatorByMani | Share & Enjoy!)\n\n"
        "Example:\n"
        "https://x.com/username/status/123456789"
    )
    await update.message.reply_text(welcome_message)

async def set_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /watermark command"""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide the watermark text.\n"
            "Usage: /watermark <your text>\n"
            "Example: /watermark 🎥 Created by @MyChannel | Share & Enjoy!"
        )
        return
    
    watermark_text = " ".join(context.args)
    if len(watermark_text) > 50:  # Increased limit to accommodate longer default text
        await update.message.reply_text(
            "❌ Watermark text is too long. Please keep it under 50 characters."
        )
        return
    
    # Store watermark in user data
    context.user_data["watermark"] = watermark_text
    await update.message.reply_text(
        f"✅ Watermark set to: {watermark_text}\n"
        "It will be used for your future video processing requests."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for all messages"""
    if not update.message or not update.message.text:
        await update.message.reply_text(
            "❌ Please send a valid tweet URL.\n"
            "Example: https://x.com/username/status/123456789"
        )
        return
        
    # Check if message contains a URL to X/Twitter
    text = update.message.text.lower()
    if "x.com" in text or "twitter.com" in text:
        await process_tweet(update, context)
    else:
        help_message = (
            "❌ I couldn't find a valid tweet URL in your message.\n\n"
            "Please send me a tweet URL from X/Twitter.\n"
            "Example: https://x.com/username/status/123456789\n\n"
            "Need help? Use /start to see all available commands!"
        )
        await update.message.reply_text(help_message)

def main():
    """Main function to run the bot"""
    try:
        # Initialize bot with proper error handling and retry logic
        application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("process", process_tweet))
        application.add_handler(CommandHandler("watermark", set_watermark))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Log successful initialization
        logger.info("Bot initialized successfully! Starting polling...")
        
        # Start the bot with error handling
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            timeout=60,
            read_timeout=60,
            write_timeout=60,
            pool_timeout=60,
            connect_timeout=60
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
