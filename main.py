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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize API clients and configurations
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Google Cloud TTS client with explicit credentials
try:
    # Try using environment variable directly first
    creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if creds_json:
        try:
            # Debug log the first few characters of the credentials
            logger.info(f"Credentials string starts with: {creds_json[:50]}...")
            
            # Remove any potential wrapping quotes if present
            creds_json = creds_json.strip()
            if creds_json.startswith('"') and creds_json.endswith('"'):
                creds_json = creds_json[1:-1]
            
            # Clean up the JSON string
            creds_json = creds_json.replace('\\n', '\n')
            creds_json = creds_json.replace('\\"', '"')
            creds_json = creds_json.replace('\\\\', '\\')
            
            # Debug log the processed string
            logger.info("Attempting to parse JSON credentials...")
            
            try:
                creds_dict = json.loads(creds_json)
                logger.info("JSON parsing successful")
                
                # Verify required fields
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in creds_dict]
                if missing_fields:
                    raise ValueError(f"Missing required fields in credentials: {missing_fields}")
                
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
                logger.info("Successfully initialized Google Cloud TTS client with credentials from environment")
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error at position {je.pos}: {je.msg}")
                logger.error(f"Problematic JSON section: {creds_json[max(0, je.pos-20):min(len(creds_json), je.pos+20)]}")
                raise
            
        except json.JSONDecodeError as je:
            logger.error(f"Error parsing credentials JSON: {str(je)}")
            logger.error("Please ensure the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable contains valid JSON")
            raise
        except Exception as e:
            logger.error(f"Error creating credentials from JSON: {str(e)}")
            raise
    else:
        # Fallback to file-based credentials
        credentials_path = "/app/credentials/google_credentials.json"
        if os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Successfully initialized Google Cloud TTS client with credentials file")
        else:
            raise Exception("No Google Cloud credentials found in environment or file system")
except Exception as e:
    logger.error(f"Error initializing Google Cloud TTS client: {str(e)}")
    raise

class TwitterVideoProcessor:
    def __init__(self):
        self.temp_dir = os.path.join('/tmp', f'twitter_processor_{int(time.time())}')
        os.makedirs(self.temp_dir, exist_ok=True)
        # Initialize OpenAI client with explicit configuration
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0,
            max_retries=3
        )
        # Initialize DeepSeek client with explicit configuration
        self.deepseek_client = openai.OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1",
            timeout=30.0,
            max_retries=3
        )
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
        start_time = time.time()
        animation_frames = self.loading_animations[animation_key]
        frame_index = 0
        
        while time.time() - start_time < duration:
            try:
                await message_obj.edit_text(animation_frames[frame_index])
                frame_index = (frame_index + 1) % len(animation_frames)
                await asyncio.sleep(0.5)  # Update animation every 0.5 seconds
            except Exception as e:
                logger.error(f"Animation error: {e}")
                break

    async def download_tweet(self, url: str, message_obj) -> bool:
        """Download tweet video and metadata using yt-dlp"""
        self.context["tweet_url"] = url
        self.context["stage"] = "downloading"
        animation_task = asyncio.create_task(self.animate_loading(message_obj, 'download'))
        
        try:
            await message_obj.edit_text("🎬 Starting video processing...")
            
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(self.temp_dir, '%(id)s.%(ext)s'),
                'extract_flat': True,
                'extractor_args': {'twitter': {'api': ['graphql']}}
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    url = url.replace('x.com', 'twitter.com')
                    info = ydl.extract_info(url, download=False)
                    
                    ydl_opts.update({
                        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                        'extract_flat': False,
                        'writeinfojson': True,
                        'writedescription': True
                    })
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                        info = ydl_download.extract_info(url, download=True)
                        video_path = ydl_download.prepare_filename(info)
                        video_path = os.path.splitext(video_path)[0] + ".mp4"
                        info_filename = os.path.splitext(video_path)[0] + ".info.json"
                        
                        # Update context with downloaded data
                        self.context.update({
                            "video_path": video_path,
                            "video_info": {
                                "duration": info.get('duration', 0),
                                "title": info.get('title', ''),
                                "uploader": info.get('uploader', '')
                            }
                        })
                        
                        # Extract and store tweet text
                        tweet_text = info.get('description', '')
                        if not tweet_text and os.path.exists(info_filename):
                            try:
                                with open(info_filename, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    tweet_text = metadata.get("description", "")
                            except json.JSONDecodeError:
                                logger.error("Failed to parse tweet metadata")
                                self.context["error"] = "Failed to parse tweet metadata"
                                tweet_text = ""
                        
                        self.context["tweet_text"] = tweet_text
                        animation_task.cancel()
                        return True
                        
            except Exception as e:
                self.context["error"] = f"Error downloading tweet: {str(e)}"
                logger.error(f"Error downloading tweet: {e}")
                raise
                
        except Exception as e:
            self.context["error"] = f"Download failed: {str(e)}"
            animation_task.cancel()
            raise e

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
            
            # Use context data for analysis
            tweet_text = self.context.get("tweet_text", "")
            frame_url = self.context.get("frame_url")
            
            if not frame_url:
                logger.error("No frame URL available for analysis")
                self.context["error"] = "No frame URL available for analysis"
                return False
            
            try:
                # Test Vision API first
                logger.info("Testing OpenAI Vision API...")
                try:
                    vision_response = self.openai_client.chat.completions.create(
                        model="gpt-4-turbo-2024-04-09",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a social media content analyzer. Analyze the video frame and tweet text to understand the content and generate an engaging, empathetic comment that captures the essence of the content."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Tweet text: {tweet_text}\n\nAnalyze this frame in the context of the tweet."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": frame_url}
                                    }
                                ]
                            }
                        ],
                        max_tokens=300
                    )
                    
                    frame_analysis = vision_response.choices[0].message.content
                    if not frame_analysis:
                        raise Exception("No analysis generated from Vision API")
                    
                    logger.info("Vision API Response received successfully")
                    self.context["frame_analysis"] = frame_analysis
                    
                except Exception as vision_error:
                    logger.error(f"Vision API error: {type(vision_error).__name__} - {str(vision_error)}")
                    raise
                
                # Test DeepSeek API
                logger.info("Testing DeepSeek API...")
                try:
                    comment_response = self.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {
                                "role": "system",
                                "content": """Generate a short, engaging social media comment that sounds natural when narrated.
                                Key rules:
                                1. Write in a conversational, flowing style
                                2. No quotes or special characters
                                3. Keep it under 15 seconds (30-35 words)
                                4. Focus on smooth transitions and natural speech"""
                            },
                            {
                                "role": "user",
                                "content": f"Based on this analysis:\n\n{frame_analysis}\n\nTweet text: {tweet_text}\n\nGenerate a comment that flows naturally when spoken aloud."
                            }
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    if not comment_response or not hasattr(comment_response, 'choices') or not comment_response.choices:
                        raise Exception("Invalid response from DeepSeek API")
                    
                    generated_comment = comment_response.choices[0].message.content.strip()
                    if not generated_comment:
                        raise Exception("No comment was generated")
                    
                    logger.info("DeepSeek API Response received successfully")
                    logger.info(f"Generated comment: {generated_comment}")
                    
                    # Store the comment in context
                    self.context["comment"] = generated_comment
                    return True
                    
                except Exception as deepseek_error:
                    logger.error(f"DeepSeek API error: {type(deepseek_error).__name__} - {str(deepseek_error)}")
                    raise
                
            except Exception as api_error:
                logger.error(f"API error: {type(api_error).__name__} - {str(api_error)}")
                self.context["error"] = f"API error: {str(api_error)}"
                raise
            
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__} - {str(e)}")
            self.context["error"] = f"Analysis failed: {str(e)}"
            raise e
        finally:
            animation_task.cancel()

    def clean_text_for_narration(self, text: str) -> str:
        """Clean text by removing hashtags, emojis, and numbers for better narration"""
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
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

    async def cleanup_resources(self):
        """Clean up all resources including Cloudinary"""
        try:
            # Clean up Cloudinary resources and verify
            cleanup_success = await self.verify_cloudinary_cleanup()
            if not cleanup_success:
                logger.warning("Some Cloudinary resources may not have been cleaned up properly")
            
            # Add a delay to ensure Cloudinary operations complete
            await asyncio.sleep(5)
            
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            raise

    async def add_padding_and_frame(self, video_path: str, message_obj) -> str:
        """Add padding to make video reel-sized (9:16) and add white frame using Cloudinary"""
        await message_obj.edit_text("🎨 Formatting video for reels...")
        
        uploaded_resource_id = None
        try:
            # Get original video dimensions and verify
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Store video info like in test processor
            self.context["video_info"] = {
                "original_width": width,
                "original_height": height,
                "duration": float(video_info.get('duration', 0))
            }
            
            logger.info(f"Original video dimensions: {width}x{height}")
            
            # Calculate target dimensions exactly like test processor
            if height > width:
                target_width = width + 40
                target_height = int((target_width * 16) / 9)
            else:
                target_height = height + 40
                target_width = int((target_height * 9) / 16)
            
            logger.info(f"Target dimensions: {target_width}x{target_height}")
            
            # Upload with test processor's exact settings
            upload_result = cloudinary.uploader.upload(
                video_path,
                resource_type="video",
                transformation=[
                    {
                        'width': target_width,
                        'height': target_height,
                        'crop': 'pad',
                        'background': 'white'
                    }
                ],
                eager_async=False,
                timeout=60
            )
            
            uploaded_resource_id = upload_result['public_id']
            self.context["cloudinary_resources"].append(uploaded_resource_id)
            padded_video = os.path.join(self.temp_dir, "padded_video.mp4")
            
            # Use test processor's optimized FFmpeg settings
            stream = (
                ffmpeg
                .input(upload_result['secure_url'])
                .output(padded_video, 
                       acodec='copy', 
                       vcodec='copy',
                       movflags='+faststart',
                       preset='ultrafast',
                       threads='auto'
                )
                .overwrite_output()
            )
            
            # Run with test processor's timeout
            await asyncio.wait_for(
                self.run_ffmpeg_async(stream),
                timeout=300
            )
            
            # Verify with test processor's strict checks
            format_info = self.verify_video_format(padded_video)
            if format_info:
                if not format_info["is_reel_ratio"]:
                    logger.warning("Padded video does not have correct reel ratio")
                    if abs(format_info["aspect_ratio"] - 16/9) > 0.1:
                        logger.info("Attempting to fix aspect ratio...")
                        fixed_video = os.path.join(self.temp_dir, "fixed_video.mp4")
                        fix_stream = (
                            ffmpeg
                            .input(padded_video)
                            .filter('scale', width='-1', height='1351')
                            .output(fixed_video, 
                                   acodec='copy',
                                   preset='ultrafast',
                                   threads='auto'
                            )
                            .overwrite_output()
                        )
                        await self.run_ffmpeg_async(fix_stream)
                        padded_video = fixed_video
            
            # Clean up immediately like test processor
            try:
                cloudinary.uploader.destroy(uploaded_resource_id, resource_type="video")
                logger.info(f"Cleaned up Cloudinary video resource: {uploaded_resource_id}")
                self.context["cloudinary_resources"].remove(uploaded_resource_id)
            except Exception as e:
                logger.error(f"Error cleaning up Cloudinary video resource {uploaded_resource_id}: {e}")
            
            return padded_video
                
        except Exception as e:
            if uploaded_resource_id:
                try:
                    cloudinary.uploader.destroy(uploaded_resource_id, resource_type="video")
                    logger.info(f"Cleaned up Cloudinary video resource on error: {uploaded_resource_id}")
                    if uploaded_resource_id in self.context["cloudinary_resources"]:
                        self.context["cloudinary_resources"].remove(uploaded_resource_id)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up Cloudinary video resource {uploaded_resource_id}: {cleanup_error}")
            
            logger.error(f"Error adding padding: {e}")
            raise

    async def show_progress_animation(self, message_obj, animation_key: str, duration: float = None):
        """Show progress animation with optional duration"""
        animation_task = asyncio.create_task(self.animate_loading(message_obj, animation_key))
        if duration:
            await asyncio.sleep(duration)
            animation_task.cancel()
        return animation_task

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
                    
                    # Get video duration
                    video_probe = ffmpeg.probe(padded_video)
                    video_info = next(s for s in video_probe['streams'] if s['codec_type'] == 'video')
                    video_duration = float(video_info.get('duration', 0))
                    
                    # Get audio duration
                    audio_probe = ffmpeg.probe(self.context["audio_path"])
                    audio_info = next(s for s in audio_probe['streams'] if s['codec_type'] == 'audio')
                    audio_duration = float(audio_info.get('duration', 0))
                    
                    logger.info(f"Video duration: {video_duration}s, Audio duration: {audio_duration}s")
                    
                    # Mix audio streams with proper settings
                    video_stream = ffmpeg.input(padded_video)
                    video_audio = video_stream.audio.filter('volume', 0.3)  # 30% volume
                    narration_audio = ffmpeg.input(self.context["audio_path"]).filter('volume', 1.0)  # 100% volume
                    
                    mixed_audio = ffmpeg.filter(
                        [video_audio, narration_audio],
                        'amix',
                        inputs=2,
                        duration='first',
                        dropout_transition=0.5
                    )
                    
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
            logger.info(f"Cleaned text for narration: {narration_text}")
            
            if not narration_text:
                logger.error("No text available for narration")
                self.context["error"] = "No text available for narration"
                return False
            
            synthesis_input = texttospeech.SynthesisInput(text=narration_text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",  # Using a female voice
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=2.0
            )
            
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            audio_path = os.path.join(self.temp_dir, "narration.mp3")
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
                    logger.info(f"Resource already cleaned up: {resource_id} ({resource_type})")
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
            
            # Upload animation
            upload_task = asyncio.create_task(processor.animate_loading(message, 'upload'))
            try:
                await update.message.reply_video(
                    video=open(saved_video_path, 'rb'),
                    caption=f"✨ Your masterpiece is ready! 🎉\n\n"
                           f"💭 Generated comment:\n{processor.context['comment']}\n\n"
                           f"🎨 Created with love by your Video Assistant 🤖"
                )
                upload_task.cancel()
                await message.edit_text("✅ Processing completed successfully! Check out your video above ⬆️")
                logger.info("All processing completed successfully!")
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
        # Ensure cleanup happens even if processing fails
        if processor:
            await processor.cleanup_resources()
            # Wait for any async cleanup operations
            await asyncio.sleep(5)
            # Additional cleanup attempt
            await processor.verify_cloudinary_cleanup()
            if processor.temp_dir and os.path.exists(processor.temp_dir):
                shutil.rmtree(processor.temp_dir)
                logger.info(f"Cleaned up temporary directory: {processor.temp_dir}")

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
