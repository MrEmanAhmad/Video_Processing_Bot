"""
Telegram bot for video processing with loading animations
"""

import os
import asyncio
import logging
import multiprocessing
from queue import Queue
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor
import psutil
import logging.handlers
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from dotenv import load_dotenv

import pipeline.Step_4_generate_commentary as commentary
from pipeline import (
    Step_1_download_video,
    Step_2_extract_frames,
    Step_3_analyze_frames,
    Step_4_generate_commentary,
    Step_5_generate_audio,
    Step_6_video_generation,
    Step_7_cleanup
)

CommentaryStyle = commentary.CommentaryStyle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def setup_google_credentials():
    """Setup Google credentials from environment variables."""
    try:
        # Get credentials from environment variable
        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_json:
            logger.error("Google credentials not found in environment variables")
            return False
            
        # Create credentials directory if it doesn't exist
        creds_dir = Path("credentials")
        creds_dir.mkdir(exist_ok=True)
        
        # Parse and format the JSON properly
        try:
            creds_data = json.loads(creds_json)
            if "type" not in creds_data:
                creds_data["type"] = "service_account"
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in credentials")
            return False
            
        # Save credentials to JSON file
        creds_file = creds_dir / "google_credentials.json"
        with open(creds_file, "w") as f:
            json.dump(creds_data, f)
            
        # Set environment variable to point to the credentials file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_file)
        logger.info("Google credentials setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up Google credentials: {e}")
        return False

# Global settings
MAX_CONCURRENT_PROCESSES = 4
MAX_QUEUE_SIZE = 20
RESOURCE_LIMIT_PER_USER = 2
STATUS_UPDATE_INTERVAL = 5
MAX_MEMORY_USAGE = 0.75
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50MB limit
CLEANUP_INTERVAL = 300  # 5 minutes

# Processing queue and resource tracking
processing_queue = Queue(maxsize=MAX_QUEUE_SIZE)
active_processes: Dict[int, int] = {}
queue_lock = threading.Lock()

def set_worker_priority():
    """Set worker thread priority to below normal."""
    try:
        import psutil
        current_process = psutil.Process()
        if os.name == 'nt':  # Windows
            current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:  # Linux/Unix
            current_process.nice(10)  # Nice value between -20 and 19, higher means lower priority
    except Exception as e:
        logger.warning(f"Could not set worker priority: {e}")

worker_pool = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_PROCESSES,
    thread_name_prefix="worker",
    initializer=set_worker_priority
)

# Configure logging with file rotation
log_handler = logging.handlers.RotatingFileHandler(
    'bot.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(log_handler)

class ResourceManager:
    """Manages system resources and cleanup during processing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.temp_files = set()
        self.temp_dirs = set()
        self.cloudinary_resources = set()
        
    def register_temp_file(self, file_path: Path):
        """Register a temporary file for cleanup."""
        self.temp_files.add(file_path)
        
    def register_temp_dir(self, dir_path: Path):
        """Register a temporary directory for cleanup."""
        self.temp_dirs.add(dir_path)
        
    def register_cloudinary_resource(self, public_id: str):
        """Register a Cloudinary resource for cleanup."""
        self.cloudinary_resources.add(public_id)
        
    def cleanup_temp_file(self, file_path: Path):
        """Clean up a temporary file if it exists."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up temp file: {file_path}")
                self.temp_files.discard(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def cleanup_temp_dir(self, dir_path: Path):
        """Clean up a temporary directory if it exists."""
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up temp directory: {dir_path}")
                self.temp_dirs.discard(dir_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {dir_path}: {e}")
            
    def cleanup_cloudinary_resources(self):
        """Clean up all registered Cloudinary resources."""
        for public_id in self.cloudinary_resources:
            try:
                cloudinary.uploader.destroy(public_id)
                logger.info(f"Cleaned up Cloudinary resource: {public_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup Cloudinary resource {public_id}: {e}")
        self.cloudinary_resources.clear()
    
    def cleanup_all(self):
        """Clean up all temporary files, directories and cloud resources."""
        # First cleanup Cloudinary resources
        self.cleanup_cloudinary_resources()
        
        # Then cleanup local files
        for file_path in list(self.temp_files):
            self.cleanup_temp_file(file_path)
            
        # Finally cleanup directories
        for dir_path in list(self.temp_dirs):
            self.cleanup_temp_dir(dir_path)
            
        # Log cleanup completion
        logger.info(f"Completed cleanup for output directory: {self.output_dir}")

async def cleanup_old_processes():
    """Periodically clean up completed or stale processes and manage resources."""
    process = psutil.Process()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Clean up old processes
            with queue_lock:
                for user_id in list(active_processes.keys()):
                    if active_processes[user_id] <= 0:
                        del active_processes[user_id]
            
            # Check memory usage
            memory_percent = process.memory_percent()
            if memory_percent > MAX_MEMORY_USAGE:
                logger.warning(f"High memory usage detected: {memory_percent:.1f}%. Cleaning up...")
                # Clean up temporary files
                for temp_dir in Path().glob("analysis_*"):
                    if temp_dir.is_dir():
                        age = (current_time - datetime.fromtimestamp(temp_dir.stat().st_mtime)).total_seconds()
                        # Clean up files older than 1 hour
                        if age > 3600:
                            try:
                                shutil.rmtree(temp_dir)
                                logger.info(f"Cleaned up old directory: {temp_dir}")
                            except Exception as e:
                                logger.error(f"Error cleaning up {temp_dir}: {e}")
            
            await asyncio.sleep(CLEANUP_INTERVAL)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup_old_processes: {e}")
            await asyncio.sleep(60)

# Loading animations
LOADING_ANIMATIONS = {
    'download': ["📥", "⬇️", "📥", "⬇️"],
    'process': ["⚙️", "🔄", "⚙️", "🔄"],
    'analyze': ["🔍", "🔎", "🔍", "🔎"],
    'generate': ["✨", "💫", "✨", "💫"],
    'complete': ["✅", "🎉", "✅", "🎉"]
}

class ProcessingStatus:
    """Tracks the processing status and handles loading animations."""
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 7
        self.animation_frame = 0
        self.processing = False
        self.current_message = None
    
    def get_progress_bar(self) -> str:
        """Generate a progress bar based on current step."""
        filled = "█" * self.current_step
        empty = "░" * (self.total_steps - self.current_step)
        percentage = (self.current_step / self.total_steps) * 100
        return f"{filled}{empty} {percentage:.0f}%"
    
    def get_loading_animation(self, animation_type: str) -> str:
        """Get the current frame of the loading animation."""
        frames = LOADING_ANIMATIONS.get(animation_type, LOADING_ANIMATIONS['process'])
        frame = frames[self.animation_frame % len(frames)]
        self.animation_frame += 1
        return frame

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    keyboard = [
        [
            InlineKeyboardButton("Documentary 🎥", callback_data="style_documentary"),
            InlineKeyboardButton("Energetic 🔥", callback_data="style_energetic")
        ],
        [
            InlineKeyboardButton("Analytical 🔬", callback_data="style_analytical"),
            InlineKeyboardButton("Storyteller 📖", callback_data="style_storyteller")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "👋 Welcome to the Video Commentary Bot!\n\n"
        "Send me a video or URL, and I'll create an engaging commentary for it.\n"
        "Choose your preferred commentary style:",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "🎥 *Video Commentary Bot Help*\n\n"
        "*Commands:*\n"
        "/start - Start the bot and choose commentary style\n"
        "/help - Show this help message\n"
        "/style - Change commentary style\n\n"
        "*How to use:*\n"
        "1. Choose your preferred commentary style\n"
        "2. Send a video file or URL\n"
        "3. Wait while I process and generate commentary\n"
        "4. Receive your video with professional narration!\n\n"
        "*Supported formats:*\n"
        "- Video files (MP4, MOV, AVI)\n"
        "- URLs (YouTube, Vimeo, etc.)\n\n"
        "*Commentary Styles:*\n"
        "🎥 Documentary - Professional and informative\n"
        "🔥 Energetic - Dynamic and enthusiastic\n"
        "🔬 Analytical - Detailed and technical\n"
        "📖 Storyteller - Narrative and emotional",
        parse_mode='Markdown'
    )

async def update_status_message(
    message,
    status: ProcessingStatus,
    step_name: str,
    animation_type: str
) -> None:
    """Update the status message with current progress."""
    text = (
        f"{status.get_loading_animation(animation_type)} *Processing Video*\n\n"
        f"Current Step: {step_name}\n"
        f"Progress: {status.get_progress_bar()}\n\n"
        f"Please wait while I process your video..."
    )
    
    try:
        if status.current_message:
            await status.current_message.edit_text(text, parse_mode='Markdown')
        else:
            status.current_message = await message.reply_text(text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error updating status message: {e}")

async def process_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_path: Optional[str] = None,
    url: Optional[str] = None
) -> None:
    """Process the video and generate commentary."""
    
    user_id = update.effective_user.id
    
    # Get video description/title from the message
    video_text = ""
    if update.message.caption:
        video_text = update.message.caption
    elif update.message.text and url:
        video_text = update.message.text
    
    # If no description provided, ask user
    if not video_text:
        await update.message.reply_text(
            "Please provide a brief description of the video to help me generate better commentary.\n"
            "You can either:\n"
            "1. Add a caption when sending the video\n"
            "2. Reply to this message with a description"
        )
        return
    
    # Check user resource limits
    with queue_lock:
        if user_id in active_processes and active_processes[user_id] >= RESOURCE_LIMIT_PER_USER:
            await update.message.reply_text(
                "⚠️ You have too many videos being processed. Please wait for them to complete."
            )
            return
        
        # Check queue size
        if processing_queue.full():
            await update.message.reply_text(
                "⚠️ The processing queue is full. Please try again later."
            )
            return
        
        # Increment active processes for user
        active_processes[user_id] = active_processes.get(user_id, 0) + 1
    
    # Get the selected style from user data
    style_name = context.user_data.get('style', 'documentary')
    style_map = {
        'documentary': CommentaryStyle.DOCUMENTARY,
        'energetic': CommentaryStyle.ENERGETIC,
        'analytical': CommentaryStyle.ANALYTICAL,
        'storyteller': CommentaryStyle.STORYTELLER
    }
    
    # Initialize processing status and resource manager
    status = ProcessingStatus()
    status.processing = True
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"analysis_{timestamp}")
        resource_manager = ResourceManager(output_dir)
        
        # Register the output directory for cleanup
        resource_manager.register_temp_dir(output_dir)
        
        # Step 1: Download video
        status.current_step = 1
        await update_status_message(update.message, status, "Downloading Video", "download")
        
        loop = asyncio.get_event_loop()
        success, metadata, video_title = await loop.run_in_executor(
            worker_pool,
            lambda: Step_1_download_video.execute_step(url if url else video_path, output_dir)
        )
        
        if not success:
            await update.message.reply_text("❌ Failed to process video. Please try again.")
            resource_manager.cleanup_all()
            return
            
        video_file = next(Path(output_dir / "video").glob("*.mp4"))
        resource_manager.register_temp_file(video_file)
        
        # Add video text to metadata
        metadata['description'] = video_text
        metadata['title'] = video_title or video_text.split('\n')[0]
        
        # Step 2: Extract frames
        status.current_step = 2
        await update_status_message(update.message, status, "Extracting Frames", "process")
        frames_dir = output_dir / "frames"
        resource_manager.register_temp_dir(frames_dir)
        
        key_frames, scene_changes, motion_scores = await loop.run_in_executor(
            worker_pool,
            lambda: Step_2_extract_frames.execute_step(
                video_file=video_file,
                output_dir=output_dir
            )
        )
        
        # Step 3: Analyze frames
        status.current_step = 3
        await update_status_message(update.message, status, "Analyzing Content", "analyze")
        
        analysis_results = await loop.run_in_executor(
            worker_pool,
            lambda: Step_3_analyze_frames.execute_step(
                frames_dir=frames_dir,
                output_dir=output_dir,
                metadata=metadata,
                scene_changes=scene_changes,
                motion_scores=motion_scores,
                video_duration=metadata.get('duration', 0)
            )
        )
        
        # Cleanup extracted frames after analysis
        resource_manager.cleanup_temp_dir(frames_dir)
        
        # Step 4: Generate commentary
        status.current_step = 4
        await update_status_message(update.message, status, "Generating Commentary", "generate")
        
        commentary, audio_script = await loop.run_in_executor(
            worker_pool,
            lambda: Step_4_generate_commentary.execute_step(
                analysis_file=output_dir / "final_analysis.json",
                output_dir=output_dir,
                style=style_map[style_name]
            )
        )
        
        # Log the generated commentary
        logger.info(f"Generated Commentary for user {user_id}:\n{commentary}")
        
        # Step 5: Generate audio
        status.current_step = 5
        await update_status_message(update.message, status, "Generating Audio", "generate")
        if audio_script:
            audio_file = await loop.run_in_executor(
                worker_pool,
                lambda: Step_5_generate_audio.execute_step(
                    audio_script=audio_script,
                    output_dir=output_dir,
                    style_name=style_name
                )
            )
            
            # Step 6: Generate final video
            status.current_step = 6
            await update_status_message(update.message, status, "Creating Final Video", "generate")
            if audio_file:
                resource_manager.register_temp_file(audio_file)
                
                final_video = await loop.run_in_executor(
                    worker_pool,
                    lambda: Step_6_video_generation.execute_step(
                        video_file=video_file,
                        audio_file=audio_file,
                        output_dir=output_dir,
                        style_name=style_name
                    )
                )
                
                if final_video:
                    # Step 7: Cleanup
                    status.current_step = 7
                    await update_status_message(update.message, status, "Finishing Up", "complete")
                    
                    try:
                        # Send the final video
                        await update.message.reply_video(
                            video=open(final_video, 'rb'),
                            caption=f"✨ Here's your video with {style_name} style commentary!\n\nBased on: {video_text[:100]}...",
                            supports_streaming=True
                        )
                    finally:
                        # Ensure cleanup happens after video is sent
                        resource_manager.cleanup_all()
                        return
                    
        await update.message.reply_text("❌ Failed to generate video. Please try again.")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        await update.message.reply_text(
            "❌ An error occurred while processing your video. Please try again later."
        )
    finally:
        status.processing = False
        # Cleanup resources and decrement active processes
        resource_manager.cleanup_all()
        with queue_lock:
            if user_id in active_processes:
                active_processes[user_id] = max(0, active_processes[user_id] - 1)

async def handle_style_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the style selection callback."""
    query = update.callback_query
    await query.answer()
    
    # Extract style from callback data
    style = query.data.replace('style_', '')
    context.user_data['style'] = style
    
    await query.edit_message_text(
        f"🎨 Selected style: *{style.title()}*\n\n"
        "Now send me a video file or URL to process!",
        parse_mode='Markdown'
    )

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video messages."""
    if not context.user_data.get('style'):
        await update.message.reply_text(
            "Please select a commentary style first using /start or /style!"
        )
        return
    
    video = update.message.video
    if video.file_size > MAX_VIDEO_SIZE:  # 50MB limit
        await update.message.reply_text(
            "❌ Video file is too large. Please send a video under 50MB or use a URL instead."
        )
        return
    
    # Download the video file
    video_file = await context.bot.get_file(video.file_id)
    video_path = f"temp_{video.file_unique_id}.mp4"
    await video_file.download_to_drive(video_path)
    
    # Process the video
    await process_video(update, context, video_path=video_path)
    
    # Clean up
    try:
        os.remove(video_path)
    except:
        pass

async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle URL messages."""
    if not context.user_data.get('style'):
        await update.message.reply_text(
            "Please select a commentary style first using /start or /style!"
        )
        return
    
    url = update.message.text
    await process_video(update, context, url=url)

def main() -> None:
    """Start the bot."""
    # Create the Application with optimized settings
    application = (
        Application.builder()
        .token(os.environ["TELEGRAM_BOT_TOKEN"])
        .concurrent_updates(True)  # Enable concurrent updates
        .http_version('2')  # Use HTTP/2
        .get_updates_http_version('2')
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # Add handlers with rate limiting
    application.add_handler(
        CommandHandler(
            "start",
            start,
            block=False
        )
    )
    application.add_handler(
        MessageHandler(
            filters.VIDEO | filters.Document.VIDEO,
            handle_video,
            block=False
        )
    )
    application.add_handler(
        MessageHandler(
            filters.Entity("url"),
            handle_url,
            block=False
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            handle_style_selection,
            block=False
        )
    )

    # Add cleanup job to the application
    async def start_cleanup(app: Application):
        app.create_task(cleanup_old_processes())

    application.post_init = start_cleanup

    # Start the Bot with optimized settings
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        pool_timeout=30,
        read_timeout=30,
        write_timeout=30,
        connect_timeout=30
    )

if __name__ == '__main__':
    try:
        # Setup Google credentials
        if not setup_google_credentials():
            logger.error("Failed to setup Google credentials. Exiting...")
            exit(1)
        
        # Set process priority (cross-platform)
        try:
            import psutil
            current_process = psutil.Process()
            if os.name == 'nt':  # Windows
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:  # Linux/Unix
                current_process.nice(10)
        except Exception as e:
            logger.warning(f"Could not set process priority: {e}")
        
        # Set up process pool
        multiprocessing.freeze_support()
        
        # Start the bot
        main()
    finally:
        # Cleanup on shutdown
        worker_pool.shutdown(wait=True)
        
        # Clean up credentials file
        try:
            creds_file = Path("credentials/google_credentials.json")
            if creds_file.exists():
                creds_file.unlink()
            creds_dir = Path("credentials")
            if creds_dir.exists():
                creds_dir.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up credentials: {e}")
        
        # Clean up any remaining temporary files
        for temp_dir in Path().glob("analysis_*"):
            try:
                if temp_dir.is_dir():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up {temp_dir}: {e}")
