# Twitter Video Processing Bot

A Telegram bot that processes Twitter/X videos by adding professional narration, optimizing video format, and enhancing the overall presentation.

## Features

- Downloads videos from Twitter/X posts
- Adds AI-generated narration using Google Cloud Text-to-Speech
- Optimizes video format for social media (9:16 aspect ratio)
- Adds white frame padding for professional look
- Balances audio between original and narration tracks
- Generates engaging comments using AI
- Supports custom watermarks

## Requirements

- Python 3.8+
- FFmpeg
- Google Cloud Text-to-Speech API credentials
- OpenAI API key
- Cloudinary account
- Telegram Bot token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MrEmanAhmad/Twitter_Video_Processing_Bot.git
cd Twitter_Video_Processing_Bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
```

## Usage

1. Start a chat with your bot on Telegram
2. Send a Twitter/X video URL
3. Wait for the bot to process and return the enhanced video

### Commands

- `/start` - Display welcome message and instructions
- `/process <tweet_url>` - Process a specific tweet URL
- `/watermark <text>` - Set custom watermark for your videos

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Project Structure

```
project_root/
├── main.py              # Main bot implementation
├── test_processor.py    # Test suite
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
├── test_outputs/       # Generated test videos
└── SF-Pro-Display-Regular.ttf  # Font file for text overlay
```

## Environment Variables

Create a `.env` file with the following variables:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
LOG_LEVEL=INFO  # Logging level (DEBUG, INFO, WARNING, ERROR)
```

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- python-telegram-bot
- yt-dlp
- ffmpeg-python
- cloudinary
- google-cloud-texttospeech
- openai
- python-dotenv
- emoji

## Class Structure

### TwitterVideoProcessor

Main class handling video processing pipeline.

#### Instance Variables

```python
self.temp_dir                 # Temporary directory for processing
self.openai_client           # OpenAI API client
self.deepseek_client         # DeepSeek API client
self.context                 # Processing context dictionary
self.loading_animations      # Animation frames dictionary
```

#### Context Dictionary Structure

```python
context = {
    "temp_dir": str,              # Temporary directory path
    "cloudinary_resources": list, # List of Cloudinary resource IDs
    "error": str,                # Error message if any
    "stage": str,               # Current processing stage
    "watermark": str,          # Watermark text
    "cloudinary_resources_tracked": set,  # Set of tracked resources
    "video_path": str,        # Downloaded video path
    "frame_path": str,       # Extracted frame path
    "frame_url": str,       # Cloudinary frame URL
    "tweet_text": str,     # Original tweet text
    "comment": str,       # Generated comment
    "audio_path": str,   # Generated audio path
    "output_video_path": str  # Final video path
}
```

## Key Methods

### Video Processing Pipeline

1. `download_tweet(url: str, message_obj) -> bool`
   - Downloads tweet video using yt-dlp
   - Returns success status

2. `extract_frame(message_obj) -> bool`
   - Extracts representative frame using FFmpeg
   - Frame saved as "frame.jpg"

3. `upload_to_cloudinary(message_obj) -> bool`
   - Uploads frame to Cloudinary
   - Tracks resource for cleanup

4. `analyze_content(message_obj) -> bool`
   - Uses OpenAI Vision API for frame analysis
   - Uses DeepSeek API for comment generation
   - 60-second timeout

5. `generate_speech(message_obj) -> bool`
   - Converts comment to speech using Google Cloud TTS
   - 30-second timeout
   - Output: "narration.mp3"

6. `merge_audio_video(message_obj) -> bool`
   - Adds padding and white frame
   - Mixes audio streams (video: 30%, narration: 100%)
   - 5-minute timeout

### Helper Methods

1. `clean_text_for_narration(text: str) -> str`
   - Removes hashtags, emojis, numbers
   - Cleans punctuation and spacing

2. `verify_video_format(video_path: str) -> dict`
   - Checks video dimensions and ratio
   - Verifies frame presence

3. `track_cloudinary_resource(resource_id: str, resource_type: str = "image")`
   - Tracks Cloudinary resources for cleanup

4. `verify_cloudinary_cleanup() -> bool`
   - Verifies all resources are cleaned up

5. `save_output_video() -> str`
   - Saves to test_outputs/YYYYMMDD_HHMMSS/final_video.mp4

## Telegram Bot Handlers

1. `process_tweet(update: Update, context: ContextTypes.DEFAULT_TYPE)`
   - Main processing handler
   - Handles tweet URLs

2. `start(update: Update, context: ContextTypes.DEFAULT_TYPE)`
   - Welcome message and commands

3. `set_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE)`
   - Sets custom watermark text

4. `handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE)`
   - General message handler

## Testing

Run tests using:
```bash
python test_processor.py
```

Test video URL:
```
https://x.com/AMAZlNGNATURE/status/1887223066909937941
```

## FFmpeg Settings

### Video Padding
```python
ffmpeg_settings = {
    'acodec': 'copy',
    'vcodec': 'copy',
    'movflags': '+faststart',
    'preset': 'ultrafast',
    'threads': 'auto'
}
```

### Audio Mixing
```python
audio_settings = {
    'video_volume': 0.3,     # 30% original audio
    'narration_volume': 1.0,  # 100% narration
    'dropout_transition': 0.5
}
```

## Error Handling

Error stages:
- downloading
- uploading_frame
- analyzing
- generating_speech
- merging

Each stage has specific error messages and cleanup procedures.

## Resource Cleanup

1. Cloudinary resources
   - Images and videos tracked separately
   - Verified cleanup with 5-second delay

2. Temporary files
   - All files in temp_dir removed
   - Directory removed after processing

## Output Directory Structure

```
test_outputs/
└── YYYYMMDD_HHMMSS/
    └── final_video.mp4
```

## Timeouts

- Content Analysis: 60 seconds
- Speech Generation: 30 seconds
- Video Processing: 300 seconds (5 minutes)
- Cloudinary Upload: 60 seconds

## Animation Frames

Loading animations for each stage:
- download
- analyze
- speech
- merge
- merge_audio
- final_touch
- upload

Each animation has 4 frames updating every 0.5 seconds. 