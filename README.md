# Twitter Video Processing Bot 🎬

A Telegram bot that enhances Twitter/X videos with AI-powered narration, professional formatting, and engaging comments.

## Features ✨

- 🎥 Downloads videos from Twitter/X posts
- 🗣️ Adds AI-generated narration using Google Cloud Text-to-Speech
- 📱 Optimizes video format for social media (9:16 aspect ratio)
- 🎨 Adds professional white frame padding
- 🎵 Balances audio between original and narration tracks
- 💡 Generates engaging comments using AI
- 🏷️ Supports custom watermarks

## Requirements 📋

- Python 3.8+
- FFmpeg
- Google Cloud Text-to-Speech API credentials
- OpenAI API key
- DeepSeek API key
- Cloudinary account
- Telegram Bot token

## Installation 🚀

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
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS_JSON=your_google_credentials_json_string

# Logging Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Docker Deployment 🐳

1. Build the Docker image:
```bash
docker build -t twitter-video-bot .
```

2. Run the container:
```bash
docker run -d \
  --name twitter-video-bot \
  --env-file .env \
  twitter-video-bot
```

## Usage 📱

1. Start a chat with your bot on Telegram
2. Send a Twitter/X video URL
3. Wait for the bot to process and return the enhanced video

### Commands

- `/start` - Display welcome message and instructions
- `/process <tweet_url>` - Process a specific tweet URL
- `/watermark <text>` - Set custom watermark for your videos

### Example

```
https://x.com/username/status/123456789
```

## Project Structure 📁

```
project_root/
├── main.py              # Main bot implementation
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (not in git)
├── Dockerfile          # Docker configuration
├── .dockerignore       # Docker ignore rules
├── .gitignore          # Git ignore rules
└── README.md           # Documentation
```

## Environment Variables 🔐

Create a `.env` file with the following variables:

### Required Variables
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `OPENAI_API_KEY`: OpenAI API key for content analysis
- `DEEPSEEK_API_KEY`: DeepSeek API key for comment generation
- `CLOUDINARY_CLOUD_NAME`: Cloudinary cloud name
- `CLOUDINARY_API_KEY`: Cloudinary API key
- `CLOUDINARY_API_SECRET`: Cloudinary API secret
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Google Cloud service account credentials JSON string

### Optional Variables
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Dependencies 📦

Key dependencies and their purposes:
- `python-telegram-bot`: Telegram bot framework
- `yt-dlp`: Video downloading
- `ffmpeg-python`: Video processing
- `cloudinary`: Cloud media management
- `google-cloud-texttospeech`: Text-to-speech conversion
- `openai`: AI content analysis
- `python-dotenv`: Environment variable management
- `emoji`: Emoji handling

## Video Processing Pipeline 🔄

1. **Download**: Extract video and metadata from tweet
2. **Frame Extraction**: Capture representative frame
3. **Content Analysis**: Analyze video content using AI
4. **Comment Generation**: Create engaging narration
5. **Speech Generation**: Convert comment to audio
6. **Video Enhancement**: 
   - Add padding for 9:16 aspect ratio
   - Add white frame
   - Mix audio tracks
   - Apply watermark

## Error Handling 🛠️

The bot includes robust error handling for:
- Invalid URLs
- Download failures
- API errors
- Processing timeouts
- Resource cleanup

## Timeouts ⏱️

- Content Analysis: 60 seconds
- Speech Generation: 30 seconds
- Video Processing: 300 seconds (5 minutes)
- API Calls: 30 seconds

## Logging 📝

- Console output with color-coded levels
- Daily log files with detailed information
- Separate logs for different components
- Debug logging for troubleshooting

## Security 🔒

- Environment variables for sensitive data
- Temporary file cleanup
- Resource tracking and verification
- Secure credential handling

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License 📄

[MIT](https://choosealicense.com/licenses/mit/)

## Support 💬

For support, contact [@AutomatorByMani](https://t.me/AutomatorByMani) on Telegram.

## Acknowledgments 🙏

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg](https://ffmpeg.org/)
- [Cloudinary](https://cloudinary.com/)
- [Google Cloud](https://cloud.google.com/)
- [OpenAI](https://openai.com/)
- [DeepSeek](https://deepseek.com/) 