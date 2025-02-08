# Video Commentary Bot

A Telegram bot that generates engaging commentary for videos using AI.

## Features

- Multiple commentary styles (Documentary, Energetic, Analytical, Storyteller)
- Intelligent video analysis using Google Cloud Vision AI
- Natural language commentary generation
- Professional audio synthesis
- Automatic video processing and generation

## Deployment on Railway

1. Fork this repository to your GitHub account

2. Create a new project on Railway and connect it to your GitHub repository

3. Add the following environment variables in Railway:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_APPLICATION_CREDENTIALS_JSON=your_google_credentials_json
   CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_secret
   ```

4. Deploy! Railway will automatically build and deploy your bot using the Dockerfile

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-commentary-bot.git
   cd video-commentary-bot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your credentials (see `.env.example`)

5. Run the bot:
   ```bash
   python bot.py
   ```

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather
- `OPENAI_API_KEY`: OpenAI API key for commentary generation
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Google Cloud credentials JSON string
- `CLOUDINARY_CLOUD_NAME`: Cloudinary cloud name
- `CLOUDINARY_API_KEY`: Cloudinary API key
- `CLOUDINARY_API_SECRET`: Cloudinary API secret

## System Requirements

- Python 3.10 or higher
- FFmpeg
- OpenCV dependencies

## Docker Support

Build the Docker image:
```bash
docker build -t video-commentary-bot .
```

Run the container:
```bash
docker run -d --env-file .env video-commentary-bot
```

## License

MIT License - feel free to use and modify for your own projects! 