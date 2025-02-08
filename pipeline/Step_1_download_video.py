"""
Step 1: Video download module
Downloads videos from various sources using yt-dlp
"""

import logging
import os
import re
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import yt_dlp
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoDownloader:
    """Downloads videos using yt-dlp."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize video downloader.
        
        Args:
            output_dir: Directory to save downloaded video
        """
        self.output_dir = output_dir
        
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to ensure compatibility."""
        # Convert x.com to twitter.com
        if 'x.com' in url:
            url = url.replace('x.com', 'twitter.com')
        return url
        
    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitize the filename to remove problematic characters.
        
        Args:
            title: Original video title
            
        Returns:
            Sanitized filename
        """
        # Remove emojis and special characters
        title = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces and dashes with underscores
        title = re.sub(r'[-\s]+', '_', title)
        # Ensure it's not empty and not too long
        if not title:
            title = 'video'
        return title[:100].strip('_')  # Limit length to 100 chars
        
    def _get_ydl_opts(self) -> Dict[str, Any]:
        """Get yt-dlp options."""
        video_dir = self.output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Use timestamp for filename instead of video title
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'outtmpl': str(video_dir / f'video_{timestamp}.%(ext)s'),  # Simple timestamp-based filename
            'progress_hooks': [self._progress_hook],
            'verbose': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'format': 'best',
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'no_warnings': False,
            'extract_flat': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate'
            },
            'postprocessors': [{
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4'
            }],
            'compat_opts': ['no-youtube-unavailable-videos']
        }
    
    def _progress_hook(self, d: Dict[str, Any]) -> None:
        """
        Progress hook for download status.
        
        Args:
            d: Download status dictionary
        """
        if d['status'] == 'finished':
            logger.info('Download completed')
            # No need to rename since we're using timestamp-based names
            logger.info(f'Downloaded file: {d["filename"]}')
                
    def download(self, url: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Download video from URL.
        
        Args:
            url: Video URL
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Video metadata (dict or None)
            - Video title (str or None)
        """
        try:
            # Normalize URL first
            url = self._normalize_url(url)
            logger.info(f"Downloading video from: {url}")
            
            with yt_dlp.YoutubeDL(self._get_ydl_opts()) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=True)
                
                if not info:
                    raise Exception("No video information extracted")
                
                # Store original title in metadata but use timestamp for filename
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
                
                # Save metadata to file
                metadata_file = self.output_dir / "video_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                return True, metadata, self._sanitize_filename(info.get('title', 'video'))
                
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return False, None, None

def execute_step(url_or_path: str, output_dir: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Execute video download step.
    
    Args:
        url_or_path: Video URL or local file path
        output_dir: Directory to save downloaded video
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Video metadata (dict or None)
        - Video title (str or None)
    """
    # Check if input is a local file
    if os.path.isfile(url_or_path):
        try:
            # Create video directory
            video_dir = output_dir / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to output directory
            import shutil
            filename = os.path.basename(url_or_path)
            sanitized_name = VideoDownloader(output_dir)._sanitize_filename(filename)
            dest_path = video_dir / f"{sanitized_name}.mp4"
            shutil.copy2(url_or_path, dest_path)
            
            # Create basic metadata
            metadata = {
                'title': sanitized_name,
                'duration': 0,  # Will be updated later
                'description': 'Local video file',
                'uploader': 'Local',
                'upload_date': ''
            }
            
            return True, metadata, sanitized_name
            
        except Exception as e:
            logger.error(f"Error copying local file: {str(e)}")
            return False, None, None
    
    # Download from URL
    downloader = VideoDownloader(output_dir)
    success, metadata, video_title = downloader.download(url_or_path)
    
    if not success:
        logger.error("Video download failed")
        
    return success, metadata, video_title 