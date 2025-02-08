"""
Step 6: Video generation module
Combines video and audio using Cloudinary for professional video processing
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional, Dict
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary import CloudinaryVideo
import requests

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Handles video generation and audio overlay using Cloudinary."""
    
    def __init__(self, cloud_name: str, api_key: str, api_secret: str):
        """
        Initialize the VideoGenerator with Cloudinary credentials.
        
        Args:
            cloud_name: Cloudinary cloud name
            api_key: Cloudinary API key
            api_secret: Cloudinary API secret
        """
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )
        self.uploaded_resources = []
        self._setup_cloudinary_config()
        
    def _setup_cloudinary_config(self):
        """Configure Cloudinary for optimal performance."""
        cloudinary.config(
            secure=True,
            chunk_size=6000000,  # 6MB chunks for faster uploads
            use_cache=True,
            cache_duration=3600  # 1 hour cache
        )
        
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for Cloudinary public ID.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename suitable for Cloudinary
        """
        # Remove file extension
        filename = os.path.splitext(filename)[0]
        # Remove emojis and special characters
        filename = re.sub(r'[^\w\s-]', '', filename)
        # Replace spaces with underscores
        filename = re.sub(r'[-\s]+', '_', filename)
        # Ensure it's not empty
        if not filename:
            filename = 'video'
        return filename.strip('_')
        
    def upload_media(self, file_path: str, resource_type: str) -> Optional[Dict]:
        """
        Upload media file to Cloudinary with optimized settings.
        
        Args:
            file_path: Path to the media file
            resource_type: Type of resource ('video' or 'raw' for audio)
            
        Returns:
            Upload response if successful, None otherwise
        """
        try:
            # Sanitize the filename for the public_id
            public_id = self._sanitize_filename(os.path.basename(file_path))
            logger.info(f"Uploading {resource_type}: {file_path}")
            
            # Optimize upload settings
            response = cloudinary.uploader.upload(
                file_path,
                resource_type=resource_type,
                public_id=public_id,
                overwrite=True,
                chunk_size=6000000,  # 6MB chunks
                eager_async=True,  # Async transformations
                eager=[  # Pre-generate common transformations
                    {"quality": "auto:good"},
                    {"fetch_format": "auto"}
                ],
                use_filename=True,
                unique_filename=False,
                invalidate=True
            )
            
            logger.info(f"Upload successful. Public ID: {response['public_id']}")
            self.uploaded_resources.append(response['public_id'])
            return response
        except Exception as e:
            logger.error(f"Error uploading media: {str(e)}")
            return None
            
    def cleanup_resources(self):
        """Clean up uploaded resources from Cloudinary."""
        for resource_id in self.uploaded_resources:
            try:
                cloudinary.uploader.destroy(resource_id)
                logger.info(f"Cleaned up resource: {resource_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up resource {resource_id}: {str(e)}")
        self.uploaded_resources = []
            
    def generate_video(self, video_id: str, audio_id: str, output_path: Path) -> Optional[Path]:
        """
        Generate final video with optimized processing.
        
        Args:
            video_id: Public ID of uploaded video
            audio_id: Public ID of uploaded audio
            output_path: Path to save the final video
            
        Returns:
            Path to generated video if successful, None otherwise
        """
        try:
            video = CloudinaryVideo(video_id)
            
            # Get video details
            details = cloudinary.api.resource(video_id, resource_type='video')
            width = details.get('width', 0)
            height = details.get('height', 0)
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height else 0
            target_ratio = 9/16
            
            # Optimize transformations
            transformation = [
                {'quality': 'auto:good'},
                {'fetch_format': 'auto'}
            ]
            
            # Add padding only if needed
            if abs(aspect_ratio - target_ratio) > 0.01:
                logger.info(f"Adding vertical padding (ratio: {aspect_ratio:.2f}, target: {target_ratio:.2f})")
                target_height = int(width * (16/9))
                
                transformation.extend([
                    {
                        'width': width,
                        'height': target_height,
                        'crop': "pad",
                        'background': "white",
                        'y_padding': "auto",
                        'gravity': "center"
                    }
                ])
            
            # Add audio with optimized settings
            transformation.extend([
                {
                    'overlay': f"video:{audio_id}",
                    'resource_type': "video"
                },
                {
                    'flags': 'layer_apply',
                    'audio_codec': 'aac',
                    'bit_rate': '192k'
                }
            ])
            
            # Generate optimized video URL
            video_url = video.build_url(
                transformation=transformation,
                resource_type='video',
                format='mp4',
                secure=True
            )
            
            # Download with streaming and chunking
            with requests.get(video_url, stream=True) as response:
                if response.status_code == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"Video generated successfully: {output_path}")
                    return output_path
                else:
                    logger.error(f"Error downloading video: {response.status_code}")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return None
        finally:
            self.cleanup_resources()

def execute_step(
    video_file: Path,
    audio_file: Path,
    output_dir: Path,
    style_name: str
) -> Optional[Path]:
    """
    Execute video generation step.
    
    Args:
        video_file: Path to the input video file
        audio_file: Path to the generated audio file
        output_dir: Directory to save generated video
        style_name: Name of the commentary style used
        
    Returns:
        Path to the generated video if successful, None otherwise
    """
    logger.debug("Step 6: Generating final video...")
    
    # Initialize video generator with Cloudinary credentials
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    
    if not all([cloud_name, api_key, api_secret]):
        logger.error("Missing Cloudinary credentials")
        return None
    
    generator = VideoGenerator(cloud_name, api_key, api_secret)
    
    try:
        # Upload video and audio
        video_response = generator.upload_media(str(video_file), 'video')
        audio_response = generator.upload_media(str(audio_file), 'video')  # Use video type for audio to support overlay
        
        if not video_response or not audio_response:
            return None
        
        # Generate final video
        output_file = output_dir / f"final_video_{style_name}.mp4"
        result = generator.generate_video(
            video_response['public_id'],
            audio_response['public_id'],
            output_file
        )
        
        if result:
            logger.debug(f"Video generated successfully: {output_file}")
            return output_file
        else:
            logger.error("Video generation failed")
            return None
            
    except Exception as e:
        logger.error(f"Error in video generation step: {str(e)}")
        return None 