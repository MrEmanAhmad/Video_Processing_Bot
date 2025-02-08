"""
Main entry point for video analysis pipeline.
Orchestrates the execution of all pipeline steps.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from pipeline.Step_4_generate_commentary import CommentaryStyle
from pipeline import (
    Step_1_download_video,
    Step_2_extract_frames,
    Step_3_analyze_frames,
    Step_4_generate_commentary,
    Step_5_generate_audio,
    Step_6_video_generation,
    Step_7_cleanup
)

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_url> [commentary_style]")
        print("\nAvailable commentary styles:")
        print("  documentary - Professional and informative (default)")
        print("  energetic  - Dynamic and enthusiastic")
        print("  analytical - Detailed and technical")
        print("  storyteller - Narrative and emotional")
        sys.exit(1)
    
    # Parse command line arguments
    url = sys.argv[1]
    style_name = sys.argv[2].lower() if len(sys.argv) > 2 else "documentary"
    
    # Select commentary style
    style_map = {
        "documentary": CommentaryStyle.DOCUMENTARY,
        "energetic": CommentaryStyle.ENERGETIC,
        "analytical": CommentaryStyle.ANALYTICAL,
        "storyteller": CommentaryStyle.STORYTELLER
    }
    
    if style_name not in style_map:
        logger.error(f"Unknown style: {style_name}")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"analysis_{timestamp}")
    final_video = None
    
    try:
        logger.debug("Starting video analysis pipeline...")
        
        # Step 1: Download video
        success, metadata, video_title = Step_1_download_video.execute_step(url, output_dir)
        if not success:
            sys.exit(1)
            
        video_file = next(Path(output_dir / "video").glob("*.mp4"))
        
        # Step 2: Extract frames
        key_frames, scene_changes, motion_scores = Step_2_extract_frames.execute_step(
            video_file=video_file,
            output_dir=output_dir
        )
        
        # Step 3: Analyze frames
        analysis_results = Step_3_analyze_frames.execute_step(
            frames_dir=output_dir / "frames",
            output_dir=output_dir,
            metadata=metadata,
            scene_changes=scene_changes,
            motion_scores=motion_scores,
            video_duration=metadata.get('duration', 0)
        )
        
        # Step 4: Generate commentary
        commentary, audio_script = Step_4_generate_commentary.execute_step(
            analysis_file=output_dir / "final_analysis.json",
            output_dir=output_dir,
            style=style_map[style_name]
        )
        
        # Step 5: Generate audio
        if audio_script:
            audio_file = Step_5_generate_audio.execute_step(
                audio_script=audio_script,
                output_dir=output_dir,
                style_name=style_name
            )
            
            # Step 6: Generate final video with audio
            if audio_file:
                final_video = Step_6_video_generation.execute_step(
                    video_file=video_file,
                    audio_file=audio_file,
                    output_dir=output_dir,
                    style_name=style_name
                )
        
        # Step 7: Final cleanup
        keep_files = [
            "final_analysis.json",
            "video_metadata.json",
            f"commentary_{style_name}.json",
            f"commentary_{style_name}.txt"
        ]
        if final_video:
            keep_files.append(final_video.name)
            
        Step_7_cleanup.execute_step(output_dir, style_name, keep_files)
        
        logger.info("Processing complete!")
        logger.info(f"Results saved in: {output_dir}")
        if final_video:
            logger.info(f"Final video generated: {final_video}")
        
    except Exception as e:
        logger.exception(f"Error in processing pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 