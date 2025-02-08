"""
Step 3: Frame analysis module
Analyzes extracted frames using Google Vision and OpenAI Vision APIs
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.cloud import vision
from openai import OpenAI

logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """Handles image analysis using multiple vision APIs with optimized usage."""
    
    def __init__(self, frames_dir: Path, output_dir: Path, metadata: Optional[dict] = None):
        """
        Initialize vision analyzer.
        
        Args:
            frames_dir: Directory containing frames to analyze
            output_dir: Directory to save analysis results
            metadata: Video metadata dictionary
        """
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.metadata = metadata or {}
        
        # Initialize API clients
        self.vision_client = vision.ImageAnnotatorClient()
        self.openai_client = OpenAI()  # Initialize without explicit API key
        
        # Analysis storage
        self.google_vision_results = {}
        self.openai_results = {}
    
    def select_key_frames(self, scene_changes: List[Path], motion_scores: List[Tuple[Path, float]], max_frames: int = 8) -> List[Path]:
        """
        Select key frames for detailed analysis.
        Prioritizes scene changes and high motion frames.
        
        Args:
            scene_changes: List of frames where scene changes were detected
            motion_scores: List of tuples containing (frame_path, motion_score)
            max_frames: Maximum number of frames to select (default: 8)
            
        Returns:
            List of selected frame paths
        """
        selected_frames = []
        
        # Include all scene changes up to half of max_frames
        scene_limit = max_frames // 2
        if scene_changes:
            selected_frames.extend(scene_changes[:scene_limit])
        
        # Sort motion scores by magnitude
        sorted_motion = sorted(motion_scores, key=lambda x: x[1], reverse=True)
        
        # Add highest motion frames that aren't too close to already selected frames
        for frame_path, _ in sorted_motion:
            if len(selected_frames) >= max_frames:
                break
                
            # Check if frame is sufficiently different in time from selected frames
            frame_time = float(frame_path.name.split('_')[1].replace('s.jpg', ''))
            is_unique = all(
                abs(float(f.name.split('_')[1].replace('s.jpg', '')) - frame_time) > 2.0  # Reduced time difference threshold
                for f in selected_frames
            )
            
            if is_unique and frame_path not in selected_frames:
                selected_frames.append(frame_path)
        
        return selected_frames
    
    def analyze_frame_google_vision(self, frame_path: Path) -> Tuple[Optional[dict], bool]:
        """
        Analyze a frame using Google Vision API.
        Optimized to use only essential features.
        """
        try:
            with open(frame_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.vision_client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.LABEL_DETECTION},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
                ]
            })
            
            return {
                "labels": [label.description for label in response.label_annotations],
                "objects": [obj.name for obj in response.localized_object_annotations],
                "confidence": response.label_annotations[0].score if response.label_annotations else 0
            }, True
        except Exception as e:
            logger.error(f"Google Vision API error: {str(e)}")
            return None, False
    
    def analyze_frame_openai(self, frame_path: Path, google_analysis: Optional[dict] = None) -> Tuple[Optional[dict], bool]:
        """
        Analyze a frame using OpenAI Vision API.
        Provides detailed scene understanding.
        """
        try:
            with open(frame_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_openai_prompt(google_analysis)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            },
                        ],
                    }
                ]
            )
            
            return {"detailed_description": completion.choices[0].message.content}, True
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {str(e)}")
            return None, False
    
    def _build_openai_prompt(self, google_analysis: Optional[dict] = None) -> str:
        """Build prompt for OpenAI Vision API analysis."""
        prompt = f"""Analyze this frame in detail, considering both the visual content and the following context:

Video Title: {self.metadata.get('title', 'Unknown')}
Description: {self.metadata.get('description', 'No description available')}

Previous computer vision analysis detected:"""
        
        if google_analysis:
            if google_analysis.get("labels"):
                prompt += "\nKey elements detected:"
                for label in google_analysis["labels"][:5]:
                    prompt += f"\n- {label}"
            
            if google_analysis.get("objects"):
                prompt += "\n\nObjects detected:"
                for obj in google_analysis["objects"][:5]:
                    prompt += f"\n- {obj}"
        
        prompt += """

Please provide a comprehensive analysis that:
1. Describes the main focus or subject of this frame
2. Explains any actions, movements, or interactions
3. Notes significant details or patterns
4. Provides context about how this moment fits into the video's narrative"""
        
        return prompt
    
    def analyze_video(self, scene_changes: List[Path], motion_scores: List[Tuple[Path, float]], video_duration: float) -> dict:
        """
        Main analysis workflow with optimized API usage.
        Analyzes more frames with Google Vision and uses OpenAI for final confirmation.
        
        Args:
            scene_changes: List of frames where scene changes were detected
            motion_scores: List of tuples containing (frame_path, motion_score)
            video_duration: Duration of the video in seconds
            
        Returns:
            Dictionary containing analysis results
        """
        final_results = {"metadata": self.metadata, "frames": []}
        
        # Select key frames for analysis
        key_frames = self.select_key_frames(scene_changes, motion_scores)
        logger.info(f"Selected {len(key_frames)} key frames for analysis")
        
        # Analyze all selected frames with Google Vision
        google_vision_results = []
        for frame_path in key_frames:
            frame_result = {
                "frame": frame_path.name,
                "timestamp": frame_path.name.split('_')[1].replace('s.jpg', '')
            }
            
            # Google Vision Analysis for all frames
            google_analysis, success = self.analyze_frame_google_vision(frame_path)
            if success:
                frame_result["google_vision"] = google_analysis
                google_vision_results.append(frame_result)
                final_results["frames"].append(frame_result)
        
        # Select one representative frame for OpenAI analysis
        if google_vision_results:
            # Choose the frame with the highest confidence score
            best_frame = max(google_vision_results, 
                           key=lambda x: x["google_vision"].get("confidence", 0))
            frame_path = self.frames_dir / best_frame["frame"]
            
            # OpenAI Vision Analysis for final confirmation
            openai_analysis, success = self.analyze_frame_openai(
                frame_path,
                {
                    "labels": list(set(
                        label
                        for result in google_vision_results
                        for label in result["google_vision"].get("labels", [])
                    )),
                    "objects": list(set(
                        obj
                        for result in google_vision_results
                        for obj in result["google_vision"].get("objects", [])
                    ))
                }
            )
            
            if success:
                # Add OpenAI analysis to the best frame
                for frame in final_results["frames"]:
                    if frame["frame"] == best_frame["frame"]:
                        frame["openai_vision"] = openai_analysis
                        break
        
        # Save results
        analysis_file = self.output_dir / "final_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {analysis_file}")
        return final_results

def execute_step(
    frames_dir: Path,
    output_dir: Path,
    metadata: dict,
    scene_changes: List[Path],
    motion_scores: List[Tuple[Path, float]],
    video_duration: float
) -> dict:
    """
    Execute frame analysis step.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_dir: Directory to save analysis results
        metadata: Video metadata dictionary
        scene_changes: List of frames where scene changes were detected
        motion_scores: List of tuples containing (frame path, motion score)
        video_duration: Duration of the video in seconds
        
    Returns:
        Dictionary containing analysis results
    """
    logger.debug("Step 3: Analyzing frames...")
    
    analyzer = VisionAnalyzer(frames_dir, output_dir, metadata)
    results = analyzer.analyze_video(scene_changes, motion_scores, video_duration)
    
    logger.debug(f"Analyzed {len(results['frames'])} frames")
    return results 