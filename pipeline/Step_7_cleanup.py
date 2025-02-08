"""
Step 6: Cleanup module
Cleans up temporary files and directories
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

def cleanup_workspace(output_dir: Union[str, Path], keep_files: Optional[List[str]] = None) -> None:
    """Clean up workspace keeping only specified files."""
    keep_files = keep_files or ["final_analysis.json", "video_metadata.json"]
    output_dir = Path(output_dir)
    
    # Delete files
    for item in output_dir.glob("**/*"):
        if item.is_file() and item.name not in keep_files:
            try:
                item.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {item}: {e}")
    
    # Remove empty directories
    for item in output_dir.glob("**/*"):
        if item.is_dir():
            try:
                item.rmdir()
            except Exception:
                pass

def execute_step(output_dir: Path, style_name: str, keep_files: Optional[List[str]] = None) -> None:
    """
    Execute cleanup step.
    
    Args:
        output_dir: Directory to clean up
        style_name: Name of the commentary style used
        keep_files: List of filenames to keep (optional)
    """
    logger.debug("Step 6: Cleaning up workspace...")
    
    # Files to keep
    default_keep_files = [
        "final_analysis.json",
        "video_metadata.json",
        f"commentary_{style_name}.wav",
        f"commentary_{style_name}.txt"
    ]
    
    # Combine default and custom keep files
    if keep_files:
        default_keep_files.extend(keep_files)
    
    # Clean up workspace
    cleanup_workspace(output_dir, default_keep_files)
    logger.debug("Cleanup complete") 