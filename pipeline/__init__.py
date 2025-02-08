"""
Pipeline package initialization.
Provides modular steps for video analysis and commentary generation.
"""

from .Step_4_generate_commentary import CommentaryStyle
from . import (
    Step_1_download_video,
    Step_2_extract_frames,
    Step_3_analyze_frames,
    Step_4_generate_commentary,
    Step_5_generate_audio,
    Step_6_video_generation,
    Step_7_cleanup
)

__all__ = [
    'CommentaryStyle',
    'Step_1_download_video',
    'Step_2_extract_frames',
    'Step_3_analyze_frames',
    'Step_4_generate_commentary',
    'Step_5_generate_audio',
    'Step_6_video_generation',
    'Step_7_cleanup'
] 