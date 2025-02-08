"""
Step 4: Commentary generation module
Generates styled commentary based on frame analysis
"""
import json
import logging
import os
import re
import random
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from openai import OpenAI

logger = logging.getLogger(__name__)

class CommentaryStyle(Enum):
    """Available commentary styles."""
    DOCUMENTARY = "documentary"
    ENERGETIC = "energetic"
    ANALYTICAL = "analytical"
    STORYTELLER = "storyteller"

class CommentaryGenerator:
    """Generates video commentary using OpenAI."""
    
    def __init__(self, style: CommentaryStyle):
        """
        Initialize commentary generator.
        
        Args:
            style: Style of commentary to generate
        """
        self.style = style
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
    def _build_system_prompt(self) -> str:
        """Build system prompt based on commentary style."""
        base_prompt = """You are a charismatic and engaging video commentator who speaks naturally and conversationally.
Your goal is to create commentary that sounds like a real person talking, not a scripted narration.

Key guidelines for natural speech:
1. Use casual language and contractions (it's, here's, that's)
2. Add natural filler words and pauses (um, you know, like, well)
3. Express genuine reactions and emotions
4. Use varied sentence lengths and rhythms
5. Include brief personal observations
6. React to what you're seeing in real-time
7. Sound enthusiastic and interested

Make it feel like a friend excitedly telling someone about an interesting video they just watched."""
        
        style_prompts = {
            CommentaryStyle.DOCUMENTARY: """Blend professional insights with casual observations. 
Think David Attenborough meets everyday conversation - knowledgeable but approachable.
React naturally to interesting moments with phrases like "Oh wow, look at that!" or "You know what's fascinating here...".""",

            CommentaryStyle.ENERGETIC: """Be super enthusiastic and dynamic! 
Use expressions like "Oh my gosh!", "This is incredible!", and "You won't believe what happens next!"
Let your genuine excitement shine through with natural reactions and energetic tone.""",

            CommentaryStyle.ANALYTICAL: """Share technical insights in a conversational way.
Think of explaining complex things to a friend - "So basically...", "You see what's happening here is...", "The interesting thing about this..."
Mix expertise with natural curiosity and excitement.""",

            CommentaryStyle.STORYTELLER: """Tell the story like you're sharing an amazing experience with a friend.
Use phrases like "Picture this...", "You're not gonna believe this but...", "Here's the cool part..."
React emotionally to the story as it unfolds."""
        }
        
        return base_prompt + style_prompts[self.style]

    def _analyze_scene_sequence(self, frames: List[Dict]) -> Dict:
        """
        Analyze the sequence of scenes to identify narrative patterns.
        
        Args:
            frames: List of frame analysis dictionaries
            
        Returns:
            Dictionary containing scene sequence analysis
        """
        sequence = {
            "timeline": [],
            "key_objects": set(),
            "recurring_elements": set(),
            "scene_transitions": []
        }

        for frame in frames:
            timestamp = float(frame['timestamp'])
            
            # Track objects and elements
            if 'google_vision' in frame:
                objects = set(frame['google_vision']['objects'])
                sequence['key_objects'].update(objects)
                
                # Check for recurring elements
                if len(sequence['timeline']) > 0:
                    prev_objects = set(sequence['timeline'][-1].get('objects', []))
                    recurring = objects.intersection(prev_objects)
                    sequence['recurring_elements'].update(recurring)
            
            # Track scene transitions
            if len(sequence['timeline']) > 0:
                prev_time = sequence['timeline'][-1]['timestamp']
                if timestamp - prev_time > 2.0:  # Significant time gap
                    sequence['scene_transitions'].append(timestamp)
            
            sequence['timeline'].append({
                'timestamp': timestamp,
                'objects': list(objects) if 'google_vision' in frame else [],
                'description': frame.get('openai_vision', {}).get('detailed_description', '')
            })
        
        # Convert sets to lists before returning
        sequence['key_objects'] = list(sequence['key_objects'])
        sequence['recurring_elements'] = list(sequence['recurring_elements'])
        
        return sequence

    def _estimate_speech_duration(self, text: str) -> float:
        """
        Estimate the duration of speech in seconds.
        Based on average speaking rate of ~150 words per minute.
        
        Args:
            text: Text to estimate duration for
            
        Returns:
            Estimated duration in seconds
        """
        words = len(text.split())
        return (words / 150) * 60  # Convert from minutes to seconds

    def _build_narration_prompt(self, analysis: Dict, sequence: Dict) -> str:
        """
        Build a prompt specifically for generating narration-friendly commentary.
        
        Args:
            analysis: Video analysis dictionary
            sequence: Scene sequence analysis
            
        Returns:
            Narration-optimized prompt string
        """
        video_duration = float(analysis['metadata'].get('duration', 0))
        video_title = analysis['metadata'].get('title', '')
        video_description = analysis['metadata'].get('description', '')
        
        # Target shorter duration to ensure final audio fits
        target_duration = max(video_duration * 0.8, video_duration - 2)  # 80% of duration or 2 seconds shorter
        target_words = int(target_duration * 2.0)  # Target 2 words per second for natural pace
        
        # Collect all scene descriptions and objects
        all_descriptions = []
        for item in sequence['timeline']:
            if item['description']:
                all_descriptions.append(item['description'])
        
        prompt = f"""Create a short, engaging, and natural commentary for a {self.style.value} style video that feels like someone casually describing what they're watching.

Video Context:
- Title: {video_title}
- Description: {video_description}
- Duration: {video_duration:.1f} seconds
- Word Limit: {target_words} words (to fit the duration)
- Key Elements: {', '.join(sorted(sequence['key_objects']))}

Important Guidelines:
1. Use the video title and description as the primary context for the commentary
2. Create ONE continuous, flowing commentary that tells a cohesive story
3. Incorporate key points from the video description into your commentary
4. Use natural, conversational language like you're excitedly telling a friend
5. Start with an engaging opener that references the video's context
6. React naturally to what you see while staying true to the video's theme
7. End with a brief personal observation that ties back to the video's description
8. Keep it concise to fit within the video duration

Key Scenes to Cover (in context of the description):
{chr(10).join('- ' + desc for desc in all_descriptions)}

Remember: Make it sound completely natural and conversational, like someone watching and reacting to the video in real-time while understanding its intended message and context."""

        return prompt
    
    def generate_commentary(self, analysis_file: Path, output_file: Path) -> Optional[Dict]:
        """
        Generate commentary from analysis results.
        
        Args:
            analysis_file: Path to analysis results JSON
            output_file: Path to save generated commentary
            
        Returns:
            Generated commentary dictionary if successful, None otherwise
        """
        try:
            # Load analysis results
            with open(analysis_file, encoding='utf-8') as f:
                analysis = json.load(f)
            
            video_duration = float(analysis['metadata'].get('duration', 0))
            sequence = self._analyze_scene_sequence(analysis['frames'])
            
            # Generate narration-optimized commentary
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_narration_prompt(analysis, sequence)}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            commentary_text = response.choices[0].message.content
            estimated_duration = self._estimate_speech_duration(commentary_text)
            
            # If estimated duration is too long, try up to 3 times to get shorter version
            attempts = 0
            while estimated_duration > video_duration and attempts < 3:
                attempts += 1
                logger.debug(f"Commentary too long ({estimated_duration:.1f}s), attempt {attempts}/3...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-2024-04-09",
                    messages=[
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": self._build_narration_prompt(analysis, sequence)},
                        {"role": "assistant", "content": commentary_text},
                        {"role": "user", "content": f"The commentary is still too long. Create an extremely concise version using no more than {int(video_duration * 1.8)} words total. Focus only on the most essential elements."}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                commentary_text = response.choices[0].message.content
                estimated_duration = self._estimate_speech_duration(commentary_text)
            
            commentary = {
                "style": self.style.value,
                "commentary": commentary_text,
                "metadata": analysis['metadata'],
                "scene_analysis": sequence,
                "estimated_duration": estimated_duration,
                "word_count": len(commentary_text.split())
            }
            
            # Save commentary
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(commentary, f, indent=2, ensure_ascii=False)
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating commentary: {str(e)}")
            return None
    
    def format_for_audio(self, commentary: Dict) -> str:
        """
        Format commentary for text-to-speech.
        
        Args:
            commentary: Generated commentary dictionary
            
        Returns:
            Formatted text suitable for audio generation
        """
        text = commentary['commentary']
        
        # Clean up text for narration
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([.!?])\s*', r'\1\n', text)  # Add line breaks after sentences
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple line breaks
        text = re.sub(r'[*_]', '', text)  # Remove markdown formatting
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis
        
        # Add natural speech patterns and pauses
        text = re.sub(r'([.!?])\n', r'\1 <break time="1s"/>\n', text)  # Longer pauses between sentences
        text = re.sub(r'([,;:])\s', r'\1 <break time="0.3s"/> ', text)  # Short pauses after punctuation
        text = re.sub(r'\.\.\.\s', '... <break time="0.5s"/> ', text)  # Medium pauses after ellipsis
        
        # Add occasional filler words and natural patterns
        sentences = text.split('\n')
        enhanced_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Randomly add filler words at the start of some sentences
                if i > 0 and random.random() < 0.3:  # 30% chance
                    fillers = ['Um... ', 'Ah... ', 'Well... ', 'You see... ', 'So... ', 'And... ', 'Now... ']
                    sentence = random.choice(fillers) + sentence
                
                # Add thoughtful pauses within longer sentences
                if len(sentence.split()) > 8 and random.random() < 0.4:  # 40% chance for long sentences
                    words = sentence.split()
                    mid = len(words) // 2
                    words.insert(mid, '<break time="0.2s"/>')
                    sentence = ' '.join(words)
                
                enhanced_sentences.append(sentence)
        
        return '\n'.join(enhanced_sentences).strip()

def execute_step(
    analysis_file: Path,
    output_dir: Path,
    style: CommentaryStyle
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute commentary generation step.
    
    Args:
        analysis_file: Path to the analysis results file
        output_dir: Directory to save commentary
        style: Style of commentary to generate
        
    Returns:
        Tuple containing:
        - Generated commentary (dict or None)
        - Formatted audio script (str or None)
    """
    logger.debug(f"Step 4: Generating commentary...")
    
    commentary_gen = CommentaryGenerator(style=style)
    commentary = commentary_gen.generate_commentary(
        analysis_file=analysis_file,
        output_file=output_dir / f"commentary_{style.name.lower()}.json"
    )
    
    audio_script = None
    if commentary:
        audio_script = commentary_gen.format_for_audio(commentary)
        script_file = output_dir / f"audio_script_{style.name.lower()}.txt"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(audio_script)
            
    return commentary, audio_script 