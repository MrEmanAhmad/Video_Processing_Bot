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
        base_prompt = """You are a charismatic and engaging content creator who reacts to videos in a completely natural, human way.
Your goal is to create commentary that feels like a genuine reaction video, not a description or narration.

Key guidelines for authentic reactions:
1. Focus on emotional responses and personal thoughts rather than describing what's visible
2. Use very casual, conversational language (gonna, wanna, y'know)
3. Express genuine feelings and reactions (aww, omg, wow)
4. Share relatable thoughts and experiences
5. Add personality through tone variations
6. React like you're sharing something amazing with a friend
7. Be enthusiastic but authentic

Make it feel like a TikTok or YouTube reaction video where someone is genuinely excited to share their thoughts."""
        
        style_prompts = {
            CommentaryStyle.DOCUMENTARY: """Be the cool nature enthusiast friend who gets excited about animals and their behaviors.
Share interesting facts naturally, like "You know what's amazing about this?" or "I love how they do this..."
Focus on your emotional connection to what you're seeing rather than just describing it.""",

            CommentaryStyle.ENERGETIC: """React with genuine excitement and enthusiasm!
Share your real feelings with expressions like "I can't even handle how cute this is!" or "This literally made my whole day!"
Make it feel like you're sharing your favorite viral video with your best friend.""",

            CommentaryStyle.ANALYTICAL: """Be the friend who notices fascinating details but keeps it fun and engaging.
Share your thoughts like you're having a casual conversation - "Okay but can we talk about how smart this is?" or "I'm obsessed with how they..."
Mix your expertise with genuine excitement and wonder.""",

            CommentaryStyle.STORYTELLER: """React like you're sharing an incredible moment with your audience.
Use phrases that build emotional connection - "I'm literally melting watching this" or "This reminds me of..."
Focus on the feelings and moments that make the video special."""
        }
        
        return base_prompt + "\n\n" + style_prompts[self.style]

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
        target_duration = max(video_duration * 0.8, video_duration - 2)
        target_words = int(target_duration * 2.0)
        
        prompt = f"""Create a natural, emotional reaction to this {self.style.value} style video that feels like someone sharing their genuine thoughts and feelings.

Video Context:
- Title: {video_title}
- Description: {video_description}
- Duration: {video_duration:.1f} seconds
- Word Target: {target_words} words

Important Guidelines:
1. DON'T describe what's happening in the video - react to it emotionally
2. Share personal thoughts, feelings, and reactions
3. Use very casual, natural language like you're talking to a friend
4. Focus on what makes this video special or meaningful
5. Add relatable comments or experiences
6. Express genuine emotions (joy, wonder, excitement)
7. Keep it concise but authentic
8. Make it feel like a real person reacting in the moment

Remember: 
- Focus on YOUR reaction to the video, not what's in it
- Share why this video resonates with you
- Be genuine and relatable
- Keep the energy high but natural
- Make viewers feel the emotions you're feeling

The goal is to sound like someone who just HAD to share this amazing video with their friends because it made them feel something special."""

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
        Format commentary for text-to-speech with style-specific patterns.
        
        Args:
            commentary: Generated commentary dictionary
            
        Returns:
            Formatted text suitable for audio generation
        """
        text = commentary['commentary']
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s,.!?;:()\-\'\"]+', '', text)  # Keep only basic punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Style-specific speech patterns
        style_patterns = {
            CommentaryStyle.DOCUMENTARY: {
                'fillers': ['You know what...', 'Check this out...', 'Oh wow...', 'Look at that...', 'This is fascinating...'],
                'transitions': ['And here\'s the amazing part...', 'Now watch this...', 'See how...'],
                'emphasis': ['absolutely', 'incredibly', 'fascinating', 'remarkable'],
                'pause_frequency': 0.4  # More thoughtful pauses
            },
            CommentaryStyle.ENERGETIC: {
                'fillers': ['Oh my gosh...', 'This is insane...', 'I can\'t even...', 'Just wait...', 'Are you seeing this...'],
                'transitions': ['But wait there\'s more...', 'And then...', 'This is the best part...'],
                'emphasis': ['literally', 'absolutely', 'totally', 'completely'],
                'pause_frequency': 0.2  # Fewer pauses, more energetic flow
            },
            CommentaryStyle.ANALYTICAL: {
                'fillers': ['Interestingly...', 'You see...', 'What\'s fascinating here...', 'Notice how...'],
                'transitions': ['Let\'s look at this...', 'Here\'s what\'s happening...', 'The key detail is...'],
                'emphasis': ['particularly', 'specifically', 'notably', 'precisely'],
                'pause_frequency': 0.5  # More pauses for analysis
            },
            CommentaryStyle.STORYTELLER: {
                'fillers': ['You know...', 'Picture this...', 'Here\'s the thing...', 'Imagine...'],
                'transitions': ['And this is where...', 'That\'s when...', 'The beautiful part is...'],
                'emphasis': ['magical', 'wonderful', 'touching', 'heartwarming'],
                'pause_frequency': 0.3  # Balanced pauses for storytelling
            }
        }
        
        style_config = style_patterns[self.style]
        
        # Add natural speech patterns and pauses
        sentences = text.split('.')
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sentence = sentence.strip()
            
            # Add style-specific fillers at the start of some sentences
            if i > 0 and random.random() < 0.3:
                sentence = random.choice(style_config['fillers']) + ' ' + sentence
            
            # Add transitions between ideas
            if i > 1 and random.random() < 0.25:
                sentence = random.choice(style_config['transitions']) + ' ' + sentence
            
            # Add emphasis words
            if random.random() < 0.2:
                emphasis = random.choice(style_config['emphasis'])
                words = sentence.split()
                if len(words) > 4:
                    insert_pos = random.randint(2, len(words) - 2)
                    words.insert(insert_pos, emphasis)
                    sentence = ' '.join(words)
            
            # Add thoughtful pauses based on style
            if len(sentence.split()) > 6 and random.random() < style_config['pause_frequency']:
                words = sentence.split()
                mid = len(words) // 2
                words.insert(mid, '<break time="0.2s"/>')
                sentence = ' '.join(words)
            
            enhanced_sentences.append(sentence)
        
        # Join sentences with appropriate pauses
        text = '. '.join(enhanced_sentences)
        
        # Add final formatting and pauses
        text = re.sub(r'([,;])\s', r'\1 <break time="0.2s"/> ', text)  # Short pauses
        text = re.sub(r'([.!?])\s', r'\1 <break time="0.4s"/> ', text)  # Medium pauses
        text = re.sub(r'\.\.\.\s', '... <break time="0.3s"/> ', text)  # Thoughtful pauses
        
        # Add natural variations in pace
        text = re.sub(r'(!)\s', r'\1 <break time="0.2s"/> ', text)  # Quick pauses after excitement
        text = re.sub(r'(\?)\s', r'\1 <break time="0.3s"/> ', text)  # Questioning pauses
        
        # Add occasional emphasis for important words
        for emphasis in style_config['emphasis']:
            text = re.sub(f'\\b{emphasis}\\b', f'<emphasis level="strong">{emphasis}</emphasis>', text)
        
        # Clean up any duplicate breaks or spaces
        text = re.sub(r'\s*<break[^>]+>\s*<break[^>]+>\s*', ' <break time="0.4s"/> ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

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