import os
import re
import time
import torch
import shutil
import requests
import subprocess
import tempfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    TextClip,
    concatenate_videoclips
)
from moviepy.video.fx.all import speedx, time_mirror
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.io import wavfile
from scipy.signal import correlate
from deep_translator import GoogleTranslator
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import load_codec_model, codec_decode
from TTS.api import TTS
from yt_dlp import YoutubeDL
from tqdm import tqdm

# Configure logging
logger = logging.getLogger("VideoTranslator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(  
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class Utterance:
    """
    Represents a single utterance in the transcription.

    Attributes:
        speaker: Speaker identifier
        text: Transcribed text
        start: Start time in milliseconds
        end: End time in milliseconds
        confidence: Confidence score of transcription
        words: List of word-level information
        gender: Speaker's gender
    """

    speaker: str
    text: str
    start: int
    end: int
    confidence: float
    words: List[Dict[str, Any]]
    gender: str


class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass


class FileManager:
    """
    Manages file operations and temporary files.

    Attributes:
        base_dir: Base directory for all files
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
    """

    def __init__(self, base_dir: str = "downloads"):
        """
        Initializes the FileManager instance.

        Args:
            base_dir: Base directory for all files (default: "downloads")
        """
        # Convert to absolute path if relative
        self.base_dir = Path(base_dir).resolve()
        self.temp_dir = self.base_dir / "temp"
        self.output_dir = self.base_dir / "output"
        self.init_directories()
        logger.info(f"FileManager initialized with base directory: {self.base_dir}")

    def init_directories(self):
        """
        Initializes the base, temporary, and output directories.
        """
        try:
            for directory in [self.base_dir, self.temp_dir, self.output_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                if not directory.exists():
                    raise RuntimeError(f"Could not create directory: {directory}")
            logger.info(f"Directories initialized: {self.base_dir}")
        except Exception as e:
            logger.error(f"Error initializing directories: {str(e)}")
            raise

    def get_temp_path(self, prefix: str, suffix: str) -> Path:
        """
        Generates a temporary file path with the given prefix and suffix.

        Args:
            prefix: Prefix for the temporary file name
            suffix: Suffix for the temporary file name

        Returns:
            Temporary file path
        """
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}{suffix}"
        return self.temp_dir / filename

    def get_output_path(self, prefix: str, suffix: str) -> Path:
        """
        Generates an output file path with the given prefix and suffix.

        Args:
            prefix: Prefix for the output file name
            suffix: Suffix for the output file name

        Returns:
            Output file path
        """
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}{suffix}"
        return self.output_dir / filename

    def ensure_path_exists(self, path: Path) -> None:
        """Ensures that the path exists."""
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Removes temporary files older than the specified maximum age.

        Args:
            max_age_hours: Maximum age of temporary files in hours (default: 24)
        """
        current_time = datetime.now().timestamp()
        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    try:
                        file_path.unlink()
                        logger.debug(f"Deleted old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {str(e)}")

    def cleanup_old_outputs(self, max_files: int = 10):
        """
        Removes old output files, keeping only the specified maximum number.

        Args:
            max_files: Maximum number of output files to keep (default: 10)
        """
        files = sorted(
            self.output_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        for file_path in files[max_files:]:
            try:
                file_path.unlink()
                logger.debug(f"Deleted old output file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {str(e)}")

    def save_output(self, source_path: Path, prefix: str) -> Path:
        """
        Saves the output file with the given prefix.

        Args:
            source_path: Path to the source file
            prefix: Prefix for the output file name

        Returns:
            Output file path
        """
        output_path = self.get_output_path(prefix, source_path.suffix)
        shutil.copy2(source_path, output_path)
        logger.info(f"Output saved to: {output_path}")
        return output_path


class TTSGenerator:
    """
    Generates speech using the specified TTS model.

    Attributes:
        model_type: Type of TTS model (default: "tacotron2")
        use_gpu: Whether to use the GPU for processing (default: True)
        file_manager: FileManager instance for managing files
        voice_presets: Mapping of speaker genders to voice presets
        current_voice_index: Current voice index for each speaker gender
    """

    def __init__(
        self,
        model_type: Literal["tacotron2", "bark"] = "tacotron2",
        use_gpu: bool = True,
    ):
        """
        Initializes the TTSGenerator instance.

        Args:
            model_type: Type of TTS model (default: "tacotron2")
            use_gpu: Whether to use the GPU for processing (default: True)
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.file_manager = FileManager()
        self.voice_presets = {
            "male": [
                "v2/de_speaker_0",
                "v2/de_speaker_1",
                "v2/de_speaker_2",
                "v2/de_speaker_4",
                "v2/de_speaker_5",
                "v2/de_speaker_6",
                "v2/de_speaker_7",
                "v2/de_speaker_9",
            ],
            "female": ["v2/de_speaker_3", "v2/de_speaker_8"],
        }
        self.current_voice_index = {"male": 0, "female": 0}
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the specified TTS model.

        Returns:
            Loaded TTS model instance
        """
        logger.info(f"Loading TTS model: {self.model_type}")
        if self.model_type == "tacotron2":
            model = TTS(
                model_name="tts_models/de/thorsten/tacotron2-DDC",
                progress_bar=True,
                gpu=self.use_gpu,
            )
            logger.debug("Tacotron2 model loaded successfully.")
            return model
        elif self.model_type == "bark":
            preload_models()
            logger.debug("Bark models preloaded successfully.")
            return None  # Bark does not require a persistent model instance
        else:
            raise ValueError(f"Unsupported TTS model type: {self.model_type}")

    def get_next_voice(self, gender: str) -> str:
        """
        Returns the next voice preset for the specified speaker gender.

        Args:
            gender: Speaker gender

        Returns:
            Voice preset
        """
        voices = self.voice_presets.get(gender, self.voice_presets["male"])
        voice = voices[self.current_voice_index[gender]]
        self.current_voice_index[gender] = (
            self.current_voice_index[gender] + 1
        ) % len(voices)
        logger.debug(f"Selected voice for {gender}: {voice}")
        return voice

    def clean_text_for_tts(self, text: str) -> str:
        """Cleans text for TTS processing."""
        # Replace problematic characters
        replacements = {
            '"': '',  # Remove quotation marks
            "'": "'",  # Replace smart quotes with normal ones
            '…': '...',  # Replace ellipsis with dots
            '–': '-',  # Replace en-dash with hyphen
            '—': '-',  # Replace em-dash with hyphen
            '\u0361': '',  # Remove combining characters
            '\u035c': '',
            '˞': '',
            'ˈ': '',
            'ˌ': '',
            'ː': '',
            'ɜ': 'e',  # Replace IPA characters with normal letters
            'ɛ': 'e',
            'ə': 'e',
            'ɪ': 'i',
            'ʊ': 'u',
            'ɔ': 'o',
            'ɑ': 'a',
            'æ': 'a',
            'ʌ': 'a',
            'ɒ': 'o',
            'ʃ': 'sch',
            'ç': 'ch',
            'ŋ': 'ng',
            'ɾ': 'r',
            'ʔ': '',
        }
        
        # Perform all replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Keep only allowed characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-\'() ')
        text = ''.join(c for c in text if c in allowed_chars)
        
        return text.strip()

    def preprocess_text(self, text: str) -> str:
        """Prepares text for TTS."""
        text = self.clean_text_for_tts(text)
        
        # Additional preprocessing if needed
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        logger.debug(f"Preprocessed text: {text}")
        return text

    def split_text_into_chunks(self, text: str, max_chars: int = 200) -> List[str]:
        """
        Splits the text into chunks of maximum length.

        Args:
            text: Input text
            max_chars: Maximum chunk length (default: 200)

        Returns:
            List of text chunks
        """
        text = self.preprocess_text(text)
        sentences = re.split('([.!?]+)', text)
        chunks = []
        current_chunk = ""
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + sentences[i + 1]
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.debug(f"Text split into {len(chunks)} chunks.")
        return chunks

    def generate_speech(self, text: str, speaker: Optional[str] = None, gender: str = "male") -> Path:
        """Generates speech output with TTS."""
        try:
            text = self.preprocess_text(text)
            output_path = self.file_manager.get_temp_path("tts_audio", ".wav")
            self.file_manager.ensure_path_exists(output_path)
            
            logger.info(f"Generating speech at: {output_path}")

            if self.model_type == "tacotron2":
                self._generate_with_tacotron2(text, output_path, speaker)
            else:
                self._generate_with_bark(text, output_path, speaker or self.get_next_voice(gender))

            if not output_path.exists():
                raise FileNotFoundError(f"Audio file not generated: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error during speech generation: {str(e)}")
            raise
        finally:
            self.file_manager.cleanup_temp_files()

    def _generate_with_tacotron2(self, text: str, output_path: Path, speaker: Optional[str]) -> None:
        """Generates audio with Tacotron2."""
        logger.info("Generating with Tacotron2...")
        self.model.tts_to_file(text=text, file_path=str(output_path), speaker=speaker)

    def _generate_with_bark(self, text: str, output_path: Path, speaker_preset: str) -> None:
        """Generates audio with Bark."""
        logger.info("Generating with Bark...")
        text_chunks = self.split_text_into_chunks(text)
        combined_audio = AudioSegment.empty()

        for i, chunk in enumerate(text_chunks, 1):
            chunk_path = self.file_manager.get_temp_path(f"chunk_{i}", ".wav")
            self.file_manager.ensure_path_exists(chunk_path)
            
            try:
                # Generate audio
                audio_array = self._generate_chunk_audio(chunk, speaker_preset)
                
                # Save as WAV
                wavfile.write(str(chunk_path), SAMPLE_RATE, audio_array)
                
                # Add to combined audio
                chunk_audio = AudioSegment.from_wav(str(chunk_path))
                combined_audio += chunk_audio
                
            except Exception as e:
                logger.error(f"Error in chunk {i}: {str(e)}")
                raise
            finally:
                if chunk_path.exists():
                    chunk_path.unlink()

        # Export final audio
        logger.info("Exporting combined audio...")
        combined_audio.export(str(output_path), format="wav")

    def _generate_chunk_audio(self, text: str, speaker_preset: str) -> np.ndarray:
        """Generates audio array for a chunk."""
        with torch.cuda.amp.autocast(enabled=self.use_gpu):
            audio_array = generate_audio(
                text=text,
                history_prompt=speaker_preset,
                text_temp=0.7,
                waveform_temp=0.7,
            )

        # Convert to float32 and normalize
        audio_array = audio_array.astype(np.float32)
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        return audio_array


@dataclass
class SyncConfig:
    """
    Configuration for synchronizing audio with video.

    Attributes:
        min_pause_duration: Minimum pause duration in seconds
        max_pause_duration: Maximum pause duration in seconds
        max_speed_up: Maximum speed-up factor
        min_speed_down: Minimum speed-down factor
        segment_margin: Segment margin in seconds
        crossfade_duration: Audio crossfade duration in seconds
    """

    min_pause_duration: float = 0.3
    max_pause_duration: float = 2.0
    max_speed_up: float = 1.5
    min_speed_down: float = 0.7
    segment_margin: float = 0.2
    crossfade_duration: float = 0.3


class AudioSegmentInfo:
    """
    Represents an audio segment.

    Attributes:
        start: Start time in milliseconds
        end: End time in milliseconds
        text: Text associated with the segment
        duration: Segment duration in seconds
    """

    def __init__(self, start: int, end: int, text: str = ""):
        """
        Initializes the AudioSegmentInfo instance.

        Args:
            start: Start time in milliseconds
            end: End time in milliseconds
            text: Text associated with the segment (optional)
        """
        self.start = start
        self.end = end
        self.text = text
        self.duration = (end - start) / 1000
        logger.debug(
            f"AudioSegmentInfo created: start={self.start}, end={self.end}, duration={self.duration}s"
        )


class Synchronizer:
    """
    Synchronizes audio with video.

    Attributes:
        file_manager: FileManager instance for managing files
        config: SyncConfig instance for synchronization configuration
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """
        Initializes the Synchronizer instance.

        Args:
            config: SyncConfig instance for synchronization configuration (optional)
        """
        self.file_manager = FileManager()
        self.config = config or SyncConfig()
        self.MIN_SPEED = 0.5
        self.MAX_SPEED = 2.0
        self.FRAME_BLEND_THRESHOLD = 0.8  # Speed threshold for frame blending

    def cleanup_temp_files(self):
        """
        Cleans up temporary files created during synchronization.
        """
        try:
            logger.info("Cleaning up temporary files...")
            # Clean up temp files
            self.file_manager.cleanup_temp_files()
            # Clean up old outputs, keeping last 10
            self.file_manager.cleanup_old_outputs(max_files=10)
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def detect_scene_changes(self, video: VideoFileClip, threshold: float = 30.0) -> List[float]:
        """
        Detects major scene changes in the video.
        
        Args:
            video: VideoFileClip instance
            threshold: Difference threshold for scene detection
            
        Returns:
            List of timestamps where scenes change
        """
        logger.info("Detecting scene changes...")
        scenes = [0.0]  # Start with beginning of video
        
        # Sample frames at 1fps for efficiency
        frame_times = np.arange(0, video.duration, 1.0)
        prev_frame = None
        
        for t in tqdm(frame_times, desc="Analyzing frames"):
            frame = video.get_frame(t)
            if prev_frame is not None:
                # Calculate mean absolute difference between frames
                diff = np.mean(np.abs(frame - prev_frame))
                if diff > threshold:
                    scenes.append(t)
            prev_frame = frame
        
        scenes.append(video.duration)
        logger.info(f"Detected {len(scenes)-2} scene changes")
        return scenes

    def find_optimal_segments(
        self,
        video_duration: float,
        audio_duration: float,
        scene_changes: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Creates optimal segments for synchronization based on scene changes.
        
        Args:
            video_duration: Duration of the video in seconds
            audio_duration: Duration of the audio in seconds
            scene_changes: List of scene change timestamps
            
        Returns:
            List of segment dictionaries
        """
        logger.info("Creating optimal synchronization segments...")
        segments = []
        global_speed_factor = audio_duration / video_duration
        
        for i in range(len(scene_changes) - 1):
            start_time = scene_changes[i]
            end_time = scene_changes[i + 1]
            segment_duration = end_time - start_time
            
            # Calculate proportional audio segment
            audio_start = start_time * (audio_duration / video_duration)
            audio_end = audio_start + (segment_duration * global_speed_factor)
            
            # Calculate local speed factor with bounds
            speed_factor = (audio_end - audio_start) / segment_duration
            speed_factor = min(self.MAX_SPEED, max(self.MIN_SPEED, speed_factor))
            
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "speed_factor": speed_factor
            })
        
        return segments

    def apply_frame_interpolation(self, clip: VideoFileClip, speed_factor: float) -> VideoFileClip:
        """
        Applies frame interpolation for smoother speed changes.
        
        Args:
            clip: Video clip to process
            speed_factor: Speed adjustment factor
            
        Returns:
            Processed video clip
        """
        if speed_factor > self.FRAME_BLEND_THRESHOLD:
            # For significant speedup, use frame blending
            logger.info(f"Applying frame blending for speed factor {speed_factor:.2f}")
            return clip.fx(speedx, speed_factor)
        else:
            # For slower speeds, use frame interpolation
            logger.info(f"Applying frame interpolation for speed factor {speed_factor:.2f}")
            return clip.fx(speedx, speed_factor)

    def sync_audio_with_video(
        self, video_path: str, audio_path: str, 
        segments: Optional[List[AudioSegmentInfo]] = None
    ) -> str:
        """Synchronizes audio with video."""
        global _LAST_DOWNLOADED_VIDEO_PATH
        try:
            # Convert to Path objects
            video_path = Path(video_path)
            audio_path = Path(audio_path)
            
            # Check existence with fallback
            if not video_path.exists() and _LAST_DOWNLOADED_VIDEO_PATH:
                logger.warning(f"Video not found at {video_path}, using fallback path: {_LAST_DOWNLOADED_VIDEO_PATH}")
                video_path = Path(_LAST_DOWNLOADED_VIDEO_PATH)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
                
            logger.info("Loading video and audio files...")
            
            # Process video and audio
            video = VideoFileClip(str(video_path))
            audio = AudioFileClip(str(audio_path))
            
            # Ensure audio duration matches video duration exactly
            video_duration = video.duration
            audio_duration = audio.duration
            
            logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
            
            # Calculate duration difference
            duration_diff = abs(video_duration - audio_duration)
            DURATION_TOLERANCE = 10.0  # 10 seconds tolerance
            
            if duration_diff > DURATION_TOLERANCE:
                if audio_duration > video_duration:
                    # Trim audio if longer than video
                    logger.info("Trimming audio to match video duration (difference > 10s)")
                    audio = audio.subclip(0, video_duration)
                elif audio_duration < video_duration:
                    # Extend audio if shorter than video by adding silence
                    logger.info("Extending audio to match video duration (difference > 10s)")
                    from moviepy.audio.AudioClip import AudioClip
                    silence_duration = video_duration - audio_duration
                    silence = AudioClip(lambda t: 0, duration=silence_duration)
                    audio = CompositeAudioClip([audio, silence.set_start(audio_duration)])
            else:
                logger.info(f"Duration difference ({duration_diff:.2f}s) within tolerance, keeping original audio")
            
            # Set audio to video
            final_video = video.set_audio(audio)
            
            # Save result with optimized CPU usage
            output_path = self.file_manager.get_output_path("synced_video", ".mp4")
            logger.info(f"Saving synchronized video: {output_path}")
            
            # Use ffmpeg with optimized CPU usage
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.file_manager.get_temp_path("temp_audio", ".m4a")),
                remove_temp=True,
                ffmpeg_params=[
                    '-preset', 'fast',
                    '-crf', '23',  # Quality
                    '-threads', '8'  # Number of threads
                ],
                logger=None  # Suppress ffmpeg logs
            )
            
            # Cleanup
            video.close()
            audio.close()
            final_video.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error during synchronization: {str(e)}")
            raise
        finally:
            self.cleanup_temp_files()


_LAST_DOWNLOADED_VIDEO_PATH = None

def download_video(video_url: str, file_manager: FileManager) -> str:
    """
    Downloads a video from the specified URL using yt-dlp.

    Args:
        video_url: Video URL
        file_manager: FileManager instance

    Returns:
        Path to the downloaded video file
    """
    global _LAST_DOWNLOADED_VIDEO_PATH
    try:
        logger.info(f"Downloading video: {video_url}")
        
        # Create temporary directory for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = file_manager.get_temp_path(f"video_{timestamp}", "")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download video
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'downloaded_video')
        
        # Find the downloaded file
        downloaded_files = list(temp_dir.glob('*.mp4'))
        if not downloaded_files:
            raise FileNotFoundError(f"Video was not downloaded in: {temp_dir}")
            
        output_path = downloaded_files[0]
        _LAST_DOWNLOADED_VIDEO_PATH = str(output_path)  # Save path as fallback
        logger.info(f"Video downloaded to: {output_path}")
        return _LAST_DOWNLOADED_VIDEO_PATH
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        raise


def extract_audio(video_path: str, file_manager: FileManager) -> str:
    """
    Extracts audio from the specified video file.

    Args:
        video_path: Path to the video file
        file_manager: FileManager instance

    Returns:
        Path to the extracted audio file
    """
    audio_output = str(file_manager.get_temp_path("audio", ".wav"))
    try:
        logger.info("Extracting audio from video...")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-y",
            audio_output,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if not Path(audio_output).exists() or Path(audio_output).stat().st_size == 0:
            raise Exception("Audio extraction failed: Output file is empty or doesn't exist.")
        logger.info(f"Audio extracted to: {audio_output}")
        return audio_output
    except subprocess.CalledProcessError:
        logger.error("FFmpeg failed to extract audio.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while extracting audio: {str(e)}")
        raise


def transcribe_audio(
    audio_path: str,
    api_key: str,
    language_code: str = "en",
    speakers_expected: Optional[int] = None,
    file_manager: Optional[FileManager] = None,
) -> List[Utterance]:
    """
    Transcribes audio using AssemblyAI API with speaker diarization.

    Args:
        audio_path: Path to the audio file
        api_key: AssemblyAI API key
        language_code: Language code for transcription (default: "en")
        speakers_expected: Number of speakers expected (improves accuracy)
        file_manager: FileManager instance for managing files

    Returns:
        List of Utterance instances
    """
    if file_manager is None:
        file_manager = FileManager()

    try:
        # Convert audio to MP3 if needed
        if not audio_path.lower().endswith(".mp3"):
            logger.info("Converting audio to MP3 format for transcription...")
            audio_path = convert_audio_to_mp3(audio_path, file_manager)

        logger.info("Uploading audio file to AssemblyAI...")
        upload_url = upload_audio(audio_path, api_key)

        headers = {
            "authorization": api_key,
            "content-type": "application/json",
        }

        # Configure transcription options
        data = {
            "audio_url": upload_url,
            "language_code": language_code,
        }

        if speakers_expected and speakers_expected > 1:
            data["speaker_labels"] = True
            data["speakers_expected"] = speakers_expected
            logger.info("Enabling speaker diarization...")
        else:
            logger.info("Single speaker detected or speaker count not specified. Skipping diarization.")

        # Start transcription
        response = requests.post(
            "https://api.assemblyai.com/v2/transcript", json=data, headers=headers
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]
        logger.info(f"Transcription started. ID: {transcript_id}")

        # Poll for transcription completion
        while True:
            time.sleep(5)
            status_response = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers,
            )
            status_response.raise_for_status()
            transcription = status_response.json()
            status = transcription["status"]

            if status == "completed":
                logger.info("Transcription completed.")
                if (
                    data.get("speaker_labels")
                    and "utterances" in transcription
                    and transcription["utterances"]
                ):
                    # Multiple speakers: use diarization results
                    speaker_config = get_speaker_config(transcription["utterances"])

                    utterances = []
                    for utterance in transcription["utterances"]:
                        speaker_gender = (
                            "female"
                            if speaker_config["speaker_info"][utterance["speaker"]][
                                "is_female"
                            ]
                            else "male"
                        )
                        utterances.append(
                            Utterance(
                                speaker=utterance["speaker"],
                                text=utterance["text"],
                                start=utterance["start"],
                                end=utterance["end"],
                                confidence=utterance.get("confidence", 0.0),
                                words=utterance.get("words", []),
                                gender=speaker_gender,
                            )
                        )
                    return utterances
                else:
                    # Single speaker: create one utterance
                    # For non-interactive mode, default to male. Modify as needed.
                    utterances = [
                        Utterance(
                            speaker="A",
                            text=transcription["text"],
                            start=0,
                            end=int(float(transcription.get("audio_duration", 0)) * 1000),
                            confidence=1.0,
                            words=[],
                            gender="male",
                        )
                    ]
                    return utterances
            elif status == "error":
                error_msg = transcription.get("error", "Unknown error during transcription.")
                logger.error(f"Transcription failed: {error_msg}")
                raise TranscriptionError(f"Transcription failed: {error_msg}")
            else:
                logger.info("Transcription in progress...")

    except requests.HTTPError as e:
        logger.error(f"HTTP error during transcription: {str(e)}")
        raise TranscriptionError(f"HTTP error during transcription: {str(e)}")
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise TranscriptionError(f"Transcription failed: {str(e)}")
    finally:
        # Clean up temporary files
        file_manager.cleanup_temp_files()


def translate_text(text: str, target_lang: str = "de") -> str:
    """
    Translates the specified text using the Google Translate API.

    Args:
        text: Input text
        target_lang: Target language (default: "de")

    Returns:
        Translated text
    """
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        chunks = chunk_text(text)
        logger.info(f"Translating text in {len(chunks)} chunks.")
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Translating chunk {i}/{len(chunks)}...")
            translated = translator.translate(chunk)
            translated_chunks.append(translated)
        translated_text = " ".join(translated_chunks)
        logger.info("Translation completed.")
        return translated_text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise Exception(f"Translation failed: {str(e)}")


def chunk_text(text: str, max_length: int = 4500) -> List[str]:
    """
    Splits the specified text into chunks of maximum length.

    Args:
        text: Input text
        max_length: Maximum chunk length (default: 4500)

    Returns:
        List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(sentence)
        if current_length + sentence_length + 1 <= max_length:
            current_chunk += sentence
            current_length += sentence_length + 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
    if current_chunk:
        chunks.append(current_chunk.strip())
    logger.debug(f"Text split into {len(chunks)} chunks.")
    return chunks


def convert_audio_to_mp3(input_path: str, file_manager: FileManager) -> str:
    """
    Converts the specified audio file to MP3 format.

    Args:
        input_path: Path to the input audio file
        file_manager: FileManager instance

    Returns:
        Path to the converted MP3 file
    """
    output_path = str(file_manager.get_temp_path("converted_audio", ".mp3"))
    try:
        logger.info("Converting audio to MP3 format...")
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-acodec",
            "libmp3lame",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-y",
            output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
            raise TranscriptionError("Audio conversion failed: Output file is empty or doesn't exist.")
        logger.info(f"Audio converted to MP3 at: {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        logger.error("FFmpeg failed to convert audio to MP3.")
        raise TranscriptionError("FFmpeg failed to convert audio to MP3.")
    except Exception as e:
        logger.error(f"Failed to convert audio to MP3: {str(e)}")
        raise TranscriptionError(f"Failed to convert audio to MP3: {str(e)}")


def upload_audio(audio_path: str, api_key: str) -> str:
    """
    Uploads the specified audio file to the AssemblyAI API.

    Args:
        audio_path: Path to the audio file
        api_key: AssemblyAI API key

    Returns:
        Upload URL
    """
    try:
        logger.info(f"Uploading audio file: {audio_path} ({os.path.getsize(audio_path)} bytes)")
        headers = {"authorization": api_key}

        with open(audio_path, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, data=f
            )
        response.raise_for_status()
        upload_url = response.json()["upload_url"]
        logger.info(f"Audio uploaded successfully. URL: {upload_url}")
        return upload_url
    except requests.HTTPError as e:
        logger.error(f"HTTP error during audio upload: {str(e)}")
        raise TranscriptionError(f"HTTP error during audio upload: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to upload audio: {str(e)}")
        raise TranscriptionError(f"Failed to upload audio: {str(e)}")


def get_speaker_config(utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Configures speaker settings based on utterances.

    Args:
        utterances: List of utterance dictionaries

    Returns:
        Speaker configuration dictionary
    """
    speaker_samples = {}
    for utterance in utterances:
        speaker = utterance["speaker"]
        if speaker not in speaker_samples:
            speaker_samples[speaker] = utterance["text"][:200]
    speaker_count = len(speaker_samples)
    logger.info(f"Detected {speaker_count} speakers in the audio.")

    speaker_info = {}
    for speaker, sample in speaker_samples.items():
        # In non-interactive mode, default to male. Modify as needed.
        speaker_info[speaker] = {"is_female": False, "order": int(speaker.replace("A", "1").replace("B", "2").replace("C", "3").replace("D", "4"))}

    return {"speaker_count": speaker_count, "speaker_info": speaker_info}


def ensure_downloads_dir():
    """
    Ensures the downloads directory exists.
    """
    Path("downloads").mkdir(exist_ok=True)
    logger.debug("Ensured 'downloads' directory exists.")


def get_tts_model_choice() -> str:
    """
    Prompts the user to select a TTS model.

    Returns:
        Selected TTS model
    """
    while True:
        print("\nSelect TTS model:")
        print("1. Tacotron2 (Thorsten German voice)")
        print("2. Bark (More expressive, potentially slower)")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            return "tacotron2"
        elif choice == "2":
            return "bark"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_gpu_choice() -> bool:
    """
    Prompts the user to select whether to use the GPU.

    Returns:
        Whether to use the GPU
    """
    while True:
        print("\nUse GPU for processing?")
        print("1. Yes (Recommended if available)")
        print("2. No (CPU only)")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_language_choice() -> str:
    """
    Prompts the user to select the source video language.

    Returns:
        Selected language code
    """
    while True:
        print("\nSelect source video language:")
        print("1. English")
        print("2. Spanish")
        print("3. French")
        print("4. Italian")
        choice = input("Enter your choice (1-4): ").strip()

        language_codes = {
            "1": "en",
            "2": "es",
            "3": "fr",
            "4": "it",
        }

        if choice in language_codes:
            return language_codes[choice]
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main():
    """
    Main function for the video translation and synchronization pipeline.
    """
    parser = argparse.ArgumentParser(
        description="YouTube Video Translator: Translate and dub YouTube videos."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="YouTube video URL to translate.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="AssemblyAI API key.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "es", "fr", "it"],
        help="Source language code (en, es, fr, it).",
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        choices=["tacotron2", "bark"],
        help="TTS model to use (tacotron2 or bark).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for processing if available.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )
    args = parser.parse_args()

    # Set logging level based on verbosity
    logger.setLevel(getattr(logging, args.verbosity.upper(), logging.INFO))

    try:
        logger.info("=== YouTube Video Translator ===")

        # Ensure downloads directory exists
        ensure_downloads_dir()
        file_manager = FileManager()

        # Get video URL
        video_url = args.url or input("\nEnter YouTube video URL: ").strip()
        if not video_url:
            logger.error("No video URL provided.")
            return

        # Get AssemblyAI API key
        api_key = args.api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.error(
                "AssemblyAI API key not found! Please set the ASSEMBLYAI_API_KEY environment variable or provide it via --api_key."
            )
            return

        # Get source language
        source_language = args.language or get_language_choice()

        # Get GPU usage preference
        use_gpu = args.use_gpu or get_gpu_choice()

        # Get TTS model choice
        tts_model = args.tts_model or get_tts_model_choice()

        logger.info("\nStarting video translation process...\n")

        # Step 1: Download video
        logger.info("Step 1: Downloading video...")
        video_path = download_video(video_url, file_manager)
        logger.info(f"Video downloaded to: {video_path}\n")

        # Step 2: Extract audio
        logger.info("Step 2: Extracting audio...")
        audio_path = extract_audio(video_path, file_manager)
        logger.info(f"Audio extracted to: {audio_path}\n")

        # Step 3: Transcribe audio
        logger.info("Step 3: Transcribing audio...")
        utterances = transcribe_audio(
            audio_path=audio_path,
            api_key=api_key,
            language_code=source_language,
            speakers_expected=None,  # Set to desired number if known
            file_manager=file_manager,
        )
        transcribed_text = " ".join(utterance.text for utterance in utterances)
        logger.info(
            f"Transcription completed. Text preview: {transcribed_text[:200]}..."
            if len(transcribed_text) > 200
            else f"Transcription completed. Text: {transcribed_text}"
        )

        # Step 4: Translate text
        logger.info("Step 4: Translating text...")
        translated_text = translate_text(transcribed_text, target_lang="de")
        logger.info("Translation completed.\n")

        # Step 5: Generate German speech
        logger.info("Step 5: Generating German speech...")
        tts_generator = TTSGenerator(model_type=tts_model, use_gpu=use_gpu)
        tts_audio_path = tts_generator.generate_speech(translated_text)
        logger.info(f"German speech generated at: {tts_audio_path}\n")

        # Step 6: Synchronize audio with video
        logger.info("Step 6: Synchronizing audio with video...")
        synchronizer = Synchronizer()
        final_video_path = synchronizer.sync_audio_with_video(
            video_path=video_path, audio_path=tts_audio_path
        )
        logger.info(f"\nDone! Final video saved to: {final_video_path}")

    except Exception as e:
        logger.error(f"\nAn error occurred: {str(e)}")
    finally:
        # Optional: Cleanup temporary files if needed
        file_manager.cleanup_temp_files()
        file_manager.cleanup_old_outputs()


if __name__ == "__main__":
    main()
