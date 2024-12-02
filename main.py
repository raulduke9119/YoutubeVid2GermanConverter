#!/usr/bin/env python3

"""
Main script for the YouTube video translation app.
"""
import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from modules.transcriber import transcribe_audio
from modules.translator import translate_text
from modules.tts_generator import TTSGenerator
from modules.synchronizer import Synchronizer
from modules.input_handler import InputHandler
from modules.video_downloader import download_video

def ensure_downloads_dir():
    """Ensure downloads directory exists."""
    Path("downloads").mkdir(parents=True, exist_ok=True)

def main():
    """Main function."""
    try:
        input_handler = InputHandler()
        
        # 1. Get YouTube URL and API key
        url = input_handler.get_youtube_url()
        if not url:
            print("No URL provided. Exiting.")
            return
            
        api_key = input_handler.get_api_key()
        if not api_key:
            print("No API key provided. Exiting.")
            return
            
        # Create downloads directory
        ensure_downloads_dir()
        
        # 2. Download video and extract audio
        print("\n1. Downloading video...")
        video_path = download_video(url)
        print(f"Video downloaded to: {video_path}")
        
        print("\n2. Extracting audio...")
        video_clip = VideoFileClip(video_path)
        audio_path = "downloads/extracted_audio.wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Audio extracted to: {audio_path}\n")
        
        # 3. Transcribe audio
        print("\n3. Transcribing audio...")
        utterances = transcribe_audio(
            audio_path=audio_path,
            api_key=api_key,
            language_code="en"
        )
        
        # Combine all utterances into a single text
        transcribed_text = " ".join(utterance.text for utterance in utterances)
        print("\nTranscription completed. Text:", transcribed_text[:200] + "..." if len(transcribed_text) > 200 else transcribed_text)
        
        # 4. Translate text
        print("\n4. Translating text to German...")
        translated_text = translate_text(transcribed_text)
        print("Translation completed\n")
        
        # 5. Get TTS model choice
        print("\n5. Choose TTS model:")
        print("1. Tacotron2 (Thorsten German voice) - Faster, more stable")
        print("2. Bark - More expressive but slower")
        model_choice = input("Enter choice (1 or 2): ").strip()
        tts_model = "tacotron2" if model_choice != "2" else "bark"
        
        # 6. Generate speech
        print("\n5. Generating German speech...")
        tts = TTSGenerator(model_type=tts_model, use_gpu=True)
        tts_audio_path = tts.generate_speech(translated_text)
        
        print("\n6. Synchronizing audio with video...")
        synchronizer = Synchronizer()
        final_video_path = synchronizer.sync_audio_with_video(
            video_path=video_path,
            audio_path=tts_audio_path
        )
        
        print(f"\nDone! Final video saved to: {final_video_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
