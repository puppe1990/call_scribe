#!/usr/bin/env python3
"""
Audio Transcription Script using OpenAI Whisper
Transcribes audio files to text with support for multiple languages and models.
"""

import os
import sys
import argparse
import whisper
from pathlib import Path


def transcribe_audio(
    audio_file,
    model_size="base",
    language=None,
    output_file=None,
    verbose=False
):
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_file: Path to the audio file to transcribe
        model_size: Whisper model size (tiny, base, small, medium, large, turbo)
        language: Language code (e.g., 'pt', 'en', 'es'). If None, auto-detect
        output_file: Path to save transcription. If None, saves next to audio file
        verbose: Print detailed progress information
    
    Returns:
        str: The transcribed text
    """
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Load Whisper model
    print(f"📥 Loading Whisper model '{model_size}'...")
    try:
        model = whisper.load_model(model_size)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe audio
    print(f"\n🎯 Transcribing audio: {audio_file}")
    print("⏳ This may take a while depending on audio length and model size...")
    
    try:
        transcribe_options = {
            "fp16": False,
            "verbose": verbose
        }
        
        if language:
            transcribe_options["language"] = language
            print(f"🌐 Language: {language}")
        else:
            print("🌐 Language: Auto-detecting...")
        
        result = model.transcribe(audio_file, **transcribe_options)
        
        transcription = result["text"].strip()
        
        # Detect language if not specified
        detected_language = result.get("language", "unknown")
        if not language:
            print(f"🌐 Detected language: {detected_language}")
        
        # Determine output file path
        if output_file is None:
            audio_path = Path(audio_file)
            output_file = audio_path.with_suffix('.txt')
        else:
            output_file = Path(output_file)
        
        # Save transcription
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Print results
        print(f"\n📝 Transcription:")
        print("-" * 60)
        print(transcription)
        print("-" * 60)
        print(f"\n✅ Transcription saved to: {output_file}")
        
        # Print segments if verbose
        if verbose and "segments" in result:
            print(f"\n📊 Segments: {len(result['segments'])}")
        
        return transcription
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        sys.exit(1)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription (auto-detect language)
  python transcribe.py audio.wav
  
  # Specify language
  python transcribe.py audio.wav --language pt
  
  # Use larger model for better accuracy
  python transcribe.py audio.wav --model large
  
  # Save to specific file
  python transcribe.py audio.wav --output transcript.txt
  
  # Verbose output
  python transcribe.py audio.wav --verbose
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: base). "
             "Larger models are more accurate but slower."
    )
    
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., 'pt', 'en', 'es'). "
             "If not specified, language will be auto-detected."
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path for transcription. "
             "If not specified, saves as <audio_file>.txt"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Run transcription
    transcribe_audio(
        audio_file=args.audio_file,
        model_size=args.model,
        language=args.language,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Transcription interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

