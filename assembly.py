# Install the assemblyai package by executing the command "pip install assemblyai"

import assemblyai as aai
import sys
import time
import os
from pathlib import Path

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get API key from environment variable
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    raise ValueError(
        "ASSEMBLYAI_API_KEY environment variable is not set. "
        "Please set it in your environment or create a .env file with ASSEMBLYAI_API_KEY=your_key"
    )

aai.settings.api_key = api_key

# audio_file = "./local_file.mp3"
audio_file = "/Users/matheuspuppe/Desktop/Projects/github/call_scribe/gravacoes/retro-dotted-17-11-25.wav"

print("🎙️  Starting transcription...")
print(f"📁 Audio file: {os.path.basename(audio_file)}")

config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.universal,
    language_code="pt"
)

print("⏳ Transcribing audio (this may take a while)...")
transcript = aai.Transcriber(config=config).transcribe(audio_file)

# Show loading indicator while waiting for completion
spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
i = 0
while transcript.status == "queued" or transcript.status == "processing":
    sys.stdout.write(f"\r🔄 Processing {spinner[i % len(spinner)]}")
    sys.stdout.flush()
    time.sleep(0.1)
    transcript = aai.Transcriber().get_transcript(transcript.id)
    i += 1

print("\r✅ Transcription completed!")

if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")

# Save transcript to txt file
output_file = Path(audio_file).stem + "_transcript.txt"
output_path = os.path.join(os.path.dirname(audio_file), output_file)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(transcript.text)

print(f"💾 Transcript saved to: {output_path}")
print(f"\n📝 Transcript preview:\n{transcript.text[:500]}...")