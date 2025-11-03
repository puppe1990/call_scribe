# Call Scribe 🎙️

A Python tool for recording audio from your microphone and automatically transcribing it using OpenAI's open-source Whisper model. Perfect for meetings, interviews, call notes, and any situation where you need speech-to-text transcription.

## Features

- 🎤 **Real-time Audio Recording** - Record audio directly from your microphone
- 🗣️ **AI-Powered Transcription** - Uses OpenAI's Whisper model for accurate speech-to-text conversion
- 🌐 **Multi-language Support** - Defaults to Portuguese (Brazil), with support for 99+ languages
- 📁 **Organized File Storage** - Each recording is saved in its own timestamped folder
- 💾 **Local Processing** - All processing happens on your machine (no API calls, no costs)
- ⚡ **Fast & Efficient** - Uses the optimized "turbo" model by default

## Requirements

- Python 3.8 or higher
- ffmpeg (for audio processing)
- Microphone access

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/call_scribe.git
cd call_scribe
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Install PyAudio (may require pipwin)
pip install pipwin
pipwin install pyaudio
```

## Usage

### Basic Usage

1. Run the script:
```bash
python init.py
```

2. Use the following commands:
   - `start` - Begin recording
   - `stop` - Stop recording and transcribe
   - `language` - Change transcription language
   - `quit` - Exit the program

### Example Session

```
🎙️  CALL RECORDING & TRANSCRIPTION TOOL
============================================================
This tool records your microphone and creates transcripts
Use for: meetings, interviews, call notes, etc.
============================================================

📥 Loading Whisper model 'turbo'...
   (This may take a moment on first run as the model downloads)
✅ Model loaded successfully!
🌐 Language: Portuguese (Brazil)

Commands:
  'start'    - Begin recording
  'stop'     - Stop recording and transcribe
  'language' - Change transcription language
  'quit'     - Exit program

Enter command: start

============================================================
🔴 RECORDING IN PROGRESS
============================================================
📁 Recording folder: audio/call_20251103_143148
📁 Audio file: audio/call_20251103_143148/call_recording_20251103_143148.wav
📝 Transcript file: audio/call_20251103_143148/call_transcript_20251103_143148.txt
⏸️  Press Ctrl+C or type 'stop' to end recording
============================================================

Enter command: stop

⏹️  Stopping recording...
💾 Audio saved successfully
🎯 Transcribing audio...
📊 Processing audio with Whisper...
🗣️  Converting speech to text (Portuguese (Brazil))...
✅ Transcription complete!
```

## File Organization

Recordings are automatically organized in timestamped folders:

```
audio/
  call_20251103_143148/
    call_recording_20251103_143148.wav
    call_transcript_20251103_143148.txt
  call_20251103_150000/
    call_recording_20251103_150000.wav
    call_transcript_20251103_150000.txt
```

## Language Support

The tool defaults to **Portuguese (Brazil)** but supports 99+ languages. To change the language:

1. Type `language` when prompted
2. Enter the language code (e.g., `en` for English, `es` for Spanish)
3. Type `list` to see all available languages

### Common Language Codes

- `pt` - Portuguese (Brazil) - **Default**
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `ja` - Japanese
- `zh` - Chinese
- `ko` - Korean
- `ru` - Russian

## Whisper Models

The tool uses the **"turbo"** model by default, which offers a good balance between speed and accuracy. You can change the model when initializing `CallRecorder`:

```python
recorder = CallRecorder(model_name="large")  # More accurate, slower
```

Available models:
- `tiny` - Fastest, least accurate (~1 GB VRAM)
- `base` - Fast, less accurate (~1 GB VRAM)
- `small` - Balanced (~2 GB VRAM)
- `medium` - More accurate (~5 GB VRAM)
- `large` - Most accurate (~10 GB VRAM)
- `turbo` - Optimized for speed (~6 GB VRAM) - **Default**

## Project Structure

```
call_scribe/
├── init.py              # Main application file
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── audio/              # Recordings folder (created automatically)
    └── call_*/         # Individual recording folders
        ├── *.wav       # Audio files
        └── *.txt       # Transcript files
```

## Features in Detail

### Privacy-First
- All processing happens locally on your machine
- No data is sent to external APIs
- Your recordings stay private

### Cost-Free
- Uses open-source Whisper model
- No API keys required
- No usage limits or costs

### Easy to Use
- Simple command-line interface
- Clear visual indicators
- Automatic file organization

## Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### ffmpeg Not Found

Make sure ffmpeg is installed and available in your PATH:
```bash
ffmpeg -version
```

### Model Download Issues

On first run, Whisper will download the model (~800MB for turbo). Make sure you have:
- Internet connection
- Sufficient disk space
- Write permissions in the cache directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) - Audio I/O library

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Made with ❤️ for better call documentation

