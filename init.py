"""
Call Recording and Transcription Tool
Records audio from microphone and saves transcription to a file.
Shows clear visual indicators when recording is active.
"""

import pyaudio
import wave
import whisper
from datetime import datetime
import os
import threading
import time
import sys

class CallRecorder:
    def __init__(self, model_name="turbo", language="pt"):
        self.is_recording = False
        self.audio_filename = None
        self.transcript_filename = None
        self.frames = []
        self.recording_folder = None
        self.start_time = None
        self.timer_thread = None
        self.loader_active = False
        self.loader_thread = None
        self.call_title = None
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Language setting (default: Portuguese)
        self.language = language
        
        # Create audio folder if it doesn't exist
        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Initialize Whisper model
        print(f"📥 Loading Whisper model '{model_name}'...")
        print("   (This may take a moment on first run as the model downloads)")
        self._start_loader("Loading Whisper model...")
        self.model = whisper.load_model(model_name)
        self._stop_loader()
        print(f"✅ Model loaded successfully!")
        print(f"🌐 Language: {self._get_language_name(self.language)}")
        
    def start_recording(self, title=None):
        """Start recording audio from microphone"""
        if self.is_recording:
            print("⚠️  Already recording!")
            return
        
        # Store call title
        self.call_title = title.strip() if title and title.strip() else None
        
        # Create unique folder for this recording with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use title in folder name if provided, otherwise just use timestamp
        if self.call_title:
            # Sanitize title for folder name (remove invalid characters)
            safe_title = "".join(c for c in self.call_title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limit length and replace spaces
            folder_name = f"call_{timestamp}_{safe_title}"
        else:
            folder_name = f"call_{timestamp}"
        
        self.recording_folder = os.path.join(self.audio_folder, folder_name)
        os.makedirs(self.recording_folder, exist_ok=True)
        
        # Create filenames with timestamp inside the unique folder
        self.audio_filename = os.path.join(self.recording_folder, f"call_recording_{timestamp}.wav")
        self.transcript_filename = os.path.join(self.recording_folder, f"call_transcript_{timestamp}.txt")
        
        self.frames = []
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("\n" + "="*60)
        print("🔴 RECORDING IN PROGRESS")
        print("="*60)
        if self.call_title:
            print(f"📝 Call title: {self.call_title}")
        print(f"📁 Recording folder: {self.recording_folder}")
        print(f"📁 Audio file: {self.audio_filename}")
        print(f"📝 Transcript file: {self.transcript_filename}")
        print("⏸️  Press Ctrl+C or type 'stop' to end recording")
        print("="*60)
        
        # Start timer
        self.start_time = time.time()
        self.timer_thread = threading.Thread(target=self._display_timer, daemon=True)
        self.timer_thread.start()
        
        # Print initial timer line
        print("⏱️  Recording time: 00:00")
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def _display_timer(self):
        """Display recording timer in real-time"""
        try:
            while self.is_recording:
                elapsed = time.time() - self.start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                
                if hours > 0:
                    timer_str = f"⏱️  Recording time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    timer_str = f"⏱️  Recording time: {minutes:02d}:{seconds:02d}"
                
                # Overwrite the previous timer line
                sys.stdout.write('\r' + timer_str + ' ' * 20)  # Clear remaining chars
                sys.stdout.flush()
                
                time.sleep(1)  # Update every second
        except Exception:
            pass  # Ignore errors when stopping
    
    def _record(self):
        """Internal method to record audio"""
        try:
            while self.is_recording:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"❌ Recording error: {e}")
            
    def stop_recording(self):
        """Stop recording and save audio file"""
        if not self.is_recording:
            print("⚠️  Not currently recording!")
            return
            
        # Clear timer line before stopping
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        
        print("\n⏹️  Stopping recording...")
        self.is_recording = False
        
        # Calculate final duration
        if self.start_time:
            duration = time.time() - self.start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            if hours > 0:
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = f"{minutes:02d}:{seconds:02d}"
            print(f"⏱️  Total recording time: {duration_str}")
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()
        
        # Wait for timer thread to finish
        if self.timer_thread:
            self.timer_thread.join(timeout=1)
        
        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Save audio file
        self._save_audio()
        
        # Transcribe audio
        self._transcribe_audio()
        
        print("\n" + "="*60)
        print("✅ RECORDING COMPLETE")
        print("="*60)
        if self.call_title:
            print(f"📝 Call title: {self.call_title}")
        print(f"📁 Recording folder: {self.recording_folder}")
        print(f"📁 Audio saved: {self.audio_filename}")
        print(f"📝 Transcript saved: {self.transcript_filename}")
        print("="*60 + "\n")
        
    def _show_loader(self, message):
        """Display a loading spinner"""
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        try:
            while self.loader_active:
                spinner = spinner_chars[i % len(spinner_chars)]
                sys.stdout.write(f'\r{spinner} {message}')
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
            # Clear the loader line
            sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')
            sys.stdout.flush()
        except Exception:
            pass
    
    def _start_loader(self, message):
        """Start a loading spinner in a separate thread"""
        self.loader_active = True
        self.loader_thread = threading.Thread(target=self._show_loader, args=(message,), daemon=True)
        self.loader_thread.start()
    
    def _stop_loader(self):
        """Stop the loading spinner"""
        self.loader_active = False
        if self.loader_thread:
            self.loader_thread.join(timeout=0.5)
    
    def _save_audio(self):
        """Save recorded audio to WAV file"""
        try:
            self._start_loader("💾 Saving audio file...")
            wf = wave.open(self.audio_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            self._stop_loader()
            print(f"💾 Audio saved successfully")
        except Exception as e:
            self._stop_loader()
            print(f"❌ Error saving audio: {e}")
            
    def _transcribe_audio(self):
        """Transcribe the recorded audio using open-source Whisper"""
        try:
            print(f"📊 Processing audio with Whisper...")
            print(f"🗣️  Converting speech to text ({self._get_language_name(self.language)})...")
            
            # Start loader for transcription
            self._start_loader("🎯 Transcribing audio (this may take a moment)...")
            
            # Transcribe using Whisper model with specified language
            result = self.model.transcribe(self.audio_filename, language=self.language)
            text = result["text"].strip()
            
            self._stop_loader()
            
            # Start loader for saving transcript
            self._start_loader("💾 Saving transcript...")
            
            # Save transcript to file
            with open(self.transcript_filename, 'w', encoding='utf-8') as f:
                f.write(f"Call Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.call_title:
                    f.write(f"Title: {self.call_title}\n")
                f.write("="*60 + "\n\n")
                f.write(text)
                f.write("\n\n" + "="*60)
            
            self._stop_loader()
            
            print(f"✅ Transcription complete!")
            print(f"\n📄 Transcript preview:")
            print("-" * 60)
            print(text[:300] + ("..." if len(text) > 300 else ""))
            print("-" * 60)
                
        except Exception as e:
            self._stop_loader()
            print(f"❌ Error during transcription: {e}")
            with open(self.transcript_filename, 'w', encoding='utf-8') as f:
                f.write(f"Error transcribing audio: {str(e)}")
            
    def set_language(self, language_code):
        """Change the transcription language"""
        if language_code in whisper.tokenizer.LANGUAGES:
            self.language = language_code
            print(f"✅ Language changed to: {self._get_language_name(language_code)}")
            return True
        else:
            print(f"❌ Invalid language code: {language_code}")
            print(f"Available languages: {', '.join(sorted(whisper.tokenizer.LANGUAGES.keys()))}")
            return False
    
    def _get_language_name(self, code):
        """Get language name from code"""
        language_names = {
            "pt": "Portuguese (Brazil)",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "zh": "Chinese",
            "ko": "Korean",
            "ru": "Russian",
        }
        return language_names.get(code, whisper.tokenizer.LANGUAGES.get(code, code))
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


def main():
    print("\n" + "="*60)
    print("🎙️  CALL RECORDING & TRANSCRIPTION TOOL")
    print("="*60)
    print("This tool records your microphone and creates transcripts")
    print("Use for: meetings, interviews, call notes, etc.")
    print("="*60 + "\n")
    
    recorder = CallRecorder()
    
    try:
        print("Commands:")
        print("  'start'    - Begin recording")
        print("  'stop'     - Stop recording and transcribe")
        print("  'language' - Change transcription language")
        print("  'quit'     - Exit program\n")
        
        while True:
            command = input("Enter command: ").strip().lower()
            
            if command == 'start':
                # Ask for call title (optional)
                print("\n📝 Enter call title (optional, press Enter to skip):")
                title = input("Title: ").strip()
                if not title:
                    print("ℹ️  Starting recording without title...")
                else:
                    print(f"✅ Title set: {title}")
                recorder.start_recording(title if title else None)
            elif command == 'stop':
                recorder.stop_recording()
            elif command == 'language' or command == 'lang':
                print(f"\nCurrent language: {recorder._get_language_name(recorder.language)}")
                print("\nCommon languages:")
                print("  pt - Portuguese (Brazil)")
                print("  en - English")
                print("  es - Spanish")
                print("  fr - French")
                print("  de - German")
                print("  it - Italian")
                print("  ja - Japanese")
                print("  zh - Chinese")
                print("  ko - Korean")
                print("  ru - Russian")
                print("\nOr type 'list' to see all available languages")
                lang_input = input("Enter language code: ").strip().lower()
                if lang_input == 'list':
                    print("\nAll available languages:")
                    for code, name in sorted(whisper.tokenizer.LANGUAGES.items()):
                        print(f"  {code} - {name}")
                    print()
                else:
                    recorder.set_language(lang_input)
            elif command == 'quit' or command == 'exit':
                if recorder.is_recording:
                    recorder.stop_recording()
                break
            else:
                print("❌ Unknown command. Use: start, stop, language, or quit")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if recorder.is_recording:
            recorder.stop_recording()
    finally:
        recorder.cleanup()
        print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import pyaudio
        import whisper
    except ImportError as e:
        print("❌ Missing required package!")
        print("\nPlease install required packages:")
        print("  pip install pyaudio openai-whisper")
        print("\nNote: On some systems you may need:")
        print("  - Windows: pip install pipwin && pipwin install pyaudio")
        print("  - Mac: brew install portaudio && pip install pyaudio")
        print("  - Linux: sudo apt-get install portaudio19-dev python3-pyaudio")
        print("\nAlso make sure ffmpeg is installed:")
        print("  - Mac: brew install ffmpeg")
        print("  - Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - Windows: choco install ffmpeg")
        exit(1)
    
    main()