"""
Call Recording and Transcription Tool
Records audio from microphone and saves transcription to a file.
Shows clear visual indicators when recording is active.
"""

import pyaudio
import wave
import whisper
import torch
from datetime import datetime
import os
import threading
import time
import sys
import glob

class CallRecorder:
    def __init__(self, model_name="base", language="pt"):
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
        
        # Model setting (default: base)
        self.model_name = model_name
        self.model = None
        
        # Create audio folder if it doesn't exist
        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Initialize Whisper model
        self._load_model()
    
    def _load_model(self):
        """Load or reload the Whisper model"""
        print(f"📥 Loading Whisper model '{self.model_name}'...")
        print("   (This may take a moment - faster models load quicker)")
        self._start_loader("Loading Whisper model...")
        
        # Load model with device optimization (use GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.model_name, device=device)
        
        self._stop_loader()
        print(f"✅ Model loaded successfully! (device: {device})")
        print(f"🤖 Model: {self.model_name}")
        print(f"🌐 Language: {self._get_language_name(self.language)}")
        print(f"💡 Tip: Model is now cached in memory - subsequent recordings will be instant!")
        
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
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def _display_timer(self):
        """Display recording timer in real-time"""
        try:
            first_print = True
            while self.is_recording:
                elapsed = time.time() - self.start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                
                if hours > 0:
                    timer_str = f"⏱️  Recording time: {hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    timer_str = f"⏱️  Recording time: {minutes:02d}:{seconds:02d}"
                
                # Overwrite the previous timer line (or print first time)
                if first_print:
                    sys.stdout.write(timer_str + ' ' * 20)
                    first_print = False
                else:
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
        """Display a loading percentage"""
        percentage = 0
        start_time = time.time()
        try:
            while self.loader_active:
                elapsed = time.time() - start_time
                # Simulate progress: faster at start, slower as it approaches 100%
                # This gives a realistic feel without actual progress tracking
                if elapsed < 1:
                    percentage = min(30, int(elapsed * 30))
                elif elapsed < 3:
                    percentage = min(70, 30 + int((elapsed - 1) * 20))
                elif elapsed < 10:
                    percentage = min(95, 70 + int((elapsed - 3) * 3.5))
                else:
                    percentage = min(99, 95 + int((elapsed - 10) * 0.5))
                
                bar_length = 20
                filled = int(bar_length * percentage / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                sys.stdout.write(f'\r{message} [{bar}] {percentage}%')
                sys.stdout.flush()
                time.sleep(0.1)
            # Show 100% before clearing
            bar = '█' * bar_length
            sys.stdout.write(f'\r{message} [{bar}] 100%')
            sys.stdout.flush()
            time.sleep(0.2)
            # Clear the loader line
            sys.stdout.write('\r' + ' ' * (len(message) + 30) + '\r')
            sys.stdout.flush()
        except Exception:
            pass
    
    def _start_loader(self, message):
        """Start a loading percentage indicator in a separate thread"""
        self.loader_active = True
        self.loader_thread = threading.Thread(target=self._show_loader, args=(message,), daemon=True)
        self.loader_thread.start()
    
    def _stop_loader(self):
        """Stop the loading percentage indicator"""
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
    
    def set_model(self, model_name):
        """Change the Whisper model"""
        valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
        if model_name.lower() in valid_models:
            if self.is_recording:
                print("⚠️  Cannot change model while recording. Stop recording first.")
                return False
            
            old_model = self.model_name
            self.model_name = model_name.lower()
            
            print(f"\n🔄 Changing model from '{old_model}' to '{self.model_name}'...")
            self._load_model()
            return True
        else:
            print(f"❌ Invalid model name: {model_name}")
            print(f"Available models: {', '.join(valid_models)}")
            print("\nModel descriptions:")
            print("  tiny   - Fastest, least accurate (~1 GB VRAM)")
            print("  base   - Fast, less accurate (~1 GB VRAM)")
            print("  small  - Balanced (~2 GB VRAM)")
            print("  medium - More accurate (~5 GB VRAM)")
            print("  large  - Most accurate (~10 GB VRAM)")
            print("  turbo  - Optimized for speed (~6 GB VRAM)")
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
    
    def _get_audio_files(self, folder_path):
        """Get all audio files from a folder"""
        audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg', '*.wma']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
            audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        return sorted(audio_files)
    
    def _list_folders(self, base_path):
        """List all folders in a directory"""
        folders = []
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    folders.append(item)
        return sorted(folders)
    
    def transcribe_file(self, audio_file_path, output_path=None, title=None):
        """Transcribe an existing audio file"""
        if not os.path.exists(audio_file_path):
            print(f"❌ File not found: {audio_file_path}")
            return False
        
        if not os.path.isfile(audio_file_path):
            print(f"❌ Path is not a file: {audio_file_path}")
            return False
        
        # Check if it's an audio file
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        if file_ext not in audio_extensions:
            print(f"❌ File is not a supported audio format: {file_ext}")
            print(f"Supported formats: {', '.join(audio_extensions)}")
            return False
        
        # Determine output path
        if output_path is None:
            base_dir = os.path.dirname(audio_file_path)
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_transcript.txt")
        
        try:
            print(f"\n📊 Processing audio file: {os.path.basename(audio_file_path)}")
            print(f"🗣️  Converting speech to text ({self._get_language_name(self.language)})...")
            
            # Start loader for transcription
            self._start_loader("🎯 Transcribing audio (this may take a moment)...")
            
            # Transcribe using Whisper model with specified language
            result = self.model.transcribe(audio_file_path, language=self.language)
            text = result["text"].strip()
            
            self._stop_loader()
            
            # Start loader for saving transcript
            self._start_loader("💾 Saving transcript...")
            
            # Save transcript to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Audio Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Audio file: {audio_file_path}\n")
                if title:
                    f.write(f"Title: {title}\n")
                f.write("="*60 + "\n\n")
                f.write(text)
                f.write("\n\n" + "="*60)
            
            self._stop_loader()
            
            print(f"✅ Transcription complete!")
            print(f"📁 Transcript saved: {output_path}")
            print(f"\n📄 Transcript preview:")
            print("-" * 60)
            print(text[:300] + ("..." if len(text) > 300 else ""))
            print("-" * 60)
            return True
                
        except Exception as e:
            self._stop_loader()
            print(f"❌ Error during transcription: {e}")
            return False
    
    def transcribe_folder(self, folder_path, title=None):
        """Transcribe all audio files in a folder"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return False
        
        if not os.path.isdir(folder_path):
            print(f"❌ Path is not a folder: {folder_path}")
            return False
        
        audio_files = self._get_audio_files(folder_path)
        
        if not audio_files:
            print(f"❌ No audio files found in: {folder_path}")
            return False
        
        print(f"\n📁 Found {len(audio_files)} audio file(s) in folder")
        print(f"📂 Folder: {folder_path}\n")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(folder_path, f"{base_name}_transcript.txt")
            self.transcribe_file(audio_file, output_path, title)
        
        print(f"\n✅ Finished transcribing {len(audio_files)} file(s)")
        return True
    
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
        print("  'start'     - Begin recording")
        print("  'stop'      - Stop recording and transcribe")
        print("  'transcribe' - Transcribe existing audio file/folder")
        print("  'language'  - Change transcription language")
        print("  'model'     - Change Whisper model")
        print("  'quit'      - Exit program\n")
        
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
            elif command == 'transcribe' or command == 'transcrever':
                print("\n📁 Transcribe existing audio file or folder")
                print("\nOptions:")
                print("  1. Transcribe a specific audio file")
                print("  2. Transcribe all audio files in a folder")
                print("  3. Browse folders in audio directory")
                
                choice = input("\nEnter option (1/2/3): ").strip()
                
                if choice == '1':
                    file_path = input("\nEnter path to audio file: ").strip()
                    if file_path:
                        # Expand user path and resolve
                        file_path = os.path.expanduser(file_path)
                        file_path = os.path.abspath(file_path)
                        
                        print("\n📝 Enter title (optional, press Enter to skip):")
                        title = input("Title: ").strip() or None
                        
                        recorder.transcribe_file(file_path, title=title)
                    else:
                        print("❌ No file path provided")
                
                elif choice == '2':
                    folder_path = input("\nEnter path to folder: ").strip()
                    if folder_path:
                        # Expand user path and resolve
                        folder_path = os.path.expanduser(folder_path)
                        folder_path = os.path.abspath(folder_path)
                        
                        print("\n📝 Enter title (optional, press Enter to skip):")
                        title = input("Title: ").strip() or None
                        
                        recorder.transcribe_folder(folder_path, title=title)
                    else:
                        print("❌ No folder path provided")
                
                elif choice == '3':
                    # Browse folders in audio directory
                    base_folder = recorder.audio_folder
                    
                    # Check for audio files in root folder
                    root_audio_files = recorder._get_audio_files(base_folder)
                    folders = recorder._list_folders(base_folder)
                    
                    if not folders and not root_audio_files:
                        print(f"\n❌ No folders or audio files found in {base_folder}")
                        continue
                    
                    print(f"\n📂 Available options in '{base_folder}':")
                    item_index = 1
                    items_list = []
                    
                    # Add root folder option if it has audio files
                    if root_audio_files:
                        print(f"  {item_index}. 📁 {base_folder} (root) - {len(root_audio_files)} audio file(s)")
                        items_list.append(('root', base_folder))
                        item_index += 1
                    
                    # Add subfolders
                    for folder in folders:
                        folder_path = os.path.join(base_folder, folder)
                        audio_files = recorder._get_audio_files(folder_path)
                        print(f"  {item_index}. 📁 {folder} ({len(audio_files)} audio file(s))")
                        items_list.append(('folder', folder_path))
                        item_index += 1
                    
                    folder_input = input("\nEnter option number or folder name: ").strip()
                    
                    selected_item = None
                    if folder_input.isdigit():
                        idx = int(folder_input) - 1
                        if 0 <= idx < len(items_list):
                            selected_item = items_list[idx]
                    else:
                        # Try to match by folder name
                        for item_type, item_path in items_list:
                            if item_type == 'folder' and os.path.basename(item_path) == folder_input:
                                selected_item = (item_type, item_path)
                                break
                        # Also check if it's "root" or base folder name
                        if not selected_item and folder_input.lower() in ['root', '.', base_folder]:
                            for item_type, item_path in items_list:
                                if item_type == 'root':
                                    selected_item = (item_type, item_path)
                                    break
                    
                    if selected_item:
                        folder_path = selected_item[1]
                        print("\n📝 Enter title (optional, press Enter to skip):")
                        title = input("Title: ").strip() or None
                        recorder.transcribe_folder(folder_path, title=title)
                    else:
                        print("❌ Invalid selection")
                else:
                    print("❌ Invalid option")
            
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
            elif command == 'model' or command == 'modelo':
                print(f"\nCurrent model: {recorder.model_name}")
                print("\nAvailable models:")
                print("  tiny   - Fastest, least accurate (~1 GB VRAM)")
                print("  base   - Fast, less accurate (~1 GB VRAM) [Default]")
                print("  small  - Balanced (~2 GB VRAM)")
                print("  medium - More accurate (~5 GB VRAM)")
                print("  large  - Most accurate (~10 GB VRAM)")
                print("  turbo  - Optimized for speed (~6 GB VRAM)")
                model_input = input("\nEnter model name: ").strip().lower()
                recorder.set_model(model_input)
            elif command == 'quit' or command == 'exit':
                if recorder.is_recording:
                    recorder.stop_recording()
                break
            else:
                print("❌ Unknown command. Use: start, stop, transcribe, language, model, or quit")
                
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