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
import numpy as np

# Try to import sounddevice for system audio recording
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice not available. System audio recording will be limited.")
    print("   Install with: pip install sounddevice")

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
        
        # Audio source settings
        self.audio_source = "both"  # "mic", "system", or "both"
        self.mic_stream = None
        self.system_stream = None
        self.system_frames = []
        self.system_device_index = None
        self.use_sounddevice_for_system = False
        self.system_sample_rate = self.RATE
        
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
    
    def _get_audio_devices(self):
        """Get list of available audio input devices"""
        devices = []
        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
        except Exception as e:
            print(f"⚠️  Error getting audio devices: {e}")
        return devices

    def _normalize_device_name(self, name):
        """Normalize device names for reliable comparison"""
        if not name:
            return ""
        return ''.join(ch for ch in name.lower() if ch.isalnum())

    def _find_loopback_device(self):
        """Find a loopback device (like BlackHole) for system audio recording"""
        loopback_keywords = ['blackhole', 'loopback', 'soundflower', 'monitor', 'virtual']

        print("\n🔍 Procurando dispositivos de loopback...")

        candidates = []
        pyaudio_devices = []

        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                info = self.audio.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    normalized = self._normalize_device_name(info.get('name', ''))
                    device_entry = {
                        'index': i,
                        'name': info.get('name', f'Device {i}') or f'Device {i}',
                        'normalized': normalized,
                        'channels': int(info.get('maxInputChannels', 0)),
                        'default_samplerate': int(info.get('defaultSampleRate', self.RATE)) if info.get('defaultSampleRate') else None,
                    }
                    pyaudio_devices.append(device_entry)
                    if any(keyword in normalized for keyword in loopback_keywords):
                        candidates.append({
                            'display_name': device_entry['name'],
                            'normalized': normalized,
                            'pyaudio_index': device_entry['index'],
                            'channels': device_entry['channels'],
                            'default_samplerate': device_entry['default_samplerate'],
                            'sounddevice_index': None,
                        })
        except Exception as e:
            print(f"⚠️  Erro ao verificar dispositivos PyAudio: {e}")

        sd_devices_map = {}
        if SOUNDDEVICE_AVAILABLE:
            try:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        normalized = self._normalize_device_name(device['name'])
                        sd_devices_map[normalized] = {
                            'index': i,
                            'name': device['name'],
                            'channels': int(device['max_input_channels']),
                            'default_samplerate': int(device['default_samplerate']),
                        }
                        if any(keyword in normalized for keyword in loopback_keywords):
                            if not any(c['normalized'] == normalized for c in candidates):
                                candidates.append({
                                    'display_name': device['name'],
                                    'normalized': normalized,
                                    'pyaudio_index': None,
                                    'channels': int(device['max_input_channels']),
                                    'default_samplerate': int(device['default_samplerate']),
                                    'sounddevice_index': i,
                                })
            except Exception as e:
                print(f"⚠️  Erro ao verificar dispositivos sounddevice: {e}")
        else:
            print("⚠️  sounddevice não está disponível. Instale com: pip install sounddevice")

        if not candidates:
            print("   ❌ Nenhum dispositivo de loopback encontrado!")
            print("\n⚠️  DIAGNÓSTICO:")
            print("   1. Verifique se BlackHole está instalado")
            print("   2. Execute 'devices' para ver todos os dispositivos")
            print("   3. Verifique se BlackHole está configurado como saída no Sistema")
            return None

        # Enrich candidates with missing indices/info by matching normalized names
        for candidate in candidates:
            normalized = candidate['normalized']
            if candidate.get('sounddevice_index') is None and normalized in sd_devices_map:
                info = sd_devices_map[normalized]
                candidate['sounddevice_index'] = info['index']
                candidate['display_name'] = info['name']
                candidate['channels'] = info['channels']
                candidate['default_samplerate'] = info['default_samplerate']
            if candidate.get('pyaudio_index') is None:
                for device_entry in pyaudio_devices:
                    if device_entry['normalized'] == normalized:
                        candidate['pyaudio_index'] = device_entry['index']
                        candidate['display_name'] = device_entry['name']
                        candidate['channels'] = device_entry['channels']
                        candidate['default_samplerate'] = device_entry['default_samplerate']
                        break

        for candidate in candidates:
            idx_info = []
            if candidate.get('pyaudio_index') is not None:
                idx_info.append(f"PyAudio #{candidate['pyaudio_index']}")
            if candidate.get('sounddevice_index') is not None:
                idx_info.append(f"sounddevice #{candidate['sounddevice_index']}")
            idx_str = ', '.join(idx_info) if idx_info else 'sem índice disponível'
            samplerate = candidate.get('default_samplerate')
            samplerate_str = f"{samplerate} Hz" if samplerate else "desconhecida"
            print(f"   ✅ Encontrado: [{idx_str}] {candidate['display_name']} (taxa padrão: {samplerate_str})")

        def candidate_score(entry):
            name = entry['display_name'].lower()
            score = 0
            if 'blackhole' in name:
                score += 5
            if '2ch' in name:
                score += 2
            if 'loopback' in name or 'monitor' in name or 'virtual' in name:
                score += 1
            if entry.get('pyaudio_index') is not None:
                score += 2
            return score

        best_candidate = max(candidates, key=candidate_score)
        return best_candidate
    
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
        
    def start_recording(self, title=None, audio_source="both"):
        """Start recording audio from microphone, system, or both
        
        Args:
            title: Optional title for the call
            audio_source: "mic", "system", or "both" (default: "both")
        """
        if self.is_recording:
            print("⚠️  Already recording!")
            return
        
        # Validate audio source
        if audio_source not in ["mic", "system", "both"]:
            print(f"⚠️  Invalid audio source: {audio_source}. Using 'both'.")
            audio_source = "both"
        
        self.audio_source = audio_source
        
        # Check if system audio is requested but sounddevice is not available
        if audio_source in ["system", "both"] and not SOUNDDEVICE_AVAILABLE:
            print("⚠️  sounddevice não está instalado. Tentaremos capturar usando apenas PyAudio.")
            print("   Instale com: pip install sounddevice (recomendado para fallback)")
        
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
        self.system_frames = []
        self.system_stream = None
        self.system_device_index = None
        self.use_sounddevice_for_system = False
        self.system_sample_rate = self.RATE
        self.is_recording = True
        
        # Open microphone stream if needed
        if audio_source in ["mic", "both"]:
            try:
                self.mic_stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
            except Exception as e:
                print(f"❌ Error opening microphone: {e}")
                self.is_recording = False
                return
        
        # Open system audio stream if needed
        if audio_source in ["system", "both"]:
            try:
                print("\n🔧 Configurando gravação de áudio do sistema...")
                loopback_info = self._find_loopback_device()
                if loopback_info is None:
                    print("\n❌ PROBLEMA: Dispositivo de loopback não encontrado!")
                    print("\n📋 SOLUÇÃO:")
                    print("   1. Instale BlackHole: https://github.com/ExistentialAudio/BlackHole/releases")
                    print("   2. Execute: ./setup_audio.sh")
                    print("   3. Configure a saída do sistema para BlackHole 2ch")
                    print("   4. Execute 'devices' para verificar se foi detectado")
                    if audio_source == "system":
                        print("\n❌ Não é possível gravar áudio do sistema sem dispositivo de loopback.")
                        self.is_recording = False
                        if self.mic_stream:
                            self.mic_stream.close()
                        return
                    else:
                        print("\n⚠️  Continuando apenas com microfone...")
                        self.audio_source = "mic"
                else:
                    device_name = loopback_info.get('display_name', 'Loopback')
                    default_rate = loopback_info.get('default_samplerate')
                    channels = loopback_info.get('channels', self.CHANNELS)
                    print(f"✅ Usando dispositivo de loopback: {device_name}")
                    if channels:
                        print(f"   Canais: {channels}")
                    if default_rate:
                        print(f"   Sample Rate padrão: {default_rate} Hz")
                    
                    print("\n⚠️  IMPORTANTE:")
                    print("   Certifique-se de que BlackHole está configurado como")
                    print("   dispositivo de SAÍDA no Sistema > Som")
                    print("   Caso contrário, não haverá áudio para gravar!")

                    pyaudio_index = loopback_info.get('pyaudio_index')
                    stream_opened = False
                    last_error = None

                    if pyaudio_index is not None:
                        candidate_rates = [self.RATE]
                        if default_rate and default_rate not in candidate_rates:
                            candidate_rates.append(int(default_rate))
                        for fallback_rate in [44100, 48000, 32000, 16000]:
                            if fallback_rate not in candidate_rates:
                                candidate_rates.append(fallback_rate)
                        candidate_rates = [rate for rate in candidate_rates if rate]

                        for rate in candidate_rates:
                            try:
                                stream = self.audio.open(
                                    format=self.FORMAT,
                                    channels=self.CHANNELS,
                                    rate=rate,
                                    input=True,
                                    frames_per_buffer=self.CHUNK,
                                    input_device_index=pyaudio_index
                                )
                                self.system_stream = stream
                                self.system_sample_rate = int(rate)
                                self.use_sounddevice_for_system = False
                                stream_opened = True
                                print(f"\n🎧 Capturando áudio do sistema via PyAudio (taxa: {rate} Hz)")
                                break
                            except Exception as stream_error:
                                last_error = stream_error

                        if not stream_opened and last_error:
                            print(f"⚠️  Falha ao abrir PyAudio (último erro: {last_error})")

                    if not stream_opened:
                        sounddevice_index = loopback_info.get('sounddevice_index')
                        if SOUNDDEVICE_AVAILABLE and sounddevice_index is not None:
                            self.use_sounddevice_for_system = True
                            self.system_device_index = sounddevice_index
                            self.system_sample_rate = int(default_rate) if default_rate else self.RATE
                            print("🎧 Capturando áudio do sistema via sounddevice")
                        else:
                            print("\n❌ Não foi possível configurar a captura de áudio do sistema.")
                            if audio_source == "system":
                                print("   Dica: verifique permissões de microfone e reinstale o driver BlackHole")
                                self.is_recording = False
                                if self.mic_stream:
                                    self.mic_stream.close()
                                return
                            else:
                                print("   Continuando apenas com microfone...")
                                self.audio_source = "mic"
            except Exception as e:
                print(f"\n❌ Erro ao configurar áudio do sistema: {e}")
                import traceback
                traceback.print_exc()
                if audio_source == "system":
                    self.is_recording = False
                    if self.mic_stream:
                        self.mic_stream.close()
                    return
                else:
                    print("⚠️  Continuando apenas com microfone...")
                    self.audio_source = "mic"
        
        # Set stream for backward compatibility
        self.stream = self.mic_stream if self.mic_stream else None
        
        print("\n" + "="*60)
        print("🔴 RECORDING IN PROGRESS")
        print("="*60)
        if self.call_title:
            print(f"📝 Call title: {self.call_title}")
        print(f"🎤 Audio source: {self.audio_source}")
        print(f"📁 Recording folder: {self.recording_folder}")
        print(f"📁 Audio file: {self.audio_filename}")
        print(f"📝 Transcript file: {self.transcript_filename}")
        print("⏸️  Press Ctrl+C or type 'stop' to end recording")
        print("="*60)
        
        # Start timer
        self.start_time = time.time()
        self.timer_thread = threading.Thread(target=self._display_timer, daemon=True)
        self.timer_thread.start()
        
        # Start recording threads
        self.recording_threads = []
        
        if self.mic_stream:
            mic_thread = threading.Thread(target=self._record_mic, daemon=True)
            mic_thread.start()
            self.recording_threads.append(mic_thread)
        
        if self.system_stream is not None or self.use_sounddevice_for_system:
            system_thread = threading.Thread(target=self._record_system, daemon=True)
            system_thread.start()
            self.recording_threads.append(system_thread)
        
        # Keep reference for backward compatibility
        self.recording_thread = self.recording_threads[0] if self.recording_threads else None
        
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
        """Internal method to record audio (backward compatibility)"""
        self._record_mic()
    
    def _record_mic(self):
        """Internal method to record audio from microphone"""
        try:
            while self.is_recording and self.mic_stream:
                data = self.mic_stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"❌ Microphone recording error: {e}")
    
    def _record_system(self):
        """Internal method to record system audio"""
        system_rate = self.system_sample_rate or self.RATE
        max_silent_frames = max(int((system_rate / self.CHUNK) * 3), 1)
        silent_frames = 0
        frames_count = 0

        if self.use_sounddevice_for_system:
            if not SOUNDDEVICE_AVAILABLE or self.system_device_index is None:
                print("⚠️  Não é possível gravar áudio do sistema (sounddevice indisponível)")
                return

            try:
                def callback(indata, frames, time_info, status):
                    nonlocal silent_frames, frames_count

                    if status:
                        print(f"\n⚠️  Status do áudio do sistema: {status}")

                    if not self.is_recording:
                        return (None, sd.CallbackStop)

                    frames_count += 1

                    audio_buffer = indata
                    if audio_buffer.ndim > 1 and audio_buffer.shape[1] > 1:
                        audio_buffer = np.mean(audio_buffer, axis=1, keepdims=True)

                    audio_level = np.abs(audio_buffer).max()
                    if audio_level < 0.001:
                        silent_frames += 1
                        if silent_frames == max_silent_frames:
                            print("\n⚠️  AVISO: Áudio do sistema parece estar silencioso!")
                            print("   Verifique se BlackHole está como saída e se há áudio tocando")
                    else:
                        silent_frames = 0

                    audio_gained = np.clip(audio_buffer * 2.0, -1.0, 1.0)
                    audio_data = (audio_gained * 32767).astype(np.int16)
                    self.system_frames.append(audio_data.tobytes())
                    return (None, sd.CallbackContinue)

                print("🎙️  Iniciando gravação de áudio do sistema (sounddevice)...")
                with sd.InputStream(
                    device=self.system_device_index,
                    channels=self.CHANNELS,
                    samplerate=system_rate,
                    callback=callback,
                    blocksize=self.CHUNK,
                    dtype='float32'
                ):
                    while self.is_recording:
                        time.sleep(0.1)

                if len(self.system_frames) == 0:
                    print("\n❌ ERRO: Nenhum áudio do sistema foi gravado!")
                    print("   O dispositivo pode não estar recebendo áudio.")
            except Exception as e:
                print(f"\n❌ Erro ao gravar áudio do sistema: {e}")
                import traceback
                traceback.print_exc()
            return

        if not self.system_stream:
            print("⚠️  Não há stream de áudio do sistema configurado")
            return

        try:
            print("🎙️  Iniciando gravação de áudio do sistema (PyAudio)...")
            while self.is_recording:
                try:
                    data = self.system_stream.read(self.CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n⚠️  Falha ao ler áudio do sistema: {e}")
                    break

                frames_count += 1
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if audio_np.size == 0:
                    continue

                audio_level = np.abs(audio_np).max()
                if audio_level < 200:
                    silent_frames += 1
                    if silent_frames == max_silent_frames:
                        print("\n⚠️  AVISO: Áudio do sistema parece estar silencioso!")
                        print("   Verifique se BlackHole está configurado e se há áudio tocando")
                else:
                    silent_frames = 0

                boosted = np.clip(audio_np * 1.5, -32767, 32767).astype(np.int16)
                self.system_frames.append(boosted.tobytes())

            if len(self.system_frames) == 0:
                print("\n❌ ERRO: Nenhum áudio do sistema foi gravado!")
                print("   O dispositivo pode não estar recebendo áudio.")
        except Exception as e:
            print(f"\n❌ Erro ao gravar áudio do sistema: {e}")
            import traceback
            traceback.print_exc()
            
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
        
        # Wait for recording threads to finish
        if hasattr(self, 'recording_threads'):
            for thread in self.recording_threads:
                thread.join(timeout=2)
        elif self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        # Wait for timer thread to finish
        if self.timer_thread:
            self.timer_thread.join(timeout=1)
        
        # Close streams
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        
        if self.system_stream and not self.use_sounddevice_for_system:
            try:
                if self.system_stream.is_active():
                    self.system_stream.stop_stream()
            except Exception:
                pass
            try:
                self.system_stream.close()
            except Exception:
                pass

        # Reset system audio state (sounddevice stream closes automatically)
        self.system_stream = None
        self.system_device_index = None
        self.use_sounddevice_for_system = False
        self.system_sample_rate = self.RATE
        self.stream = None
        
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

    def _resample_audio(self, audio_array, original_rate, target_rate):
        """Resample audio array to the desired sample rate using linear interpolation"""
        if original_rate == target_rate or audio_array.size == 0:
            return audio_array
        if original_rate <= 0 or target_rate <= 0:
            return audio_array

        duration = len(audio_array) / float(original_rate)
        if duration <= 0:
            return np.array([], dtype=np.int16)

        target_length = int(round(duration * target_rate))
        if target_length <= 0:
            return np.array([], dtype=np.int16)

        original_times = np.linspace(0.0, duration, num=len(audio_array), endpoint=False)
        target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)

        resampled = np.interp(target_times, original_times, audio_array.astype(np.float32))
        resampled = np.clip(resampled, -32767, 32767)
        return resampled.astype(np.int16)

    def _get_mic_audio_array(self):
        """Return microphone audio as numpy array and its sample rate"""
        if not self.frames:
            return np.array([], dtype=np.int16), self.RATE
        mic_audio = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        return mic_audio, self.RATE

    def _get_system_audio_array(self, target_rate=None):
        """Return system audio as numpy array and the sample rate used"""
        original_rate = self.system_sample_rate or self.RATE
        if not self.system_frames:
            rate = target_rate if target_rate is not None else original_rate
            return np.array([], dtype=np.int16), rate

        system_audio = np.frombuffer(b''.join(self.system_frames), dtype=np.int16)
        if target_rate is not None and target_rate != original_rate:
            system_audio = self._resample_audio(system_audio, original_rate, target_rate)
            return system_audio, target_rate
        return system_audio, original_rate
    
    def _save_audio(self):
        """Save recorded audio to WAV file, mixing if both sources are recorded"""
        try:
            self._start_loader("💾 Salvando arquivo de áudio...")
            final_audio = np.array([], dtype=np.int16)
            final_rate = self.RATE

            if self.audio_source == "mic":
                final_audio, final_rate = self._get_mic_audio_array()
                print(f"\n📊 Microfone: {len(self.frames)} frames gravados")
            elif self.audio_source == "system":
                final_audio, final_rate = self._get_system_audio_array()
                print(f"\n📊 Sistema: {len(self.system_frames)} frames gravados")
            else:  # both
                print(f"\n📊 Microfone: {len(self.frames)} frames")
                print(f"📊 Sistema: {len(self.system_frames)} frames")

                if len(self.system_frames) == 0:
                    print("\n⚠️  AVISO: Nenhum áudio do sistema foi gravado!")
                    print("   Isso significa que apenas o microfone foi capturado.")
                    print("   Verifique a configuração do BlackHole.")

                final_audio, final_rate = self._mix_audio()

            if final_audio.size == 0:
                print("\n❌ ERRO: Nenhum áudio para salvar!")
                return

            wf = wave.open(self.audio_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(int(final_rate))
            wf.writeframes(final_audio.astype(np.int16).tobytes())
            wf.close()
            self._stop_loader()
            print(f"💾 Áudio salvo com sucesso")
        except Exception as e:
            self._stop_loader()
            print(f"❌ Erro ao salvar áudio: {e}")
            import traceback
            traceback.print_exc()
    
    def _mix_audio(self):
        """Mix microphone and system audio and return array with sample rate"""
        mic_audio, mic_rate = self._get_mic_audio_array()
        system_audio, system_rate = self._get_system_audio_array(target_rate=mic_rate)

        if mic_audio.size == 0 and system_audio.size == 0:
            return np.array([], dtype=np.int16), mic_rate
        if mic_audio.size == 0:
            return system_audio, system_rate
        if system_audio.size == 0:
            return mic_audio, mic_rate

        min_len = min(len(mic_audio), len(system_audio))
        mic_audio = mic_audio[:min_len]
        system_audio = system_audio[:min_len]

        mic_peak = np.abs(mic_audio).max()
        system_peak = np.abs(system_audio).max()

        target_peak = int(32767 * 0.8)

        if mic_peak > 0:
            mic_audio = (mic_audio.astype(np.float32) * (target_peak / mic_peak)).astype(np.int16)
        if system_peak > 0:
            system_audio = (system_audio.astype(np.float32) * (target_peak / system_peak)).astype(np.int16)

        mixed = mic_audio.astype(np.int32) + system_audio.astype(np.int32)
        mixed = np.clip(mixed, -32767, 32767).astype(np.int16)
        return mixed, mic_rate
            
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
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
        if self.stream and self.stream != self.mic_stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        self.audio.terminate()


def show_audio_setup_help():
    """Show help for setting up system audio recording"""
    print("\n" + "="*60)
    print("🔧 SYSTEM AUDIO SETUP GUIDE")
    print("="*60)
    print("\n📋 To record system audio (computer sound), you need:")
    print("\n1. Install BlackHole:")
    print("   Download from: https://github.com/ExistentialAudio/BlackHole/releases")
    print("   Install BlackHole 2ch (recommended)")
    print("\n2. Configure macOS Audio:")
    print("   Option A - Direct Output (no sound):")
    print("   - System Settings > Sound > Output > BlackHole 2ch")
    print("")
    print("   Option B - Multi-Output (hear AND record):")
    print("   - Open Audio MIDI Setup (Applications > Utilities)")
    print("   - Click '+' > Create Multi-Output Device")
    print("   - Check both: BlackHole 2ch AND your speakers/headphones")
    print("   - System Settings > Sound > Output > Multi-Output Device")
    print("\n3. Run setup script:")
    print("   ./setup_audio.sh")
    print("\n" + "="*60 + "\n")

def check_audio_devices():
    """Check and display available audio devices"""
    print("\n" + "="*60)
    print("🔍 AVAILABLE AUDIO DEVICES")
    print("="*60)
    
    if SOUNDDEVICE_AVAILABLE:
        try:
            devices = sd.query_devices()
            print("\n📱 Input devices (for recording):")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    loopback_indicators = ['blackhole', 'loopback', 'soundflower', 'monitor']
                    is_loopback = any(keyword in device['name'].lower() for keyword in loopback_indicators)
                    indicator = "✅ (Loopback)" if is_loopback else ""
                    print(f"  [{i}] {device['name']} {indicator}")
                    print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
            
            print("\n📢 Output devices:")
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    print(f"  [{i}] {device['name']}")
            
            # Check for BlackHole specifically
            has_blackhole = any('blackhole' in d['name'].lower() for d in devices if d['max_input_channels'] > 0)
            if has_blackhole:
                print("\n✅ BlackHole detected! System audio recording is ready.")
            else:
                print("\n⚠️  BlackHole not found. Install it to record system audio.")
                print("   Download: https://github.com/ExistentialAudio/BlackHole/releases")
        except Exception as e:
            print(f"❌ Error checking devices: {e}")
    else:
        print("\n⚠️  sounddevice not available. Install with: pip install sounddevice")
    
    print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print("🎙️  CALL RECORDING & TRANSCRIPTION TOOL")
    print("="*60)
    print("This tool records audio and creates transcripts")
    print("Use for: meetings, interviews, call notes, etc.")
    print("="*60 + "\n")
    
    recorder = CallRecorder()
    
    try:
        print("Commands:")
        print("  'start'     - Begin recording")
        print("  'stop'      - Stop recording and transcribe")
        print("  'transcribe' - Transcribe existing audio file/folder")
        print("  'setup'     - Show system audio setup guide")
        print("  'devices'   - List available audio devices")
        print("  'language'  - Change transcription language")
        print("  'model'     - Change Whisper model")
        print("  'quit'      - Exit program\n")
        
        while True:
            command = input("Enter command: ").strip().lower()
            
            if command == 'start':
                # Ask for audio source
                print("\n🎤 Select audio source:")
                print("  1. Microphone only")
                print("  2. System audio only (requires BlackHole on macOS)")
                print("  3. Both microphone and system audio (recommended for calls)")
                
                source_choice = input("\nEnter option (1/2/3, default: 3): ").strip()
                if source_choice == '1':
                    audio_source = "mic"
                elif source_choice == '2':
                    audio_source = "system"
                else:
                    audio_source = "both"
                
                # Ask for call title (optional)
                print("\n📝 Enter call title (optional, press Enter to skip):")
                title = input("Title: ").strip()
                if not title:
                    print("ℹ️  Starting recording without title...")
                else:
                    print(f"✅ Title set: {title}")
                recorder.start_recording(title if title else None, audio_source=audio_source)
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
            
            elif command == 'setup' or command == 'config':
                show_audio_setup_help()
            elif command == 'devices' or command == 'device':
                check_audio_devices()
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
                print("❌ Unknown command. Use: start, stop, transcribe, setup, devices, language, model, or quit")
                
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
