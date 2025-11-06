#!/usr/bin/env python3
"""
Audio Recorder and Transcriber using Whisper
Make sure to run this with the virtual environment activated:
    source venv/bin/activate
    python3 test.py
Or use the venv Python directly:
    venv/bin/python test.py
"""

import pyaudio
import wave
import whisper
import os
import threading
import time
from datetime import datetime
import sys

class AudioRecorderTranscriber:
    def __init__(self, model_size="base"):
        """
        Inicializa o gravador e transcritor de áudio.
        
        Args:
            model_size: Tamanho do modelo Whisper (tiny, base, small, medium, large)
        """
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.recording = False
        self.frames = []
        self.start_time = None
        
        print(f"Carregando modelo Whisper '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("Modelo carregado com sucesso!")
        
        self.audio = pyaudio.PyAudio()
        
    def _format_time(self, seconds):
        """Formata o tempo em MM:SS ou HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _update_timer(self):
        """Atualiza o cronômetro na tela"""
        while self.recording:
            if self.start_time:
                elapsed = time.time() - self.start_time
                time_str = self._format_time(elapsed)
                print(f"\r🔴 Gravando... {time_str} | Pressione ENTER para parar", end="", flush=True)
            time.sleep(0.1)  # Atualiza a cada 0.1 segundos para suavidade
    
    def start_recording(self):
        """Inicia a gravação de áudio"""
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Iniciar thread do cronômetro
        timer_thread = threading.Thread(target=self._update_timer, daemon=True)
        timer_thread.start()
        
        print("\n🔴 Gravando... 00:00 | Pressione ENTER para parar")
        
        while self.recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"\nErro na gravação: {e}")
                break
        
    
    def stop_recording(self):
        """Para a gravação de áudio"""
        if self.recording:
            self.recording = False
            # Pequeno delay para garantir que o cronômetro pare antes de limpar
            time.sleep(0.15)
            # Limpar linha do cronômetro
            print("\r" + " " * 70 + "\r", end="", flush=True)
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                time_str = self._format_time(elapsed)
                self.stream.stop_stream()
                self.stream.close()
                print(f"⏹️  Gravação finalizada | Duração: {time_str}")
            else:
                self.stream.stop_stream()
                self.stream.close()
                print("⏹️  Gravação finalizada")
            self.start_time = None
    
    def save_audio(self, filename=None):
        """Salva o áudio gravado em arquivo WAV"""
        if not self.frames:
            print("Nenhum áudio para salvar")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gravacao_{timestamp}.wav"
        
        # Criar pasta de gravações se não existir
        os.makedirs("gravacoes", exist_ok=True)
        filepath = os.path.join("gravacoes", filename)
        
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"✅ Áudio salvo: {filepath}")
        return filepath
    
    def transcribe_audio(self, audio_file, language="pt"):
        """
        Transcreve um arquivo de áudio usando Whisper
        
        Args:
            audio_file: Caminho do arquivo de áudio
            language: Idioma do áudio (pt, en, es, etc.)
        """
        print(f"\n🎯 Transcrevendo áudio: {audio_file}")
        
        try:
            result = self.model.transcribe(
                audio_file,
                language=language,
                fp16=False,
                verbose=False
            )
            
            transcription = result["text"]
            
            # Salvar transcrição em arquivo txt
            txt_file = audio_file.replace(".wav", ".txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            print(f"\n📝 Transcrição:\n{transcription}\n")
            print(f"✅ Transcrição salva: {txt_file}")
            
            return transcription
            
        except Exception as e:
            print(f"❌ Erro na transcrição: {e}")
            return None
    
    def record_and_transcribe(self, language="pt"):
        """Grava áudio e transcreve automaticamente"""
        # Iniciar gravação em thread separada
        record_thread = threading.Thread(target=self.start_recording, daemon=True)
        record_thread.start()
        
        # Aguardar pressionar ENTER para parar
        try:
            input()  # Espera o usuário pressionar ENTER
        except (EOFError, KeyboardInterrupt):
            pass
        
        # Parar gravação
        self.stop_recording()
        record_thread.join(timeout=1.0)
        
        # Salvar áudio
        audio_file = self.save_audio()
        
        if audio_file:
            # Transcrever
            self.transcribe_audio(audio_file, language=language)
    
    def transcribe_existing_file(self, filepath, language="pt"):
        """Transcreve um arquivo de áudio existente"""
        if not os.path.exists(filepath):
            print(f"❌ Arquivo não encontrado: {filepath}")
            return None
        
        return self.transcribe_audio(filepath, language=language)
    
    def close(self):
        """Fecha recursos"""
        self.audio.terminate()


def main():
    """Função principal com menu interativo"""
    print("=" * 50)
    print("🎤 Sistema de Gravação e Transcrição com Whisper")
    print("=" * 50)
    
    # Escolher tamanho do modelo
    print("\nTamanhos de modelo disponíveis:")
    print("1. tiny   - Rápido, menos preciso")
    print("2. base   - Balanceado (recomendado)")
    print("3. small  - Mais preciso, mais lento")
    print("4. medium - Muito preciso, lento")
    print("5. large  - Máxima precisão, muito lento")
    
    model_choice = input("\nEscolha o modelo (1-5) [2]: ").strip() or "2"
    models = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    model_size = models.get(model_choice, "base")
    
    # Inicializar sistema
    recorder = AudioRecorderTranscriber(model_size=model_size)
    
    while True:
        print("\n" + "=" * 50)
        print("Escolha uma opção:")
        print("1. Gravar e transcrever nova chamada")
        print("2. Transcrever arquivo existente")
        print("3. Sair")
        print("=" * 50)
        
        choice = input("\nOpção: ").strip()
        
        if choice == "1":
            language = input("Idioma (pt/en/es) [pt]: ").strip() or "pt"
            print("\nPressione ENTER para parar a gravação")
            input("Pressione ENTER para começar a gravar...")
            recorder.record_and_transcribe(language=language)
            
        elif choice == "2":
            filepath = input("Caminho do arquivo de áudio: ").strip()
            language = input("Idioma (pt/en/es) [pt]: ").strip() or "pt"
            recorder.transcribe_existing_file(filepath, language=language)
            
        elif choice == "3":
            print("\n👋 Encerrando...")
            recorder.close()
            break
        
        else:
            print("❌ Opção inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)