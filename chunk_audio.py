import os
import wave

def chunk_audio_file(input_file, chunk_size_mb=40):
    """
    Split an audio file into chunks of specified maximum size (in MB).
    Each chunk will be a valid, playable audio file.
    :param input_file: Path to the input audio file.
    :param chunk_size_mb: Maximum size for each chunk in megabytes.
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    file_size = os.path.getsize(input_file)

    if file_size <= chunk_size_bytes:
        print(f"File is smaller than {chunk_size_mb}MB, no chunking needed.")
        return [input_file]
    
    # Open the input WAV file
    try:
        with wave.open(input_file, 'rb') as wav_in:
            # Get audio parameters
            channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            frame_rate = wav_in.getframerate()
            n_frames = wav_in.getnframes()
            comptype = wav_in.getcomptype()
            compname = wav_in.getcompname()
            
            # Calculate bytes per frame
            bytes_per_frame = channels * sample_width
            
            # Calculate approximate frames per chunk (accounting for WAV header ~44 bytes)
            # WAV header is typically 44 bytes, but we'll use a conservative estimate
            header_size = 44
            available_bytes_per_chunk = chunk_size_bytes - header_size
            frames_per_chunk = available_bytes_per_chunk // bytes_per_frame
            
            # Ensure we don't exceed the total number of frames
            frames_per_chunk = min(frames_per_chunk, n_frames)
            
            if frames_per_chunk <= 0:
                print(f"Error: Chunk size too small. Minimum size needed: {header_size + bytes_per_frame} bytes")
                return []
            
            print(f"Audio parameters: {channels} channel(s), {sample_width} byte(s) per sample, {frame_rate} Hz")
            print(f"Splitting into chunks of approximately {frames_per_chunk} frames each")
            
            basename, ext = os.path.splitext(os.path.basename(input_file))
            output_files = []
            idx = 0
            frames_read = 0
            
            while frames_read < n_frames:
                # Calculate frames for this chunk
                frames_to_read = min(frames_per_chunk, n_frames - frames_read)
                
                # Read frames for this chunk
                wav_in.setpos(frames_read)
                frames_data = wav_in.readframes(frames_to_read)
                
                if not frames_data:
                    break
                
                # Create output file path
                output_path = f"{basename}_part{idx}{ext}"
                
                # Write chunk as a valid WAV file
                with wave.open(output_path, 'wb') as wav_out:
                    wav_out.setnchannels(channels)
                    wav_out.setsampwidth(sample_width)
                    wav_out.setframerate(frame_rate)
                    wav_out.setcomptype(comptype, compname)
                    wav_out.writeframes(frames_data)
                
                output_files.append(output_path)
                chunk_duration = frames_to_read / frame_rate
                chunk_size_actual = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Created chunk {idx}: {output_path} ({chunk_duration:.2f}s, {chunk_size_actual:.2f}MB)")
                
                frames_read += frames_to_read
                idx += 1
            
            return output_files
            
    except wave.Error as e:
        print(f"Error: Not a valid WAV file or WAV file error: {e}")
        return []
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunk_audio.py <audio_file> [chunk_size_mb]")
        print("Example: python chunk_audio.py audio.wav 40")
    else:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"Error: File not found: {audio_file}")
            sys.exit(1)
        chunk_size_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        result = chunk_audio_file(audio_file, chunk_size_mb=chunk_size_arg)
        if result:
            print(f"\n✅ Successfully created {len(result)} chunk(s)")
        else:
            print("\n❌ Failed to create chunks")
            sys.exit(1)
