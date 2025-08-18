import torch
import torchaudio
from transformers import pipeline
import numpy as np

# Initialize the TTS model
def initialize_tts():
    model_id = '11mlabs/indri-0.1-350m-tts'
    task = 'indri-tts'
    pipe = pipeline(
        task,
        model=model_id,
        device=torch.device('cpu'),
        trust_remote_code=True
    )
    return pipe

class ImprovedTTSProcessor:
    def __init__(self, pipeline, sample_rate=24000):
        self.pipe = pipeline
        self.sample_rate = sample_rate
        self.silence_duration = 0.3  # seconds of silence between chunks
        self.silence_samples = int(self.sample_rate * self.silence_duration)
    
    def smart_chunk_text(self, text, max_chars=300):
        """
        Intelligently chunk text based on sentence boundaries and punctuation.
        Prioritizes natural breaks over character limits.
        """
        # Split by multiple delimiters common in Hindi/multilingual text
        import re
        
        # Define sentence endings - adjust based on your language needs
        sentence_endings = ['।', '.', '!', '?', '|']
        
        # Split by sentence endings but keep the delimiter
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Now group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit and current chunk isn't empty
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add space only if current_chunk is not empty
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def create_silence(self, duration_seconds=None):
        """Create a tensor of silence."""
        if duration_seconds is None:
            duration_seconds = self.silence_duration
        
        silence_samples = int(self.sample_rate * duration_seconds)
        return torch.zeros(1, silence_samples)
    
    def process_long_text(self, text, output_file='./output/output.wav', max_chars=100, 
                         add_silence_between_chunks=True):
        """
        Process long text and create a single audio file.
        
        Args:
            text (str): Long text to convert to speech
            output_file (str): Output filename
            max_chars (int): Maximum characters per chunk
            add_silence_between_chunks (bool): Add silence between chunks
            
        Returns:
            str: Path to the generated audio file
        """
        print(f"Processing text of length: {len(text)} characters")
        
        # Chunk the text intelligently
        chunks = self.smart_chunk_text(text, max_chars)
        print(f"Text split into {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
        
        # Process each chunk and collect audio tensors
        audio_tensors = []
        
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx+1}/{len(chunks)}...")
            
            try:
                # Generate audio for this chunk
                out = self.pipe([chunk], speaker='[spkr_77]')
                audio_tensor = out[0]['audio'][0]
                
                # Ensure audio is 2D (channels, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                audio_tensors.append(audio_tensor)
                
                # Add silence between chunks (except after the last chunk)
                if add_silence_between_chunks and idx < len(chunks) - 1:
                    silence = self.create_silence()
                    audio_tensors.append(silence)
                    
            except Exception as e:
                print(f"Error processing chunk {idx+1}: {e}")
                # Add a short silence instead of failing completely
                silence = self.create_silence(0.1)
                audio_tensors.append(silence)
        
        if not audio_tensors:
            raise ValueError("No audio was generated successfully")
        
        # Concatenate all audio tensors
        print("Concatenating audio chunks...")
        combined_audio = torch.cat(audio_tensors, dim=1)
        
        # Save the complete audio
        print(f"Saving complete audio to: {output_file}")
        torchaudio.save(output_file, combined_audio, sample_rate=self.sample_rate)
        
        print(f"✅ Complete audio saved successfully!")
        # print(f"   Duration: {combined_audio.shape[1] / self.sample_rate:.2f} seconds")
        # print(f"   File size: {combined_audio.shape[1]} samples")
        
        return output_file
    