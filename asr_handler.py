import flash_attention_patch  # Patch transformers 4.51.3 flash attention bug
import torch
import torchaudio
from pathlib import Path
from typing import Union
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

class ASRHandler:
    def __init__(self, checkpoint_dir: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.checkpoint_dir = Path(checkpoint_dir)
        print(f"Loading model from {self.checkpoint_dir} on {self.device}...")
        
        self.config = AutoConfig.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_dir,
            config=self.config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",  # Disable flash attention to avoid compatibility issues
        ).to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
        print("Model loaded successfully.")

    def get_audio_token_length(self, seconds, merge_factor=2):
        def get_T_after_cnn(L_in, dilation=1):
            for padding, kernel_size, stride in [(1,3,1), (1,3,2)]:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            return L_out

        mel_len = int(seconds * 100)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
        audio_token_num = min(audio_token_num, 1500 // merge_factor)
        return audio_token_num

    def build_prompt(self, audio_input: Union[str, Path, torch.Tensor], chunk_seconds: int = 30) -> dict:
        if isinstance(audio_input, torch.Tensor):
            wav = audio_input
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            wav, sr = torchaudio.load(str(audio_input))
            wav = wav[:1, :] # Keep only first channel
            if sr != self.feature_extractor.sampling_rate:
                wav = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)(wav)

        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")

        audios = []
        audio_offsets = []
        audio_length = []
        chunk_size = chunk_seconds * self.feature_extractor.sampling_rate
        
        for start in range(0, wav.shape[1], chunk_size):
            chunk = wav[:, start : start + chunk_size]
            mel = self.feature_extractor(
                chunk.cpu().numpy(),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding="max_length",
            )["input_features"]
            audios.append(mel)
            seconds = chunk.shape[1] / self.feature_extractor.sampling_rate
            num_tokens = self.get_audio_token_length(seconds, self.config.merge_factor)
            
            tokens += self.tokenizer.encode("<|begin_of_audio|>")
            audio_offsets.append(len(tokens))
            tokens += [0] * num_tokens
            tokens += self.tokenizer.encode("<|end_of_audio|>")
            audio_length.append(num_tokens)

        if not audios:
            raise ValueError("Empty audio or load failed.")

        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")
        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": torch.cat(audios, dim=0),
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch

    def transcribe(self, audio_input: Union[str, Path, torch.Tensor], max_new_tokens: int = 128) -> str:
        batch = self.build_prompt(audio_input)
        
        tokens = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        audios = batch["audios"].to(self.device)
        
        # Determine dtype based on device (bfloat16 for cuda, float32 for cpu usually)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(dtype),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        
        prompt_len = tokens.size(1)

        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        return transcript
