"""Wav2Vec2 phoneme extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def normalize_phoneme_sequence(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("|", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .replace("<s>", "")
    )


@dataclass
class Wav2Vec2PhonemeExtractor:
    model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    device: str = "auto"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
        from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2CTCTokenizer
        self._torch = torch
        self._Wav2Vec2Processor = Wav2Vec2Processor
        self._Wav2Vec2ForCTC = Wav2Vec2ForCTC

        # Fix for bug:
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)

        print(type(tokenizer), tokenizer)

        if self.device == "auto":
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.processor = self._Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.model = self._Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def phonemize_audio_path(self, audio_path: str | Path) -> str:
        import librosa

        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        inputs = self.processor(
            wav,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with self._torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
            pred_ids = self._torch.argmax(logits, dim=-1)

        decoded = self.processor.batch_decode(pred_ids)[0]
        return normalize_phoneme_sequence(decoded)
