import os
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


class ASR:
    """Vietnamese ASR using Gipformer (sherpa-onnx transducer).

    The constructor keeps the old (model_size, device, compute_type) signature
    for compatibility, but the actual model is resolved from ``model_dir`` or
    the bundled ``models/gipformer`` directory.
    """

    def __init__(
        self,
        model_size=None,
        device="cuda",
        compute_type="float16",
        model_dir=None,
        num_threads=4,
        decoding_method="modified_beam_search",
        quantize="int8",
    ):
        self.device = device
        self.quantize = quantize
        self.model_dir = self._resolve_model_dir(model_size, model_dir)
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.recognizer = self._build_recognizer()

    def _resolve_model_dir(self, model_size, model_dir):
        candidates = []
        if model_dir is not None:
            candidates.append(Path(model_dir))
        if isinstance(model_size, (str, Path)) and Path(model_size).exists():
            candidates.append(Path(model_size))
        env_dir = os.environ.get("GIPFORMER_MODEL_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        # bundled model next to the project root
        candidates.append(Path(__file__).resolve().parents[1] / "models" / "gipformer")
        # development cache location
        candidates.append(
            Path.home() / ".cache" / "huggingface" / "hub" /
            "models--g-group-ai-lab--gipformer1.5-68M-rnnt" /
            "snapshots" / "dd9227dcd8705c13f33bdbe59728d546ab94480f"
        )
        for c in candidates:
            if c is not None and c.is_dir() and (c / "tokens.txt").is_file():
                return c
        raise FileNotFoundError(
            "Gipformer model directory not found. Set GIPFORMER_MODEL_DIR or "
            "bundle the model under models/gipformer."
        )

    def _file(self, name):
        # prefer quantized files when requested
        if self.quantize == "int8":
            quantized = self.model_dir / name.replace(".onnx", ".int8.onnx")
            if quantized.is_file():
                return str(quantized)
        return str(self.model_dir / name)

    def _build_recognizer(self):
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._file("encoder.onnx"),
            decoder=self._file("decoder.onnx"),
            joiner=self._file("joiner.onnx"),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method=self.decoding_method,
            provider="cpu",
        )

    @staticmethod
    def load_wav(path):
        with wave.open(str(path), "rb") as w:
            sample_rate = w.getframerate()
            n_channels = w.getnchannels()
            sample_width = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width: {sample_width}")
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return audio, sample_rate

    def transcribe(self, path):
        audio, sample_rate = self.load_wav(path)
        if sample_rate != 16000:
            # simple nearest-neighbour resampling fallback
            ratio = 16000 / sample_rate
            new_len = int(len(audio) * ratio)
            idx = (np.arange(new_len) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, len(audio) - 1)
            audio = audio[idx]
            sample_rate = 16000
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_streams([stream])
        text = stream.result.text.strip()
        return text.lower()