import wave
import numpy as np
from faster_whisper import WhisperModel


class ASR:
    def __init__(self, model_size="small", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def load_wav(self, path):
        with wave.open(str(path), "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return a

    def transcribe(self, path):
        audio = self.load_wav(path)
        segs, _ = self.model.transcribe(audio, language="vi", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
