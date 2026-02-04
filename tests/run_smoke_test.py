import io
import os
import sys
import time
import threading
import numpy as np
import soundfile as sf

from fastapi.testclient import TestClient

# ensure repo root on sys.path when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rvc_server.app as app_module


class FakeSession:
    def __init__(self, delay=0.2):
        self.loaded = False
        self.delay = delay
        self._sem = threading.Semaphore(1)
        self._active = 0
        self.max_active = 0
        # metrics removed

    def load(self):
        self.loaded = True

    def convert(
        self,
        spk_id: int,
        input_audio_path: str,
        *,
        f0_up_key: int,
        f0_file,
        f0_method: str,
        index_path,
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
        loudnorm: float,
    ):
        self._sem.acquire()
        try:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            time.sleep(self.delay)
            # metrics removed
            # return 0.5s of 440Hz sine wave at 16k
            sr = 16000
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            wav = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            return ("ok", (sr, wav))
        finally:
            self._active -= 1
            self._sem.release()


class FakeRegistry:
    def __init__(self):
        self.session = FakeSession()
        # metrics removed

    def get(self, model_ckpt: str):
        return self.session

    def clear(self):
        self.session = FakeSession()

    # metrics removed


# monkeypatch the registry
app_module._registry = FakeRegistry()
client = TestClient(app_module.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_convert_success():
    # create a small input wav to send as file
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)

    files = {"audio_file": ("input.wav", buf, "audio/wav")}
    data = {
        "model": "mock.pth",
        "spk_id": "0",
        "f0_up_key": "0",
        "f0_method": "rmvpe",
        "index_rate": "1.0",
        "filter_radius": "3",
        "resample_sr": "48000",
        "rms_mix_rate": "1.0",
        "protect": "0.33",
        "loudnorm": "-26",
        "return_format": "wav",
    }

    resp = client.post("/api/v1/convert", data=data, files=files)
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")
    audio_bytes = resp.content
    assert len(audio_bytes) > 1000  # some bytes returned


def test_cache_clear():
    resp = client.post("/api/v1/cache/clear")
    assert resp.status_code == 200
    assert resp.json()["code"] == 0


# metrics test removed


def test_concurrent_serialization():
    # send 3 concurrent requests; FakeSession enforces serial via semaphore
    sr = 16000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf_bytes = buf.getvalue()

    def do_req():
        files = {"audio_file": ("input.wav", io.BytesIO(buf_bytes), "audio/wav")}
        data = {
            "model": "mock.pth",
            "spk_id": "0",
            "f0_up_key": "0",
            "f0_method": "rmvpe",
            "index_rate": "1.0",
            "filter_radius": "3",
            "resample_sr": "48000",
            "rms_mix_rate": "1.0",
            "protect": "0.33",
            "loudnorm": "-26",
            "return_format": "wav",
        }
        r = client.post("/api/v1/convert", data=data, files=files)
        assert r.status_code == 200

    threads = [threading.Thread(target=do_req) for _ in range(3)]
    t_start = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.time() - t_start
    # Each request waits ~0.2s and serializes, so ~0.6s total; allow tolerance
    assert elapsed >= 0.5, f"elapsed {elapsed} too small, not serialized?"
    # Verify FakeSession recorded max_active == 1 (serialized)
    assert app_module._registry.session.max_active == 1


if __name__ == "__main__":
    # run tests manually without pytest
    test_health()
    test_convert_success()
    test_cache_clear()
    test_concurrent_serialization()
    print("Smoke tests passed.")
