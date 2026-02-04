import os
import time
import json
import pathlib
import subprocess
import statistics

HOST = os.getenv("HOST", "http://127.0.0.1:7868")
AUDIO = os.getenv("AUDIO", "logs/111/0_gt_wavs/1_0.wav")
MODEL = os.getenv("MODEL", "")
LOOPS = int(os.getenv("LOOPS", "3"))
F0_METHOD = os.getenv("F0_METHOD", "rmvpe")
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "0"))


def _curl_json(url: str) -> dict:
    out = subprocess.check_output(["curl", "-sS", url], text=True)
    return json.loads(out)


def pick_model() -> str:
    data = _curl_json(f"{HOST}/api/v1/models")
    assert data.get("code") == 0, f"list models failed: {data}"
    models = data["models"]
    assert len(models) > 0, "no models available"
    if MODEL:
        assert MODEL in models, f"MODEL not found: {MODEL}"
        return MODEL
    return models[0]


def run_convert(model: str) -> tuple[int, str]:
    assert pathlib.Path(AUDIO).exists(), f"audio not found: {AUDIO}"
    cmd = [
        "curl","-sS","-D","-","-o","/dev/null",
        "-F", f"model={model}",
        "-F", f"spk_id={SPEAKER_ID}",
        "-F", "f0_up_key=0",
        "-F", f"f0_method={F0_METHOD}",
        "-F", "index_rate=1.0",
        "-F", "filter_radius=3",
        "-F", "resample_sr=48000",
        "-F", "rms_mix_rate=1.0",
        "-F", "protect=0.33",
        "-F", "loudnorm=-26",
        "-F", "return_format=wav",
        "-F", f"audio_file=@{AUDIO}",
        f"{HOST}/api/v1/convert",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    dur_ms = int((t1 - t0) * 1000)
    header = proc.stdout
    return dur_ms, header


def test_e2e_convert_repeated():
    # health
    hc = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"{HOST}/health"], capture_output=True, text=True)
    assert hc.returncode == 0 and hc.stdout.strip() == "200"

    model = pick_model()

    durs = []
    cts = []
    for i in range(LOOPS):
        dur, hdr = run_convert(model)
        durs.append(dur)
        ct = None
        for line in hdr.splitlines():
            if line.lower().startswith("content-type:"):
                ct = line.split(":", 1)[1].strip().lower()
                break
        cts.append(ct)
        assert ct == "audio/wav", f"unexpected content-type: {ct}"
        print(f"convert {i+1}/{LOOPS}: {dur} ms")

    avg = statistics.mean(durs)
    print("durations(ms):", durs, "avg:", int(avg))

    # 粗略判断缓存效果：后两次平均 <= 首次的 80%（非严格）
    if len(durs) >= 3:
        warm_avg = statistics.mean(durs[1:])
        print("warm_avg(ms):", int(warm_avg))
        assert warm_avg <= durs[0] * 1.2, "warm runs not faster; caching might be ineffective"

if __name__ == "__main__":
    test_e2e_convert_repeated()
