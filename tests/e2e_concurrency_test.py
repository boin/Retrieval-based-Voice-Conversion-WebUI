import os
import json
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

HOST = os.getenv("HOST", "http://127.0.0.1:7868")
AUDIO = os.getenv("AUDIO", "logs/111/0_gt_wavs/1_0.wav")
F0_METHOD = os.getenv("F0_METHOD", "rmvpe")
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "0"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))


def curl_json(url: str) -> dict:
    out = subprocess.check_output(["curl", "-sS", url], text=True)
    return json.loads(out)


def list_models() -> list[str]:
    data = curl_json(f"{HOST}/api/v1/models")
    assert data.get("code") == 0, f"list models failed: {data}"
    return data["models"]


def run_convert(model: str) -> tuple[str, int]:
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
    if proc.returncode != 0:
        raise RuntimeError(f"convert failed for {model}: rc={proc.returncode}, stderr={proc.stderr[:200]}...")
    # quick validate content-type
    header = proc.stdout
    ok = any(line.lower().startswith("content-type:") and "audio/wav" in line.lower() for line in header.splitlines())
    if not ok:
        raise AssertionError(f"unexpected content-type for {model}:\n{header}")
    return model, dur_ms


def main():
    # health
    hc = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"{HOST}/health"], capture_output=True, text=True)
    assert hc.returncode == 0 and hc.stdout.strip() == "200"

    models = list_models()
    print(f"[conc] total models: {len(models)}")
    assert models, "no models"

    # Case A: same model, N concurrent
    same_model = models[0]
    print(f"[conc] Case A: same-model concurrency={CONCURRENCY}, model={same_model}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(run_convert, same_model) for _ in range(CONCURRENCY)]
        results = []
        for fut in as_completed(futs):
            results.append(fut.result())
    t1 = time.time()
    makespan_same = int((t1 - t0) * 1000)
    durs_same = [d for _m, d in results]
    print(f"[conc] A durations(ms): {durs_same}, makespan={makespan_same} ms")

    # Case B: different models, N concurrent (if enough models)
    if len(models) >= CONCURRENCY:
        diff_models = models[:CONCURRENCY]
    else:
        diff_models = [models[i % len(models)] for i in range(CONCURRENCY)]
    print(f"[conc] Case B: diff-model concurrency={CONCURRENCY}, models={diff_models}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(run_convert, m) for m in diff_models]
        results = []
        for fut in as_completed(futs):
            results.append(fut.result())
    t1 = time.time()
    makespan_diff = int((t1 - t0) * 1000)
    durs_diff = [d for _m, d in results]
    print(f"[conc] B durations(ms): {durs_diff}, makespan={makespan_diff} ms")

    # 提示：按设计，同一模型并发会被 session 内部的信号量串行化，而不同模型可以并行。
    print("[conc] Note: 同一模型并发 -> makespan 接近多次串行总和；不同模型并发 -> makespan 接近 max(单次). 以此对比判断是否达到预期。")

if __name__ == "__main__":
    main()
