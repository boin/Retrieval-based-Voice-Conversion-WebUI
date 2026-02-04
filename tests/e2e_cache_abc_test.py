import os
import json
import time
import pathlib
import subprocess
from typing import List, Tuple

HOST = os.getenv("HOST", "http://127.0.0.1:7868")
AUDIO = os.getenv("AUDIO", "logs/111/0_gt_wavs/1_0.wav")
F0_METHOD = os.getenv("F0_METHOD", "rmvpe")
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "0"))
REPEATS = int(os.getenv("REPEATS", "2"))  # 复查次数，默认每个模型再跑2次
MODELS_ENV = os.getenv("MODELS", "")  # 可通过逗号分隔传入 A,B,C


def curl_json(url: str) -> dict:
    out = subprocess.check_output(["curl", "-sS", url], text=True)
    return json.loads(out)


def list_models() -> List[str]:
    data = curl_json(f"{HOST}/api/v1/models")
    assert data.get("code") == 0, f"list models failed: {data}"
    return data["models"]


def run_convert(model: str) -> int:
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
    header = proc.stdout
    ok = any(line.lower().startswith("content-type:") and "audio/wav" in line.lower() for line in header.splitlines())
    if not ok:
        raise AssertionError(f"unexpected content-type for {model}:\n{header}")
    return dur_ms


def pick_abc() -> List[str]:
    if MODELS_ENV.strip():
        models = [m.strip() for m in MODELS_ENV.split(",") if m.strip()]
        assert len(models) >= 3, "MODELS 环境变量至少提供 3 个模型"
        return models[:3]
    models = list_models()
    assert len(models) >= 3, f"需至少 3 个模型，当前 {len(models)}"
    return models[:3]


def main():
    # health
    hc = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"{HOST}/health"], capture_output=True, text=True)
    assert hc.returncode == 0 and hc.stdout.strip() == "200"

    abc = pick_abc()
    print(f"[abc] models: {abc}")

    cold: List[Tuple[str,int]] = []
    warm_map = {m: [] for m in abc}

    # 冷启动各一次
    print("[abc] Cold round: A,B,C 各 1 次（顺序、非并发）")
    for m in abc:
        d = run_convert(m)
        cold.append((m, d))
        print(f"  cold {m}: {d} ms")

    # 间隔一下，避免偶发干扰
    time.sleep(1.0)

    # 热缓存阶段：重复 REPEATS 轮，每轮顺序 A->B->C
    print(f"[abc] Warm rounds: repeats={REPEATS}, 顺序 A->B->C")
    for r in range(1, REPEATS + 1):
        print(f"  round {r}...")
        for m in abc:
            d = run_convert(m)
            warm_map[m].append(d)
            print(f"    warm {m}: {d} ms")

    # 汇总
    print("[abc] Summary:")
    for m, c in cold:
        ws = warm_map[m]
        avg_warm = int(sum(ws) / len(ws)) if ws else -1
        speedup = (c / avg_warm) if (avg_warm > 0) else 0
        print(f"  {m}: cold={c} ms, warm_list={ws}, warm_avg={avg_warm} ms, speedup≈{speedup:.2f}x")

    print("[abc] Note: 期望 warm_avg 明显小于 cold，三模型都应变快；若未生效，请检查 Registry 容量/TTL、是否多进程运行导致跨进程不共享缓存等。")

if __name__ == "__main__":
    main()
