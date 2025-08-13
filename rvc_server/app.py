import io
import os
import logging
import threading
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from configs.config import Config
from .registry import LRURegistry

logger = logging.getLogger(__name__)

_hubert_model = None
_hubert_lock = threading.Lock()

def _hubert_provider(config: Config):
    global _hubert_model
    if _hubert_model is not None:
        return _hubert_model
    with _hubert_lock:
        if _hubert_model is None:
            # Lazy import to avoid heavy deps at import time
            from infer.modules.vc.utils import load_hubert
            _hubert_model = load_hubert(config)
    return _hubert_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        # Preload common resources (e.g., Hubert). Optional; errors are logged but won't block service.
        _hubert_provider(_config)
        logger.info("Common resources preloaded (Hubert)")
    except Exception:
        logger.exception("Failed to preload common resources")
    yield
    # shutdown
    try:
        # If there were global resources to release, handle here. Currently none.
        pass
    except Exception:
        logger.exception("Failed during app shutdown cleanup")


app = FastAPI(title="RVC FastAPI Service", version="0.1.0", lifespan=lifespan)

# load env for local dev
load_dotenv()

# reduce access log noise for /health and /docs
class UvicornPathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # uvicorn.access uses: "%s - \"%s %s HTTP/%s\" %d"
            args = record.args
            if not args or len(args) < 5:
                return True
            method = args[1]
            path = args[2]
            if isinstance(path, bytes):
                path = path.decode(errors="ignore")
            if path in ("/health", "/docs", "/openapi.json"):
                return False
        except Exception:
            # never break logging
            return True
        return True

logging.getLogger("uvicorn.access").addFilter(UvicornPathFilter())


# create global config and registry
_config = Config()
_registry = LRURegistry(
    _config,
    max_models=int(os.getenv("RVC_MAX_MODELS", 10)),  # default 10 models
    ttl_seconds=int(os.getenv("RVC_MODEL_TTL", 1800)),  # default 30 minutes
    hubert_provider=_hubert_provider,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/cache/clear")
def clear_cache():
    try:
        _registry.clear()
        return {"code": 0, "message": "cleared"}
    except Exception:
        logger.exception("cache clear failed")
        return JSONResponse(status_code=500, content={"code": 1, "message": "cache clear failed"})


def _scan_models() -> List[str]:
    weight_root = os.getenv("weight_root")
    if not weight_root or not os.path.isdir(weight_root):
        return []
    models: List[str] = []
    # 取weight_root根目录下的非隐藏的.pth文件
    for fname in os.listdir(weight_root):
        full = os.path.join(weight_root, fname)
        if (os.path.isfile(full) and fname.endswith(".pth") and not fname.startswith(".")):
            models.append(fname)
    return sorted(models)


def _scan_indices() -> List[str]:
    indices: List[str] = [""]
    for env_name in ("index_root", "outside_index_root"):
        base = os.getenv(env_name)
        if not base or not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name and not name.startswith("."):
                    indices.append(os.path.join(root, name))
    # de-dup and sort with empty first
    uniq = []
    seen = set()
    for p in indices:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


@app.get("/api/v1/refresh")
def refresh():
    try:
        models = _scan_models()
        indices = _scan_indices()
        return {"code": 0, "models": models, "indices": indices}
    except Exception:
        logger.exception("refresh failed")
        return JSONResponse(status_code=500, content={"code": 1, "message": "refresh failed"})


@app.post("/api/v1/convert")
async def convert(
    model: str = Form(...),
    spk_id: int = Form(...),
    f0_up_key: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_path: Optional[str] = Form(None),
    index_rate: float = Form(1.0),
    filter_radius: int = Form(3),
    resample_sr: int = Form(48000),
    rms_mix_rate: float = Form(1.0),
    protect: float = Form(0.33),
    loudnorm: float = Form(-26),
    return_format: str = Form("wav"),
    audio_file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
):
    """Single-request model+infer API.
    Either `audio_file` or `audio_url` must be provided.
    Returns audio as binary stream in the requested format when possible (wav/flac return as waveform; mp3/m4a routed through existing wav->encode in vc.vc_single path).
    """
    # fetch or persist audio to a temp file path
    if not audio_file and not audio_url:
        return JSONResponse(status_code=400, content={"code": 1, "message": "audio_file or audio_url required"})

    try:
        if audio_file is not None:
            # save to a temp file
            suffix = os.path.splitext(audio_file.filename or "input.wav")[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
                content = await audio_file.read()
                tmpf.write(content)
                input_path = tmpf.name
        else:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(audio_url)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                    tmpf.write(resp.content)
                    input_path = tmpf.name

        session = _registry.get(model)
        info, result = session.convert(
            spk_id=spk_id,
            input_audio_path=input_path,
            f0_up_key=int(f0_up_key),
            f0_file=None,
            f0_method=f0_method,
            index_path=index_path,
            index_rate=float(index_rate),
            filter_radius=int(filter_radius),
            resample_sr=int(resample_sr),
            rms_mix_rate=float(rms_mix_rate),
            protect=float(protect),
            loudnorm=float(loudnorm),
        )

        if result is None or result[0] is None or result[1] is None:
            return JSONResponse(status_code=500, content={"code": 2, "message": info})

        sr, audio_np = result
        # for wav/flac we can return PCM directly as wav bytes
        import soundfile as sf
        buf = io.BytesIO()
        fmt = "wav" if return_format in (None, "", "wav", "flac") else "wav"
        # Always write WAV bytes here; the VC.vc_single already handled mp3/m4a in batch path. For single, we use wav.
        sf.write(buf, audio_np, sr, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception:
        logger.exception("convert failed")
        return JSONResponse(status_code=500, content={"code": 3, "message": "convert failed"})
    finally:
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            logger.exception("cleanup temp file failed")
