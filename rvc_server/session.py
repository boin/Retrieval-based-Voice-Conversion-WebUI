import os
import threading
import logging
from typing import Optional, Tuple

import numpy as np
from configs.config import Config
import torch

from infer.modules.vc.modules import VC
from infer.modules.vc.utils import get_index_path_from_model

logger = logging.getLogger(__name__)


class ModelSession:
    """
    A session encapsulates one loaded model (ckpt) with its own VC instance and pipeline.
    Thread-safe at the session level; control concurrency using the provided lock/semaphore at a higher level if needed.
    """

    def __init__(self, model_ckpt: str, config: Config, hubert_provider=None):
        self.model_ckpt = model_ckpt
        self.config = config
        self.hubert_provider = hubert_provider
        self.vc = VC(config)
        self._lock = threading.Lock()
        self._loaded = False
        # metrics removed: convert_count, wait_time_total
        # per-model concurrency control (default: 1 -> serialize same-model inference)
        try:
            per_model_cc = int(os.getenv("RVC_PER_MODEL_CONCURRENCY", "1"))
        except Exception:
            per_model_cc = 1
        per_model_cc = max(1, per_model_cc)
        self._sem = threading.Semaphore(per_model_cc)

    def load(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            # Emulate VC.get_vc loading path but without touching global mutable state outside this session
            sid = self.model_ckpt
            # Prefer the provided path if it is absolute or already exists; otherwise, resolve under weight_root
            if os.path.isabs(sid) or os.path.exists(sid):
                person = sid
            else:
                root = os.getenv("weight_root") or ""
                person = os.path.join(root, sid)
            person = os.path.normpath(person)
            logger.info(f"[ModelSession] Loading model: {person}")

            cpt = torch.load(person, map_location="cpu")
            tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")

            from infer.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            )

            synthesizer_class = {
                ("v1", 1): SynthesizerTrnMs256NSFsid,
                ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                ("v2", 1): SynthesizerTrnMs768NSFsid,
                ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
            }

            net_g = synthesizer_class.get((version, if_f0), SynthesizerTrnMs256NSFsid)(
                *cpt["config"], is_half=self.config.is_half
            )
            # compatibility with original VC.get_vc
            del net_g.enc_q
            net_g.load_state_dict(cpt["weight"], strict=False)
            net_g.eval().to(self.config.device)
            net_g = net_g.half() if self.config.is_half else net_g.float()

            # attach to this session's VC
            self.vc.cpt = cpt
            self.vc.tgt_sr = tgt_sr
            self.vc.if_f0 = if_f0
            self.vc.version = version
            self.vc.net_g = net_g
            self.vc.pipeline = None  # will be created lazily
            if self.hubert_provider is not None:
                self.vc.hubert_model = self.hubert_provider(self.config)
            else:
                # fallback: lazy import to avoid hard dependency at import time
                from infer.modules.vc.utils import load_hubert
                self.vc.hubert_model = load_hubert(self.config)

            # default index path based on model name
            self.default_index = get_index_path_from_model(sid)
            logger.info(f"[ModelSession] Default index: {self.default_index}")

            self._loaded = True

    def convert(
        self,
        spk_id: int,
        input_audio_path: str,
        *,
        f0_up_key: int,
        f0_file: Optional[str],
        f0_method: str,
        index_path: Optional[str],
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
        loudnorm: float,
    ) -> Tuple[str, Tuple[Optional[int], Optional[np.ndarray]]]:
        logger.debug(f"[ModelSession] Waiting semaphore for model={self.model_ckpt}")
        self._sem.acquire()
        logger.debug(f"[ModelSession] Acquired semaphore for model={self.model_ckpt}")
        try:
            self.load()
            # ensure pipeline created for this session
            if self.vc.pipeline is None:
                from infer.modules.vc.pipeline import Pipeline
                self.vc.pipeline = Pipeline(self.vc.tgt_sr, self.config)

            # prefer explicit index path, otherwise session default
            file_index = index_path or self.default_index or ""

            # call vc_single directly; keeps behavior parity
            result = self.vc.vc_single(
                spk_id,
                input_audio_path,
                f0_up_key,
                f0_file,
                f0_method,
                file_index,
                file_index,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                loudnorm,
            )
            return result
        finally:
            self._sem.release()
            logger.debug(f"[ModelSession] Released semaphore for model={self.model_ckpt}")

    def unload(self):
        with self._lock:
            try:
                if getattr(self.vc, "net_g", None) is not None:
                    del self.vc.net_g
                if getattr(self.vc, "pipeline", None) is not None:
                    del self.vc.pipeline
                if getattr(self, "default_index", None) is not None:
                    del self.default_index
            except Exception:
                logger.exception("Error while unloading model session")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.exception("Error while emptying CUDA cache")
