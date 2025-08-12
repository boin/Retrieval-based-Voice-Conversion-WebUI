from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl


class ConvertParams(BaseModel):
    model: str = Field(..., description="model ckpt file name or absolute path")
    spk_id: int = Field(..., ge=0, description="speaker id in model")
    audio_url: Optional[HttpUrl] = Field(None, description="optional URL to fetch audio")
    f0_up_key: int = 0
    f0_method: str = Field("rmvpe", description="pm|harvest|crepe|rmvpe")
    index_path: Optional[str] = None
    index_rate: float = Field(1.0, ge=0, le=1)
    filter_radius: int = 3
    resample_sr: int = 48000
    rms_mix_rate: float = Field(1.0, ge=0, le=1)
    protect: float = Field(0.33, ge=0, le=0.5)
    loudnorm: float = Field(-26, description="LUFS target, 0 to skip")
    return_format: str = Field(
        "wav",
        description="Deprecated/ignored: service returns WAV PCM stream only; clients handle post-processing",
    )


class ConvertResponse(BaseModel):
    code: int
    message: str
    sr: Optional[int] = None
    timings: Optional[dict] = None
    # The audio is returned as a binary stream; when described as JSON, we can include an URL instead.
    result_url: Optional[str] = None
