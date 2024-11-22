import io
import logging
import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from resemble_enhance.enhancer.inference import enhance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


def do_post_process(audio: np.ndarray, sr: int) -> (int, bytes):
    """
    Denoise and enhance using resemble-enhance, normalize the audio by pyloudnorm
    Args:
        audio (np.ndarray): wave audio nparray
        sr (int): sample rate
    Returns:
        int: new sample rate
        bytes: post-processed wav bytes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 转换为 PyTorch 张量，并移动到目标设备
    audio_tensor = torch.from_numpy(audio).float()
    # 如果音频是单声道，调整为单通道
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # 使用 enhance 函数进行去噪和增强，增强函数在 GPU 上运行
    wav, new_sr = enhance(audio_tensor.mean(dim=0), sr, device=device)
    np_wav = wav.cpu().numpy()  # 将结果移回 CPU

    # normalize
    meter = pyloudnorm.Meter(new_sr)  # 创建 BS.1770 meter
    loudness = meter.integrated_loudness(np_wav)
    np_wav = pyloudnorm.normalize.peak(np_wav, -1.0)
    np_wav = pyloudnorm.normalize.loudness(np_wav, loudness, -23.0)

    # 设置目标采样率
    target_sample_rate = 48000
    # 使用 librosa 进行重采样
    np_wav = librosa.resample(np_wav, orig_sr=new_sr, target_sr=target_sample_rate)

    return target_sample_rate, np_wav


# 定义 FastAPI 路由
@app.post("/enhance")
async def process_audio(file: UploadFile = File(...)):
    """
    Endpoint for processing uploaded audio file.
    Args:
        file (UploadFile): Uploaded audio file
        sample_rate (int): Sampling rate of the audio
    Returns:
        FileResponse: Processed audio file
    """
    try:
        logger.info(f"Processing audio file with size: {file.size}")

        # 读取音频文件并解析为 NumPy 数组
        audio_data, sr = sf.read(file.file)

        # 调用处理函数
        new_sr, processed_audio = do_post_process(audio_data, sr)

        # 将处理后的音频数据转换为 BytesIO 对象
        processed_audio_stream = io.BytesIO()
        sf.write(processed_audio_stream, processed_audio, new_sr, format="WAV")
        processed_audio_stream.seek(0)  # Rewind the stream to the beginning

        # 返回音频流
        return StreamingResponse(processed_audio_stream, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# 运行方法
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
