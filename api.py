import io
import logging
import mimetypes
import random
import shutil
import tempfile
import time
import warnings
import wave
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn

# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
# import gradio as gr
from dotenv import load_dotenv

# logging.basicConfig(level=logging.DEBUG)
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

warnings.simplefilter(action="ignore", category=FutureWarning)

# logging.getLogger("numba").setLevel(logging.WARNING)
# logging.getLogger("markdown_it").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
logger.info(i18n)

load_dotenv()
config = Config()
vc = VC(config)


# cache remote pth to local for performance
def load_pth(sid):
    local_pth = Path(f"./assets/weights/{sid}.pth")
    if not local_pth.exists():
        logger.info(f"Local cache for {sid}.pth not found.")
        remote_pth = Path(f"./assets-remote/weights/{sid}.pth")
        if remote_pth.exists():
            shutil.copyfile(remote_pth, local_pth)
            logger.info(f"Successfully cached {sid}.pth to local.")
        else:
            logger.warning(f"Remote file {sid}.pth does not exist. Unable to cache.")
    return sid + ".pth"


def do_post_process(audio: io.BytesIO) -> io.BytesIO:
    re_endpoint = "http://127.0.0.1:8000/enhance"
    audio_file = {"file": audio}
    try:
        response = requests.post(re_endpoint, files=audio_file)
        if response.status_code == 200:
            logger.info("Post process done. file size: %d", len(response.content))
            return io.BytesIO(response.content)
        else:
            logger.error("Post process error:", response.json())
    except requests.exceptions.RequestException as e:
        logger.error("Post Request failed: %s", str(e))
        return audio


def do_vc(
    sid,
    vc_input3,
    vc_transform0,
    seed,
):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # print('setting all random seed to', seed)
    vc.get_vc(load_pth(sid))
    output = vc.vc_single(
        0,  # float (numeric value between 0 and 2333) in '请选择说话人id' Slider component
        vc_input3,  # str (filepath on your computer (or URL) of file) in '输入待处理音频文件' File component
        vc_transform0,  # float  in '变调(整数, 半音数量, 升八度12降八度-12)' Number component
        None,  # str (filepath on your computer (or URL) of file) in 'F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调' File component
        "rmvpe",  # str  in '选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU' Radio component
        None,  # str  in '特征检索库文件路径,为空则使用下拉的选择结果' Textbox component
        None,  # str (Option from: ['assets/indices/HM_IVF211_Flat_nprobe_1_HM_v2.index', 'assets/indices/LMC1_IVF569_Flat_nprobe_1_LMC1_v2.index', 'assets/indices/LMC_IVF569_Flat_nprobe_1_LMC_v2.index', 'assets/indices/ZSYD_1_IVF1509_Flat_nprobe_1_ZSYD_1_v2.index', 'assets/indices/ZSYD_2_IVF819_Flat_nprobe_1_ZSYD_2_v2.index', 'assets/indices/guliang-LHY_IVF778_Flat_nprobe_1_guliang-LHY_v2.index', 'assets/indices/hufa-DRE_IVF825_Flat_nprobe_1_hufa-DRE_v2.index', 'assets/indices/hufa-DRE_IVF829_Flat_nprobe_1_hufa-DRE_v2.index', 'assets/indices/longwang_songzhi_IVF816_Flat_nprobe_1_longwang_songzhi_v2.index', 'logs/HM/added_IVF211_Flat_nprobe_1_HM_v2.index']) in '自动检测index路径,下拉式选择(dropdown)' Dropdown component
        0.75,  # float (numeric value between 0 and 1) in '检索特征占比' Slider component
        3,  # float (numeric value between 0 and 7) in '>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音' Slider component
        0,  # float (numeric value between 0 and 48000) in '后处理重采样至最终采样率，0为不进行重采样' Slider component
        0.25,  # float (numeric value between 0 and 1) in '输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络' Slider component
        0.33,  # float (numeric value between 0 and 0.5) in '保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果' Slider component
    )
    # print(output)
    return output


app = FastAPI()


@app.post("/infer_vc")
async def infer_vc(
    actor: str,
    index: int,
    # seed : int,
    post_process: bool = True,
    file: UploadFile = File(...),
):
    time_start = time.time()
    seed = 114514
    # 检查文件类型
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type is None or not mime_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only audio files are accepted."
        )

    wav_io = io.BytesIO()
    # 创建临时文件
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()  # 确保数据写入文件
            logger.info(
                "recieved vc request. actor: %s, size: %d", actor, len(contents)
            )
            # 调用 vc 方法
            result, wav_data = do_vc(actor, temp_file.name, index, seed)
            logger.debug("vc done. cost time: %.03f", time.time() - time_start)
            # 使用 wave 模块写入 WAV 文件
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(wav_data[0])  # 设置采样率
                wav_file.writeframes(wav_data[1].tobytes())  # 写入音频数据

            # 将指针移动到开头，以便后续读取
            wav_io.seek(0)
            if post_process:
                wav_io = do_post_process(wav_io)
                logger.info(
                    "post process done. cost time: %.03f", time.time() - time_start
                )

    except Exception as e:
        logger.error("vc task %s failed: ", e)
        raise HTTPException(status_code=500, detail=str(e))
    if time.time() - time_start > 5:
        logger.warning(
            "vc task %s long time cost: %.03f", actor, time.time() - time_start
        )
    # 返回结果
    return StreamingResponse(
        wav_io,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7866)
