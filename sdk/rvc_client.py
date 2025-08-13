"""
音频转换模块（Gradio UI）

该模块提供了调用 http://ttd-stage:7867/ RVC能力的集成GRadio UI，避免客户端自己实现UI。 

使用方法：
    from rvc_client import create_rvc_ui # 对外只保留 create_rvc_ui
    
"""

import logging
import os
from random import choice
import time
from typing import List, Optional, Tuple

import tempfile
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 服务端点
SERVER_ENDPOINT = "http://ttd-stage:7866"  # 这个外部地址保持不变，api服务会发布到 ttd-stage:7867
SESSION = requests.Session()


class VoiceChangeResult(BaseModel):
    sid: float
    param_1: float
    param_2: float
    file_index: str
    file_index2: str


class RvcConfig(BaseModel):
    output_path: Optional[str] = None
    sid: float = 0
    f0_up_key: float = 0
    f0_method: str = "rmvpe"
    file_index: Optional[str] = None
    file_index2: Optional[str] = None
    index_rate: float = 1.0
    filter_radius: float = 3.0
    resample_sr: float = 48000
    rms_mix_rate: float = 1.0
    protect: float = 0.33
    loudnorm: float = -26
    pth_name: str = ""


def refresh_resources() -> Tuple[dict, dict, List[str]] | None:
    """
    刷新并获取可用的参考音模型和特征索引

    Returns:
        Tuple[dict, dict, List[str]]: (models_comp, indices_comp, indices)
    """
    try:
        r = SESSION.get(f"{SERVER_ENDPOINT}/api/v1/refresh", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"refresh failed: {data}")
        models = list(data.get("models", []))
        indices = list(data.get("indices", []))
        logger.info(f"刷新RVC资源成功，模型数量: {len(models)}, 索引数量: {len(indices)}")
        models_comp = {"__type__":"update", "choices": models, "value": models[0] or ""}
        indices_comp = {"__type__":"update", "choices": indices, "value": indices[0] or ""}
        return models_comp, indices_comp, indices
    except Exception as e:
        logger.exception(f"刷新RVC资源失败: {e}")
        return None


def find_matching_index(model_name: str, indices: List[str] | None) -> str:
    """
    根据模型名称查找匹配的特征索引

    Args:
        model_name (str): 模型名称，例如 "枫原万叶.pth"
        indices (List[str], optional): 特征索引列表。如果不提供，将返回空字符串
        
    Returns:
        str: 匹配的特征索引路径，如果没有找到匹配的索引，则返回空字符串
    """
    base_name = os.path.splitext(os.path.basename(model_name))[0]
    try:
        if not indices:
            return ""
        for index in indices:
            if base_name in index:
                return index
        return ""
    except Exception as e:
        logger.error(f"查找匹配的特征索引失败: {e}")
        return ""


def convert_audio(input_audio_path: str, rvc_conf: RvcConfig, output_path: str | None = None) -> str:
    """
    调用 http://ttd-stage:7867/ 的 FastAPI /api/v1/convert 接口转换音频

    Args:
        input_audio_path (str): 输入音频文件路径
        output_path (str, optional): 输出文件路径。如果不提供，将使用输入文件名加后缀
        rvc_conf (RvcConfig): RVC配置参数
            sid (float, optional): 说话人ID。默认为0
            f0_up_key (float, optional): 变调参数，半音数量。默认为0
            f0_method (str, optional): 音高提取算法，可选值：pm, harvest, crepe, rmvpe。默认为rmvpe
            file_index (str, optional): 特征检索库文件路径。默认为空字符串
            file_index2 (str, optional): 自动检测的特征索引路径。默认为空字符串
            index_rate (float, optional): 检索特征占比。默认为1.0
            filter_radius (float, optional): 中值滤波半径。默认为3.0
            resample_sr (float, optional): 重采样采样率。默认为48000
            rms_mix_rate (float, optional): 音量包络融合比例。默认为1.0
            protect (float, optional): 保护清辅音和呼吸声的强度。默认为0.33
            loudnorm (float, optional): 音量标准化LUFS值。默认为-26
            pth_name (str, optional): 参考音模型名称。如果提供，将自动查找匹配的特征索引
    Returns:
        str: 转换后的音频文件路径
    """
    if not input_audio_path or not isinstance(input_audio_path, str):
        raise ValueError("输入音频文件路径必须是字符串")
    if not os.path.exists(input_audio_path):
        raise ValueError(f"输入音频文件不存在: {input_audio_path}")

    start_time = time.time()

    if not output_path:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # 检查输入文件可读
    try:
        with open(input_audio_path, "rb") as f:
            file_size = os.path.getsize(input_audio_path)
    except Exception as e:
        logger.error(f"无法读取输入文件: {e}")
        raise ValueError(f"无法读取输入文件: {e}")

    logger.info(
        f"开始RVC: {os.path.basename(input_audio_path)}，模型: {rvc_conf.pth_name}, 索引: {rvc_conf.file_index}, 参数: sid={rvc_conf.sid}, f0_up_key={rvc_conf.f0_up_key}, f0_method={rvc_conf.f0_method}"
    )

    form = {
        "model": rvc_conf.pth_name or "",
        "spk_id": int(rvc_conf.sid),
        "f0_up_key": int(rvc_conf.f0_up_key),
        "f0_method": rvc_conf.f0_method,
        "index_path": rvc_conf.file_index or rvc_conf.file_index2 or "",
        "index_rate": str(float(rvc_conf.index_rate)),
        "filter_radius": str(int(rvc_conf.filter_radius)),
        "resample_sr": str(int(rvc_conf.resample_sr)),
        "rms_mix_rate": str(float(rvc_conf.rms_mix_rate)),
        "protect": str(float(rvc_conf.protect)),
        "loudnorm": str(float(rvc_conf.loudnorm)),
        "return_format": "wav",
    }
    with open(input_audio_path, "rb") as f:
        files = {"audio_file": (os.path.basename(input_audio_path), f, "audio/wav")}
        r = SESSION.post(f"{SERVER_ENDPOINT}/api/v1/convert", data=form, files=files, timeout=300)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise Exception(f"音频转换失败: {err}")
    with open(output_path, "wb") as dst_file:
        dst_file.write(r.content)
    logger.info(
        f"音频转换完成：{time.time() - start_time:.2f} 秒，输入{file_size} 字节，输出 {os.path.getsize(output_path)} 字节"
    )
    return output_path


def create_rvc_ui(rvc_src_audio):
    """创建一个通用RVC组件"""
    import gradio as gr
    indices_state = gr.State([])
    rvc_conf = gr.State(RvcConfig())

    with gr.Blocks() as rvc_ui:
        # RVC模型选择
        with gr.Row():
            rvc_models = gr.Dropdown(label="参考音模型", choices=[], interactive=True)
            # 音高偏移设置
            pitch_shift = gr.Slider(label="变调 音高偏移（半音数）", minimum=-12, maximum=12, value=0, step=1, interactive=True)
            rvc_indices = gr.Dropdown(label="特征索引", choices=[], interactive=True)
            rvc_models.change(find_matching_index, inputs=[rvc_models, indices_state], outputs=rvc_indices)
        with gr.Row():
            rvc_generate_button = gr.Button("生成RVC输出", variant="primary")
            rvc_refresh_button = gr.Button("\U000027F3", variant="secondary", min_width=50, scale=0)
            rvc_refresh_button.click(refresh_resources, outputs=[rvc_models, rvc_indices, indices_state])

        with gr.Row():
            # RVC生成的音频
            rvc_audio = gr.Audio(interactive=False, autoplay=True, show_download_button=False)

        # RVC 设置
        with gr.Row():
            with gr.Accordion("RVC设置", open=False):
                # 响度标准化
                loudnorm = gr.Slider(label="loudnorm到指定的LUFS（0为不调整）", minimum=-40, maximum=-10, value=-26, step=1, interactive=True)
                # 重采样采样率
                resample_sr = gr.Slider(label="重采样采样率", minimum=0, maximum=48000, value=48000, step=100, interactive=True)
                # RMS混合率
                rms_mix_rate = gr.Slider(label="输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络", minimum=0, maximum=1, value=1, step=0.01, interactive=True)
                # 保护清辅音和咽音
                protect_option = gr.Slider(label="保护清辅音和咽音，防止出现artifact，启用会牺牲转换度", minimum=0, maximum=0.5, value=0.33, step=0.01, interactive=True)
                # 滤波器半径
                filter_radius = gr.Slider(label=">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音", minimum=0, maximum=7, value=3, step=1, interactive=True)
                # 检索特征占比
                index_rate = gr.Slider(label="检索特征占比", minimum=0, maximum=1, value=1, step=0.01, interactive=True)

        rvc_generate_button.click(convert_audio, inputs=[rvc_src_audio, rvc_conf], outputs=rvc_audio)

        gr.on(
            triggers=[
                pitch_shift.change, loudnorm.change, resample_sr.change, rvc_indices.change,
                index_rate.change, filter_radius.change, rms_mix_rate.change, protect_option.change,
                rvc_models.change
            ],
            fn=lambda pitch_shift, loudnorm, rvc_indices, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect_option, rvc_models: RvcConfig(
                f0_up_key=pitch_shift,
                loudnorm=loudnorm,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect_option,
                filter_radius=filter_radius,
                index_rate=index_rate,
                file_index2=rvc_indices,
                pth_name=rvc_models
            ),
            inputs=[
                pitch_shift, loudnorm, rvc_indices, index_rate, filter_radius,
                resample_sr, rms_mix_rate, protect_option, rvc_models
            ],
            outputs=[rvc_conf]
        )

        # 初始化一次资源
        rvc_ui.load(fn=refresh_resources, outputs=[rvc_models, rvc_indices, indices_state])

    components = {
        "rvc_ui": rvc_ui,
        "rvc_models": rvc_models,
        "rvc_indices": rvc_indices,
        "rvc_audio": rvc_audio,
        "rvc_refresh_btn": rvc_refresh_button,
        "rvc_pitch_shift": pitch_shift,
        "rvc_loudnorm": loudnorm,
        "rvc_resample_sr": resample_sr,
        "rvc_rms_mix_rate": rms_mix_rate,
        "rvc_protect_option": protect_option,
        "rvc_filter_radius": filter_radius,
        "rvc_index_rate": index_rate,
        "rvc_generate_btn": rvc_generate_button,
        "rvc_indices_state": indices_state,
        "rvc_conf": rvc_conf
    }
    component_list = list(components.values())

    return components, component_list
