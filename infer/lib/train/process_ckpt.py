import os
import sys
import traceback
from collections import OrderedDict

import torch

from i18n.i18n import I18nAuto

i18n = I18nAuto()


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path):
    try:
        a = torch.load(path, map_location="cpu")
        return "模型信息:%s\n采样率:%s\n模型是否输入音高引导:%s\n版本:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except:
        return traceback.format_exc()


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, "assets/weights/%s" % name)
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:

        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == i18n("是") else 0
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()

def merge4(path1: str, path2: str, alpha1: float, alpha2: float, sr: str, f0: str, name: str, version: str, path3: str = None, path4: str = None, alpha3: float = 0, alpha4: float = 0) -> str:
    """
    合并最多四个不同模型的权重。

    参数：
    - path1, path2, path3, path4: 模型检查点文件的路径。
    - alpha1, alpha2, alpha3, alpha4: 合并模型的权重。
    - sr: 采样率。
    - f0: 基频指示符。
    - name: 输出文件的名称。
    - version: 模型的版本。

    返回：
    - 字符串，指示成功或失败。
    """
    try:
        def extract(ckpt: dict) -> dict:
            model_weights = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {key: value for key, value in model_weights.items() if "enc_q" not in key}
            return opt
        
        weight_root = os.getenv("weight_root")

        # 加载检查点
        ckpt1 = torch.load(f"{weight_root}/{path1}.pth", map_location="cpu")
        ckpt2 = torch.load(f"{weight_root}/{path2}.pth", map_location="cpu")
        ckpt3 = torch.load(f"{weight_root}/{path3}.pth", map_location="cpu") if path3 else None
        ckpt4 = torch.load(f"{weight_root}/{path4}.pth", map_location="cpu") if path4 else None

        # 从检查点中提取权重
        cfg = ckpt1["config"]
        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]
        if ckpt3:
            ckpt3 = extract(ckpt3) if "model" in ckpt3 else ckpt3["weight"]
        if ckpt4:
            ckpt4 = extract(ckpt4) if "model" in ckpt4 else ckpt4["weight"]

        # 确保模型架构相同
        model_keys = sorted(list(ckpt1.keys()))
        for ckpt in [ckpt2, ckpt3, ckpt4]:
            if ckpt and sorted(list(ckpt.keys())) != model_keys:
                return "Fail to merge the models. The model architectures are not the same."

        # 归一化 alpha 值
        total_alpha = alpha1 + alpha2 + alpha3 + alpha4
        alpha1 /= total_alpha
        alpha2 /= total_alpha
        alpha3 /= total_alpha
        alpha4 /= total_alpha

        # 合并权重
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0], ckpt3[key].shape[0] if ckpt3 else float('inf'), ckpt4[key].shape[0] if ckpt4 else float('inf'))
                opt["weight"][key] = (
                    alpha1 * ckpt1[key][:min_shape0].float()
                    + alpha2 * ckpt2[key][:min_shape0].float()
                    + (alpha3 * ckpt3[key][:min_shape0].float() if ckpt3 else 0)
                    + (alpha4 * ckpt4[key][:min_shape0].float() if ckpt4 else 0)
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * ckpt1[key].float()
                    + alpha2 * ckpt2[key].float()
                    + (alpha3 * ckpt3[key].float() if ckpt3 else 0)
                    + (alpha4 * ckpt4[key].float() if ckpt4 else 0)
                ).half()

        # 添加额外信息到合并后的模型
        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == i18n("是") else 0
        opt["version"] = version
        opt["info"] = name

        # 保存合并后的模型
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except Exception as e:
        return traceback.format_exc()
