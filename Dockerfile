# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /

# Install dependenceis to add PPAs
RUN apt-get update && \
    apt-get install -y -qq tzdata ffmpeg aria2 software-properties-common curl git git-lfs && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y build-essential python3.10-dev python3.10-venv && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.10


RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d assets/pretrained_v2/ -o D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d assets/pretrained_v2/ -o G40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d assets/pretrained_v2/ -o f0D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d assets/pretrained_v2/ -o f0G40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d assets/uvr5_weights/ -o HP2-人声vocals+非人声instrumentals.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d assets/uvr5_weights/ -o HP5-主旋律人声vocals+其他instrumentals.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d assets/hubert -o hubert_base.pt
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d assets/rmvpe -o rmvpe.pt

RUN python3 -m venv /venv/vc
RUN python3 -m venv /venv/re

RUN /venv/vc/bin/pip install --upgrade pip==24.0
RUN /venv/vc/bin/pip install --no-cache-dir -r /requirements.txt

# ADD --keep-git-dir=true https://github.com/resemble-ai/resemble-enhance.git /opt/resemble-enhance
# RUN python3 -m pip install /opt/resemble-enhance

RUN /venv/re/bin/pip install --upgrade fastapi uvicorn pyloudnorm numpy soundfile
RUN /venv/re/bin/pip install --upgrade --pre resemble-enhance

EXPOSE 8000
EXPOSE 7866

CMD ["sh", "-c", "/venv/vc/bin/python3 api.py & /venv/re/bin/uvicorn api-re:app --host 0.0.0.0 --port 8000"]
#CMD ["python3", "api.py"]
