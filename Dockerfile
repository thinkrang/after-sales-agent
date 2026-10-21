FROM python:3.12-slim

# PYTHONUNBUFFERED：日志实时输出，不然 docker logs 看不到进度
# 注意：这里没有 PIP_NO_CACHE_DIR。
# 改用 BuildKit 的 cache mount 缓存 pip 下载，缓存不落在镜像层里，
# 既能让重建变快（改 requirements 时不用重新下载全部依赖），镜像也不会变大。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 国内网络下 pypi.org 和 download.pytorch.org 经常超时，默认走阿里云镜像。
# 海外环境可以用 --build-arg 覆盖：
#   docker build --build-arg PIP_INDEX=https://pypi.org/simple \
#                --build-arg TORCH_FIND_LINKS=https://download.pytorch.org/whl/cpu .
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
ARG TORCH_VERSION=2.6.0+cpu

# 显式锁死 +cpu 版本，两个原因：
#   1. 不锁的话 pip 会从 PyPI 挑版本号更高的 CUDA 版（2GB+），镜像直接爆掉
#   2. 这个项目只做 embedding 推理，CPU 版足够，镜像能小 5 倍
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" \
        --find-links ${TORCH_FIND_LINKS} \
        --index-url ${PIP_INDEX}

# 依赖单独一层：只要 requirements.txt 没变，改代码时不用重装依赖
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --index-url ${PIP_INDEX}

COPY . .

EXPOSE 8000

# 健康检查用 /ready 而不是 /health：
# /health 只证明进程活着，/ready 才验证向量库、工单库、会话库都能用。
# 用 /health 的话，依赖没就绪时容器也会被标记健康，流量照样打进来。
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
