"""下载 Reranker 模型 BAAI/bge-reranker-base。

为什么需要单独一个脚本
----------------------
项目里为了加速启动设了 HF_HUB_OFFLINE=1，
但首次下载模型必须联网，两者冲突。
本脚本会临时解除离线模式再下载，下完不影响项目里的离线设置。

用法（在你自己的终端里跑，不要在受限环境里跑）
---------------------------------------------
    python download_reranker.py

如果官方源连不上，脚本会自动改用国内镜像 hf-mirror.com 重试。

模型约 1.1GB，下载完成后 evaluate_rag.py 里的 hybrid_rerank 会自动启用。
"""

import os
import sys

MODEL = "BAAI/bge-reranker-base"
MIRROR = "https://hf-mirror.com"


def try_download(endpoint: str | None) -> bool:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)

    label = endpoint or "huggingface.co"
    print(f"\n尝试从 {label} 下载 {MODEL} ...")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=MODEL,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "sentence_bert_config.json",
                "modules.json",
                "1_Pooling/*",
            ],
        )
        print(f"下载成功：{path}")
        return True
    except Exception as e:
        print(f"失败：{type(e).__name__}: {str(e)[:160]}")
        return False


def main():
    print("=" * 60)
    print(f"下载 Reranker 模型：{MODEL}（约 1.1GB）")
    print("=" * 60)

    if try_download(None):
        print("\n完成。现在可以跑 evaluate_rag.py，hybrid_rerank 会自动启用。")
        return

    print("\n官方源失败，改用国内镜像重试...")
    if try_download(MIRROR):
        print("\n完成（走了镜像）。注意：镜像下载的模型同样可用。")
        print("如果之后启动项目报错，检查 HF_ENDPOINT 是否被永久写进了环境变量。")
        return

    print("\n两个源都失败。可以尝试：")
    print("  1. 确认网络能访问 hf-mirror.com（浏览器打开试试）")
    print("  2. 如果有代理，设置 HTTP_PROXY / HTTPS_PROXY 后重试")
    print("  3. 手动下载后放到缓存目录，参考：")
    print("     https://hf-mirror.com/BAAI/bge-reranker-base/tree/main")
    sys.exit(1)


if __name__ == "__main__":
    main()
