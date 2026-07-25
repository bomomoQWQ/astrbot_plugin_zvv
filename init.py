"""init.py —— 首次运行自动下载模型和图片

自动从指定 URL 下载 ONNX 模型和表情包图片。
也可手动运行: python init.py
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent

# ── 持久化目录（与 main.py 一致，重装不丢）──
def _get_data_root() -> Path:
    """获取 AstrBot 全局持久化目录"""
    try:
        from astrbot.api.star import StarTools
        return Path(StarTools.get_data_dir("astrbot_plugin_zvv"))
    except ImportError:
        # 回退：非 AstrBot 环境中手动运行时用插件目录
        return PLUGIN_DIR / "data"

# ── 下载源（改成你自己的仓库地址）──
MODEL_URL = "https://github.com/bomomoQWQ/astrbot_plugin_zvv_peijian/releases/download/v1.0/model_zvv.zip"
IMAGES_URL = "https://github.com/bomomoQWQ/astrbot_plugin_zvv_peijian/releases/download/v1.0/images.zip"


def download(url: str, dest: Path, desc: str) -> bool:
    """下载文件，带进度条。自动使用 HTTPS_PROXY 环境变量。"""
    print(f"[init] 下载 {desc}...")
    print(f"        {url}")
    try:
        # 支持代理
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            from urllib.request import ProxyHandler, build_opener, install_opener
            install_opener(build_opener(ProxyHandler({"https": proxy, "http": proxy})))
            print(f"        使用代理: {proxy}")
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(f"\r        进度: {pct}%", end="", flush=True)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp, reporthook)
        print()  # newline
        shutil.move(str(tmp), str(dest))
        return True
    except Exception as e:
        print(f"\n[init] 下载失败: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def extract(zip_path: Path, target_dir: Path, desc: str) -> bool:
    """解压 zip 到目标目录。"""
    print(f"[init] 解压 {desc} → {target_dir}")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink()  # 删掉 zip
        return True
    except Exception as e:
        print(f"[init] 解压失败: {e}")
        return False


def check_and_download() -> bool:
    """检查模型和图片，缺失则下载。"""
    data_root = _get_data_root()
    model_dir = data_root / "model_zvv"
    images_dir = data_root / "images"
    need_model = not (model_dir / "model_q8.onnx").exists()
    need_images = not any(images_dir.iterdir()) if images_dir.exists() else True

    if not need_model and not need_images:
        print("[init] 模型和图片已就绪，无需下载。")
        return True

    print("[init] ========================================")
    print("[init]  首次运行，下载所需文件...")
    print("[init] ========================================")

    # ── 下载模型 ──
    if need_model:
        zip_path = data_root / "data" / "model_zvv.zip"
        if not download(MODEL_URL, zip_path, "ONNX 模型 (~74MB)"):
            print("[init] 模型下载失败，请检查网络或手动放置文件。")
            return False
        if not extract(zip_path, model_dir, "ONNX 模型"):
            return False
        print("[init] [OK] 模型就绪")

    # ── 下载图片 ──
    if need_images:
        zip_path = data_root / "data" / "images.zip"
        if not download(IMAGES_URL, zip_path, "表情包图片 (~31MB)"):
            print("[init] 图片下载失败。可手动放入 images/ 目录。")
            return False
        if not extract(zip_path, images_dir, "表情包图片"):
            return False
        print(f"[init] [OK] 图片就绪 ({len(list(images_dir.iterdir()))} 张)")

    print("[init] ========================================")
    print("[init]  初始化完成！重启 AstrBot 即可使用。")
    print("[init] ========================================")
    return True


if __name__ == "__main__":
    ok = check_and_download()
    sys.exit(0 if ok else 1)
