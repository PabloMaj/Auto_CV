import gc

from src.utils.logger import get_logger

logger = get_logger(__name__)


def cuda_cleanup(label: str = "") -> None:
    """Release GPU memory: collect Python garbage, empty CUDA cache, synchronize."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            tag = f" [{label}]" if label else ""
            logger.info(f"GPU memory released{tag}")
    except Exception as exc:
        logger.warning(f"CUDA cleanup failed: {exc}")
