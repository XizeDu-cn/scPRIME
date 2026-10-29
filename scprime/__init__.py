"""scPRIME: chromatin embeddings and regulator discovery."""

from .pipeline import train, align_batches, infer_regulators, tf_activity

__all__ = ["train", "align_batches", "infer_regulators", "tf_activity"]
__version__ = "0.1.0"
