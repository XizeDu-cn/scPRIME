"""scPRIME: chromatin embeddings and regulator discovery."""

from .pipeline import train, calibrate, align_batches, infer_regulators, tf_activity

__all__ = ["train", "calibrate", "align_batches", "infer_regulators", "tf_activity"]
__version__ = "0.1.0"
