"""scPRIME command-line interface."""

import argparse
from . import pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="scprime",
        description="Cell embeddings, regulator discovery and TF activity from scATAC-seq",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "calibrate", "align-batches"]:
        p = commands.add_parser(name)
        for key in ["input", "bundle", "output"]:
            p.add_argument("--" + key, required=True)
        if name != "train":
            p.add_argument("--parent", required=True, help="Output directory of the embedding step")
        if name == "align-batches":
            p.add_argument("--batch-key", required=True, help="obs column holding the batch label")
        p.add_argument(
            "--epochs",
            type=int,
            default={"train": 20, "calibrate": 8, "align-batches": 10}[name],
        )
        p.add_argument("--seed", type=int, default=15)
        p.add_argument("--device", choices=["cpu", "cuda"])
    p = commands.add_parser("infer-regulators")
    for key in ["input", "bundle", "model", "family-map", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--label-key", default="celltype", help="obs column holding the cell type")
    p.add_argument(
        "--exclude-labels",
        default="",
        help="Comma-separated cell-type labels that are not queried (for example UNK)",
    )
    p = commands.add_parser("tf-activity")
    for key in ["input", "bundle", "model", "output"]:
        p.add_argument("--" + key, required=True)
    p = commands.add_parser("prepare-motifs")
    for key in ["input", "scan", "output"]:
        p.add_argument("--" + key, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare-motifs":
        from .annotations import prepare_bundle

        prepare_bundle(**args)
        return
    if command == "infer-regulators":
        excluded = args.pop("exclude_labels")
        args["exclude_labels"] = tuple(x for x in excluded.split(",") if x)
    getattr(pipeline, command.replace("-", "_"))(**args)
