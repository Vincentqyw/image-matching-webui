from dkm import DKMv2
from dkm.benchmarks import ScanNetBenchmark
import json
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_datasets",
        type=str,
        choices=["indoor", "outdoor"],
        default="indoor",
    )
    parser.add_argument("--r", type=float, default=2)
    parser.add_argument("--thr", type=float, default=0.5)
    args, _ = parser.parse_known_args()
    train_datasets = args.train_datasets
    model = DKMv2(pretrained=True, version=train_datasets)
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = []
    r = args.r
    thr = args.thr
    for s in range(5):
        scannet_results.append(scannet_benchmark.benchmark(model, r=r, thr=thr))
        json.dump(
            scannet_results,
            open(f"results/scannet_r{r}_{train_datasets}_{thr=}.json", "w"),
        )
