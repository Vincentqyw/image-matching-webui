from dkm import DKMv2
from dkm.benchmarks import ScanNetBenchmark
import json
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--r", type=float, default=2)
    args, _ = parser.parse_known_args()
    model = DKMv2(pretrained=True, version="indoor", resolution = "low")
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = []
    r = args.r
    for s in range(5):
        scannet_results.append(scannet_benchmark.benchmark(model, r=r))
        json.dump(
            scannet_results,
            open(f"results/scannet_r{r}.json", "w"),
        )
