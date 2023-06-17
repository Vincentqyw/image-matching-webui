from argparse import ArgumentParser
from dkm import (
    DKMv2,
)
from dkm.benchmarks import (
    Megadepth1500Benchmark,
)
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--r", type=float, default=3)
    args, _ = parser.parse_known_args()
    model = DKMv2(pretrained=True, version="outdoor", resolution="high")
    megaloftr_benchmark = Megadepth1500Benchmark("data/megadepth")
    megaloftr_results = []
    r = args.r
    for s in range(5):
        megaloftr_results.append(megaloftr_benchmark.benchmark(model, r=r))
        json.dump(megaloftr_results, open(f"results/mega1500_r{r}.json", "w"))
