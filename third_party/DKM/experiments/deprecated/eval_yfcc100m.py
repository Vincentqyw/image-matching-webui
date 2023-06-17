from argparse import ArgumentParser
from dkm import (
    DKM,
)
from dkm.benchmarks import (
    Yfcc100mBenchmark,
)
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--r", type=float, default=2)
    args, _ = parser.parse_known_args()
    train_datasets = "mega_synthetic"
    model = DKM(pretrained=True, version=train_datasets)
    yfcc_benchmark = Yfcc100mBenchmark("data/yfcc100m_test")
    yfcc_results = []
    r = args.r
    for s in range(5):
        yfcc_results.append(yfcc_benchmark.benchmark(model, r=r))
        json.dump(
            yfcc_results, open(f"results/yfcc100m_r{r}_{train_datasets}.json", "w")
        )
