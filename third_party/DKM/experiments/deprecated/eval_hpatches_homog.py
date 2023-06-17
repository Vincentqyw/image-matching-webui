from dkm import (
    DKM,
)
from dkm.benchmarks import (
    HpatchesHomogBenchmark,
)
import json


if __name__ == "__main__":
    version = "mega"
    model = DKM(pretrained=True, version=version)

    homog_benchmark = HpatchesHomogBenchmark("data/hpatches")

    homog_results = []
    r = 2
    for s in range(5):
        homog_results.append(homog_benchmark.benchmark(model, r=r))
        json.dump(
            homog_results, open(f"results/hpatches_homog_r{r}_{version}.json", "w")
        )
