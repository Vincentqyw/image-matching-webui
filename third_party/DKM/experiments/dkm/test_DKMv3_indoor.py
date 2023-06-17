import json


def test_scannet(model):
    from dkm.benchmarks import ScanNetBenchmark
    model.h_resized = 480
    model.w_resized = 640
    model.upsample_preds = False
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = []
    scannet_results.append(scannet_benchmark.benchmark(model))
    json.dump(scannet_results, open(f"results/scannet_{model.name}.json", "w"))

if __name__ == "__main__":
    from dkm.models.model_zoo import DKMv3_indoor
    model = DKMv3_indoor()
    test_scannet(model)