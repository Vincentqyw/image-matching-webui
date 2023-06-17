import json

def test_mega1500(model):
    from dkm.benchmarks import Megadepth1500Benchmark
    model.h_resized = 660
    model.w_resized = 880
    model.upsample_preds = True
    #model.upsample_res = (968, 1472)
    model.upsample_res = (1152, 1536)
    model.use_soft_mutual_nearest_neighbours = False
    megaloftr_benchmark = Megadepth1500Benchmark("data/megadepth")
    megaloftr_results = []
    megaloftr_results.append(megaloftr_benchmark.benchmark(model))
    json.dump(megaloftr_results, open(f"results/mega1500_{model.name}_1152_1536_upsample_8_4_2_1_again2.json", "w"))


def test_mega_8_scenes(model):
    from dkm.benchmarks import Megadepth1500Benchmark
    model.h_resized = 660
    model.w_resized = 880
    model.upsample_preds = True
    model.upsample_res = (1152, 1536)
    megaloftr_benchmark = Megadepth1500Benchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                            'mega_8_scenes_0025_0.1_0.3.npz',
                                                            'mega_8_scenes_0021_0.1_0.3.npz',
                                                            'mega_8_scenes_0008_0.1_0.3.npz',
                                                            'mega_8_scenes_0032_0.1_0.3.npz',
                                                            'mega_8_scenes_1589_0.1_0.3.npz',
                                                            'mega_8_scenes_0063_0.1_0.3.npz',
                                                            'mega_8_scenes_0024_0.1_0.3.npz',
                                                            'mega_8_scenes_0019_0.3_0.5.npz',
                                                            'mega_8_scenes_0025_0.3_0.5.npz',
                                                            'mega_8_scenes_0021_0.3_0.5.npz',
                                                            'mega_8_scenes_0008_0.3_0.5.npz',
                                                            'mega_8_scenes_0032_0.3_0.5.npz',
                                                            'mega_8_scenes_1589_0.3_0.5.npz',
                                                            'mega_8_scenes_0063_0.3_0.5.npz',
                                                            'mega_8_scenes_0024_0.3_0.5.npz']
                                                            )
    megaloftr_results = []
    megaloftr_results.append(megaloftr_benchmark.benchmark(model))
    json.dump(megaloftr_results, open(f"results/mega_8_scenes_{model.name}_1152_1536_upsample_8_4_2_1.json", "w"))


def test_hpatches(model):
    from dkm.benchmarks import (
    HpatchesHomogBenchmark,
    )
    model.h_resized = 540
    model.w_resized = 720
    model.upsample_preds = False
    homog_benchmark = HpatchesHomogBenchmark("data/hpatches")
    homog_results = []
    homog_results.append(homog_benchmark.benchmark(model))
    json.dump(
        homog_results, open(f"results/hpatches_homog_{model.name}.json", "w")
    )

def test_st_pauls(model):
    raise NotImplementedError("Not available yet")
    from dkm.benchmarks import (
    StPaulsCathedralBenchmark,
    )
    model.h_resized = 540
    model.w_resized = 720
    model.upsample_preds = True
    st_pauls_cathedral_benchmark = StPaulsCathedralBenchmark("data/st_pauls_cathedral")
    st_pauls_cathedral_results = []
    st_pauls_cathedral_results.append(st_pauls_cathedral_benchmark.benchmark(model))
    json.dump(
        st_pauls_cathedral_results, open(f"results/st_pauls_{model.name}.json", "w")
    )


if __name__ == "__main__":
    from dkm.models.model_zoo import DKMv3_outdoor
    model = DKMv3_outdoor()
    test_mega1500(model)
    #test_mega_8_scenes(model)
    # test_hpatches(model)
    # test_st_paults(model) # TODO: benchmark provided by ECO-TR authors, not sure about uploading.
    