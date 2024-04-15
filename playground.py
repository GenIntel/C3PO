from od3d.benchmark.benchmark import OD3D_Benchmark
from omegaconf import DictConfig
config = DictConfig(
benchmark = OD3D_Benchmark(config=config)
OD3D_Benchmark.run()