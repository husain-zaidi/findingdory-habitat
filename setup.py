from setuptools import setup
setup(
    name="findingdory",
    packages=["findingdory"],
    version="0.1",
    install_requires=[
        "numpy>=2,<2.4",
        "pandas",
        "pillow==10.4.0",
        "rtree",
        "scipy>=1.13.0",
        "wandb",
        "ipython",
        "ipdb",
        "json5",
    ],
    extras_require={
        "vlm_baseline": [
            "accelerate",
            "av",
            "bitsandbytes",
            "einops",
            "protobuf==3.20.1",
            "qwen-vl-utils[decord]",
            "openai",
            "transformers @ git+https://github.com/huggingface/transformers.git@main",
        ],
        "mapping_baseline": [
            "torch_geometric",
            "open3d",
            "scikit-image",
            "sophuspy",
            "scikit-fmm",
        ],
    },
)
