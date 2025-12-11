## Toy Model with WarpConvNet

### Installation
WarpConvNet requires Python >= 3.9 and a working CUDA toolchain.
See: [WarpConvNet/installation](https://nvlabs.github.io/WarpConvNet/getting_started/installation/).

1. Install CUDA Toolkit. `wcwc` offers `cuda@12.5.1`, but I installed 12.6 manually.  
```
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sh cuda_12.6.0_560.28.03_linux.run --tmpdir=/some/tmp/dir --toolkitpath=/path/to/cuda-12.6
```

2. Create a Python/`uv` environment to install packages:
```
uv venv ml-venv
```

3. Create a script to quickly setup your envinroment each time.
```
## local cache (optional)
export UV_CACHE_DIR=/path/to/uv_cache
export XDG_CACHE_HOME=/path/to/cache

## activate python environment
source ml-venv/bin/activate

## export local cuda 12.6 installation
export CUDA_HOME=/path/to/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

4. Install dependencies for WarpConvNet. Note: other packages might need to be installed to resolve build errors. 
```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install build ninja
uv pip install cupy-cuda12x
uv pip install git+https://github.com/rusty1s/pytorch_scatter.git
MAX_JOBS=2 uv pip install flash-attn --no-build-isolation     # reduce parallelism to help building
```

5. Clone and install WarpConvNet.
```
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
uv pip install .
```

6. Install additional packages as required to run the examples.

### Model architectures
Model architectures are available in `./models` and must be declared in the `MODEL_REGISTRY` (defined in `./models/__init__.py`) to be picked up for training.
For example:
```
from .minkunet_base import MinkUNetBase

MODEL_REGISTRY = {
    "base": MinkUNetBase,
    # add more here
    ...
}
```

### Training a model
Training is perfomed by executing `training.py`. Training parameters, such as the model architecture or the number of epochs, can be set interactively by ovveride the default parameters.
For example:
```
python training.py --model_name=base --epochs=10
python training.py --model_name=attn_dense --epochs=10
```
