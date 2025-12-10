# models/__init__.py

from .uresnet_sparse import MinkNet
from .uresnet_sparse import Net

MODEL_REGISTRY = {
    "net": Net,
    "mink_unet": MinkNet,
    # add more here
}
