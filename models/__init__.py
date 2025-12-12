# models/__init__.py

from .mnist_original import Net
from .minkunet_base import MinkUNetBase, MinkUNetBaseCompress
from .minkunet_attention import MinkUNetDenseAttention, MinkUNetSparseAttention
from .uresnet_sparse_pool import UResNetSparsePool, UResNetSparsePoolAttention

MODEL_REGISTRY = {
    "base": MinkUNetBase,
    "base_compress": MinkUNetBaseCompress,
    "attn_dense": MinkUNetDenseAttention,
    "attn_sparse": MinkUNetSparseAttention,
    "base_pool": UResNetSparsePool,
    "attn_pool": UResNetSparsePoolAttention,
    "original": Net,
    # add more here
}
