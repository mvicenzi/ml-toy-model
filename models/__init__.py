# models/__init__.py

from .minkunet_base import MinkUNetBase
from .minkunet_attention import MinkUNetDenseAttention, MinkUNetSparseAttention
from .uresnet_sparse_pool import UResNetSparsePool, UResNetSparsePoolAttention

MODEL_REGISTRY = {
    "base": MinkUNetBase,
    "attn_dense": MinkUNetDenseAttention,
    "attn_sparse": MinkUNetSparseAttention,
    "base_pool": UResNetSparsePool,
    "attn_pool": UResNetSparsePoolAttention,
    # add more here
}
