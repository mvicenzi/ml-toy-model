# models/__init__.py

from .minkunet_base import MinkUNetBase
from .minkunet_attention import MinkUNetDenseAttention, MinkUNetSparseAttention

MODEL_REGISTRY = {
    "base": MinkUNetBase,
    "attn_dense": MinkUNetDenseAttention,
    "attn_sparse": MinkUNetSparseAttention,
    # add more here
}
