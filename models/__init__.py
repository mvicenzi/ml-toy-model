# models/__init__.py

from .mnist_original import Net
from .minkunet_base import MinkUNetBase, MinkUNetBaseCompress
from .minkunet_attention import MinkUNetDenseAttention, MinkUNetSparseAttention
from .minkunet_attention import MinkUNetSparseAttentionNoEnc, MinkUNetSparseAttentionNoFlash, MinkUNetSparseAttentionNoFlashEnc
from .uresnet_sparse_pool import UResNetSparsePool, UResNetSparsePoolAttention

MODEL_REGISTRY = {

    ## this is base model --> mink_unet.py example
    ## converted to work on 2D MNIST iages
    "base": MinkUNetBase,

    ## base model + attention in the bottleneck
    ## attention via densifying + MHSA on dense 
    "attn_dense": MinkUNetDenseAttention,
    ## use attention on sparse (flash_attn + spatial_encoding)
    "attn_sparse": MinkUNetSparseAttention,

    ## varitions of the sparse attention module
    "attn_sparse_noenc" : MinkUNetSparseAttentionNoEnc,
    "attn_sparse_noflash":  MinkUNetSparseAttentionNoFlash,
    "attn_sparse_noflashenc": MinkUNetSparseAttentionNoFlashEnc,

    ## similar to base model
    ## but compress features on the way up
    "base_compress": MinkUNetBaseCompress,

    ## similar to base model 
    ## but using pooling for downsampling
    "base_pool": UResNetSparsePool,
    ## but using pooling for downsampling + attention in bottleneck
    "attn_pool": UResNetSparsePoolAttention,

    ### original mnist.py example in WarpConvNet
    "original": Net,
}
