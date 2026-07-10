from .aae import AAEAD
from .autoencoder import AutoencoderAD
from .vae import VAEAD

REGISTRY = {
    "autoencoder": AutoencoderAD,
    "vae": VAEAD,
    "aae": AAEAD,
}
