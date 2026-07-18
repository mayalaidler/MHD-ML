"""ionpred — predict ion abundances in FLASH MHD simulations with ML.

Core ideas
----------
1. Any tracked species field in a FLASH checkpoint can be a prediction
   target (``ionpred species <checkpoint>`` lists them).
2. Trace species pile up at the chemistry solver's numerical floor, which
   is noise, not physics.  ionpred auto-detects that floor and evaluates
   (or models, via the hurdle model) "is the ion present?" separately
   from "how much?".
3. R² alone misleads when targets have very different variances, so every
   evaluation also reports RMSE (dex) and the fraction of cells predicted
   within 0.5 and 1 dex.
"""

__version__ = "0.1.0"

from .features import build_features, valid_mask
from .floors import detect_floor
from .io import discover_species, read_fields, species_label
from .metrics import evaluate
from .models import HurdleModel, make_model
from .splits import loso_folds, spatial_split, temporal_split

__all__ = [
    "build_features",
    "valid_mask",
    "detect_floor",
    "discover_species",
    "read_fields",
    "species_label",
    "evaluate",
    "HurdleModel",
    "make_model",
    "loso_folds",
    "spatial_split",
    "temporal_split",
    "__version__",
]
