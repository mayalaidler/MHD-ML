"""Detection of the chemistry solver's numerical floor in a log target.

Trace species in cells where they physically don't exist are stored at a
tiny solver floor (often 1e-20..1e-33).  Those digits are numerics, not
physics, and pollute both training and evaluation.  The floor shows up as
a separate low cluster in the histogram of log10(abundance); we find the
sparsest bin between the floor cluster's peak and the physical branch's
peak and use it as the threshold.
"""

from __future__ import annotations

import numpy as np


def detect_floor(
    y_log: np.ndarray,
    split_guess: float = -14.0,
    lo: float = -35.0,
    hi: float = -2.0,
    nbins: int = 130,
) -> float | None:
    """Return the log10 threshold separating floor from physical values,
    or None when the target has no floor branch.

    Parameters
    ----------
    y_log : log10 of the abundance values.
    split_guess : rough boundary below which values are assumed to be
        candidate floor values.  The default suits FLASH mass fractions,
        where physical values rarely drop below ~1e-12.
    """
    y_log = np.asarray(y_log)
    if y_log.min() > split_guess:
        return None

    counts, edges = np.histogram(y_log, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    below = centers < split_guess
    if counts[below].sum() == 0 or counts[~below].sum() == 0:
        return None

    p_floor = centers[below][np.argmax(counts[below])]
    p_phys = centers[~below][np.argmax(counts[~below])]
    between = (centers > p_floor) & (centers < p_phys)
    if not between.any():
        return None
    return float(centers[between][np.argmin(counts[between])])
