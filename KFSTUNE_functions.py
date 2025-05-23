"""
KFSTUNE_functions.py  ––  core utilities for KFS-TUNE / ConvFS
-------------------------------------------------------------

• generate_kernels(...)             → random 1-D convolutional kernel bank
• apply_kernels(X, kernels)         → Numba-JIT feature-map extractor
• transform_and_select_features(...)→ kernel transform + scaling + K-best FS

The module is intentionally light-weight: import it in your training /
evaluation scripts and decide there which scorer (chi², MI, ANOVA …) you
want to use.
"""

import numpy as np
from numba import njit, prange
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
# scorer shortcut dictionary – pick the one you want in your pipeline
# ----------------------------------------------------------------------
scorers = {
    "chi2":   chi2,
    "mi":     mutual_info_classif,   # mutual information
    "anova":  f_classif              # 1-way ANOVA F-test
}

# ----------------------------------------------------------------------
# 1. SAFE kernel generator
# ----------------------------------------------------------------------
@njit
def generate_kernels(input_length: int,
                     num_kernels: int,
                     avg_series_length: int):
    """
    Return a kernel bank where every kernel satisfies
        (length-1)*dilation <= input_length + 2*padding
    so that out_len is always ≥ 1.
    """
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)

    lengths   = np.empty(num_kernels,        dtype=np.int32)
    biases    = np.empty(num_kernels,        dtype=np.float64)
    dilations = np.empty(num_kernels,        dtype=np.int32)
    paddings  = np.empty(num_kernels,        dtype=np.int32)

    # first choose all lengths so we know the total weight buffer size
    for i in range(num_kernels):
        lengths[i] = np.random.choice(candidate_lengths)

    weights = np.empty(lengths.sum(), dtype=np.float64)

    w_ptr = 0
    for i in range(num_kernels):
        k_len = lengths[i]

        # -------- dilation & padding selected until the constraint is met
        while True:
            dil  = 2 ** np.random.uniform(
                       0, np.log2((input_length-1)/(k_len-1)))
            dil  = np.int32(dil)
            pad  = ((k_len-1) * dil) // 2 if np.random.randint(2) else 0
            if (k_len-1)*dil <= input_length + 2*pad:
                break                     # ✅ valid
        dilations[i] = dil
        paddings[i]  = pad

        # -------- weights / bias
        w = np.random.normal(0., 1., k_len)
        w -= w.mean()
        weights[w_ptr:w_ptr+k_len] = w
        w_ptr += k_len

        biases[i] = np.random.uniform(-1., 1.)

    return weights, lengths, biases, dilations, paddings

# ----------------------------------------------------------------------
# 2. _apply_kernel_1d — safeguard against out_len <= 0 just in case
# ----------------------------------------------------------------------
@njit(fastmath=True)
def _apply_kernel_1d(x, weights, length, bias, dilation, padding):
    n = len(x)
    out_len = (n + 2*padding) - (length-1)*dilation
    if out_len <= 0:                       # <- shouldn’t happen any more,
        return 0., -np.inf, 0.            #    but stay defensive

    max_val   = -np.inf
    ppv_count = 0
    activ     = np.empty(out_len, dtype=np.float64)

    idx_out = 0
    last_i  = (n + padding) - (length-1)*dilation
    for i in range(-padding, last_i):
        s = bias
        p = i
        for j in range(length):
            if 0 <= p < n:
                s += weights[j] * x[p]
            p += dilation
        max_val   = s if s > max_val else max_val
        ppv_count += 1 if s > 0. else 0
        activ[idx_out] = s
        idx_out += 1

    return ppv_count / out_len, max_val, np.std(activ)


# ----------------------------------------------------------------------
# 3. apply_kernels  (compile *without* parallel first)
# ----------------------------------------------------------------------
@njit(
    "float64[:,:](float64[:,:], "
    "Tuple((float64[::1], int32[:], float64[:], int32[:], int32[:])))",
    parallel=False,  # switch back to True later if you want
    fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    n_samples, _      = X.shape
    n_kernels         = len(lengths)
    feats_per_kernel  = 3

    Z = np.empty((n_samples, n_kernels*feats_per_kernel), dtype=np.float64)

    for i in range(n_samples):
        w_ptr = 0
        f_ptr = 0
        for k in range(n_kernels):
            k_len = lengths[k]
            ppv, mx, sd = _apply_kernel_1d(
                X[i],
                weights[w_ptr:w_ptr+k_len],
                k_len,
                biases[k],
                dilations[k],
                paddings[k]
            )
            Z[i, f_ptr:f_ptr+3] = (ppv, mx, sd)
            w_ptr += k_len
            f_ptr += feats_per_kernel
    return Z
# ----------------------------------------------------------------------
# 4. transform + (optional) feature selection wrapper
# ----------------------------------------------------------------------
def transform_and_select_features(
    X, kernels, *,
    y=None,
    num_features=500,
    score_func=chi2,
    selector=None,
    scaler=None,
    is_train=True
):
    """
    Kernel transform  ➔  scaling  ➔  K-best selection (train / test mode).

    Parameters
    ----------
    X            : ndarray, shape (n_samples, n_timesteps)
    kernels      : tuple from `generate_kernels`
    y            : labels (needed in training mode)
    num_features : keep K best (default 500)
    score_func   : e.g. chi2, mutual_info_classif, f_classif, ...
    selector     : fitted SelectKBest (pass when is_train=False)
    scaler       : fitted MinMaxScaler (pass when is_train=False)
    is_train     : True = fit  |  False = transform only

    Returns
    -------
    X_sel                      (always)
    selector, k, scaler        (additionally when is_train=True)
    """
    # 1) kernel transform
    Z = apply_kernels(X, kernels)

    # 2) scale 0-1
    if scaler is None:
        scaler = MinMaxScaler()
        Z = scaler.fit_transform(Z)
    else:
        Z = scaler.transform(Z)

    # 3) feature selection
    if is_train:
        selector = SelectKBest(score_func, k=num_features)
        Z_sel    = selector.fit_transform(Z, y)
        return Z_sel, selector, num_features, scaler
    else:
        if selector is None:
            raise ValueError("selector must be provided when is_train=False")
        Z_sel = selector.transform(Z)
        return Z_sel
