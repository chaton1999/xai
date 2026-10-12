#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np

# ---------------- CONFIG ----------------
TAG_A = "ckpt_1"
TAG_B = "ckpt_1000000"

# These are written by normal_modes_one.py inside nm_<tag>/
A_NPZ = os.path.join("nm_{0}".format(TAG_A), "{0}_vib_filtered.npz".format(TAG_A))
B_NPZ = os.path.join("nm_{0}".format(TAG_B), "{0}_vib_filtered.npz".format(TAG_B))

# Pick which mode you mean:
#   "last"  -> last mode in filtered list
#   or an integer index into the filtered list (0-based), e.g. 0,1,2,...
MODE_PICK = "last"

# frequency match tolerance (cm^-1)
FREQ_TOL = 10.0
# ---------------------------------------


def l2norm(x):
    return np.sqrt(np.sum(x * x))

def cosine(a, b, eps=1e-20):
    na = l2norm(a)
    nb = l2norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def main():
    if not os.path.isfile(A_NPZ):
        raise IOError("Missing: {0}".format(A_NPZ))
    if not os.path.isfile(B_NPZ):
        raise IOError("Missing: {0}".format(B_NPZ))

    A = np.load(A_NPZ)
    B = np.load(B_NPZ)

    # In your normal_modes_one.py:
    #   np.savez(tag+"_vib_filtered.npz", idx=idx, freqs_cm=freqs_f, modes=modes_f)
    freqsA = A["freqs_cm"]
    modesA = A["modes"]   # shape (n_kept, 3N)
    idxA   = A["idx"]     # original mode indices (in 0..3N-1)

    freqsB = B["freqs_cm"]
    modesB = B["modes"]
    idxB   = B["idx"]

    nA = len(freqsA)
    nB = len(freqsB)
    print("A kept modes:", nA, "B kept modes:", nB)

    # Choose a mode in A (filtered index)
    if MODE_PICK == "last":
        kA = nA - 1
    else:
        kA = int(MODE_PICK)
        if kA < 0 or kA >= nA:
            raise ValueError("MODE_PICK out of range for A: {0}".format(kA))

    fA = freqsA[kA]
    qA = modesA[kA].astype(float)

    # Match mode in B by closest |freq|
    # (use abs in case any imaginary/negative conventions sneak in)
    target = float(np.abs(fA))
    diffs = np.abs(np.abs(freqsB) - target)
    kB = int(np.argmin(diffs))
    fB = freqsB[kB]
    qB = modesB[kB].astype(float)

    if float(diffs[kB]) > FREQ_TOL:
        print("[WARN] Closest freq in B differs by {:.2f} cm^-1 (tol {:.2f})".format(float(diffs[kB]), FREQ_TOL))

    # Raw dot and cosine overlap
    dot_raw = float(np.dot(qA, qB))
    cos_raw = cosine(qA, qB)
    cos_abs = abs(cos_raw)

    # Sign-aligned (flip B if needed so overlap is positive)
    if cos_raw < 0:
        qB_aligned = -qB
    else:
        qB_aligned = qB
    dot_aligned = float(np.dot(qA, qB_aligned))
    cos_aligned = cosine(qA, qB_aligned)

    print("A filtered index kA:", kA, "orig mode idx:", int(idxA[kA]), "freq:", fA)
    print("B matched index kB:", kB, "orig mode idx:", int(idxB[kB]), "freq:", fB)
    print("")
    print("dot_raw      =", dot_raw)
    print("cos_raw      =", cos_raw)
    print("|cos_raw|    =", cos_abs)
    print("dot_aligned  =", dot_aligned)
    print("cos_aligned  =", cos_aligned)

if __name__ == "__main__":
    main()

