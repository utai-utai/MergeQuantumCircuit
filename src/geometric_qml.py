"""
Geometric QML engine — corrected implementation of the transfer map and
manifold merging described in docs/THEORY.md.

All of the experiments (simulation / ViT / DiT) import from here so that the
Method (simulation) and the Experiments (ViT, DiT) use *exactly* the same,
theoretically-correct primitives.

Key corrections relative to the original code (see docs/THEORY.md §3, §5):

  * Generator extraction is the polar / nearest-unitary route:
        A = Q^H W Q  ->  U_A = polar(A)  ->  H = i log(U_A)   (Hermitian)
    NOT the Hermitian part (A + A^H)/2, which collapses to a global phase
    e^{-i} I in the near-identity regime.

  * The mixing operator O(Q,H) = Q e^{-iH} Q^H reproduces, by construction,
    the *nearest unitary* of the retained subspace block.  The quantization
    error is then exactly the singular-value deviation of A from 1
    (Lemma 4.1 / Theorem 4.2):
        ||W - O||_F  <=  ||W - Pi_Q(W)||_F  +  ||P_A - I||_F .

  * Merging uses the covering frame  Q_C = orth([Q_A, Q_B])  (dim <= 2k),
    not the top-k frame of the averaged weight, so the parallel transport
    Lift-Project-Restrict is (near-)lossless.
"""

import torch


# ----------------------------------------------------------------------------
# Core primitives
# ----------------------------------------------------------------------------
def subspace_frame(W, k):
    """Top-k left singular vectors of W as a complex Stiefel frame Q (Q^H Q = I_k).

    Args:
        W: (n, n) real or complex weight matrix.
        k: truncation rank.
    Returns:
        Q: (n, k) complex64 with orthonormal columns.
    """
    Wc = W.to(torch.complex64)
    U, _, _ = torch.linalg.svd(Wc, full_matrices=False)
    return U[:, :k].contiguous()


def nearest_unitary(A):
    """Polar unitary U_A = argmin_{U in U(k)} ||A - U||_F.

    For A = X S Y^H (SVD) the minimiser is U_A = X Y^H  (Fan-Hoffman).
    """
    X, _, Yh = torch.linalg.svd(A)
    return X @ Yh


def hermitian_generator(U):
    """Hermitian H with e^{-iH} = U for a unitary U, via the principal branch
    H = i log(U).

    Implemented through the eigendecomposition of the (normal) unitary matrix;
    the result is symmetrised to remove numerical anti-Hermitian residue.
    """
    evals, V = torch.linalg.eig(U)
    theta = torch.angle(evals)                      # in (-pi, pi]
    # e^{-iH} = U  =>  eigenvalues of H are -theta  (so e^{-i(-theta)} = e^{i theta})
    D = torch.diag((-theta).to(V.dtype))
    H = V @ D @ torch.linalg.inv(V)
    return 0.5 * (H + H.mH)                          # enforce Hermiticity


def first_order_generator(U):
    """Cheap near-identity surrogate  H ≈ (i/2)(U - U^H) = i·antiHerm(U).

    Accurate to O(||H||^3); used where a matrix log is overkill (e.g. as a
    trainable-parameter initialisation in the near-identity regime).
    """
    return 0.5j * (U - U.mH)


# ----------------------------------------------------------------------------
# Transfer map  T(W) = (Q, U_A, H)
# ----------------------------------------------------------------------------
def transfer_map(W, k, exact_log=True):
    """Analytic classical->quantum transfer map (docs/THEORY.md §3).

    Returns:
        Q   : (n, k) Stiefel frame.
        U_A : (k, k) nearest unitary of the subspace block A = Q^H W Q.
        H   : (k, k) Hermitian generator, e^{-iH} = U_A.
    """
    Q = subspace_frame(W, k)
    A = Q.mH @ W.to(torch.complex64) @ Q
    U_A = nearest_unitary(A)
    H = hermitian_generator(U_A) if exact_log else first_order_generator(U_A)
    return Q, U_A, H


def mixing_operator(Q, U):
    """O(Q,H) = Q U Q^H  with U = e^{-iH}  (pass U_A for the zero-shot case)."""
    return Q @ U @ Q.mH


# ----------------------------------------------------------------------------
# Error decomposition (Theorem 4.2) — used by the Method figures
# ----------------------------------------------------------------------------
def error_decomposition(W, k):
    """Return (total, truncation, nonunitary) Frobenius errors for rank k.

        total       = ||W - O(Q,H)||_F
        truncation  = ||W - Pi_Q(W)||_F           (subspace truncation, term i)
        nonunitary  = ||P_A - I||_F = sqrt(sum (sigma_j(A) - 1)^2)   (term ii)
    """
    Wc = W.to(torch.complex64)
    Q = subspace_frame(W, k)
    A = Q.mH @ Wc @ Q
    U_A = nearest_unitary(A)
    O = Q @ U_A @ Q.mH

    total = torch.linalg.norm(Wc - O, ord="fro").item()
    Pi = Q @ A @ Q.mH
    truncation = torch.linalg.norm(Wc - Pi, ord="fro").item()
    sigma_A = torch.linalg.svdvals(A)
    nonunitary = torch.sqrt(torch.sum((sigma_A - 1.0) ** 2)).item()
    return total, truncation, nonunitary


# ----------------------------------------------------------------------------
# Manifold merging (docs/THEORY.md §5)
# ----------------------------------------------------------------------------
def covering_frame(Q_A, Q_B):
    """Covering Stiefel frame Q_C = orth([Q_A, Q_B]),  dim(Q_C) <= 2k.

    Spans span(Q_A) u span(Q_B) so the transport in `transport_generator`
    is (near-)lossless (Lemma 5.3).
    """
    stacked = torch.cat([Q_A, Q_B], dim=1)            # (n, 2k)
    Q_C, _ = torch.linalg.qr(stacked)                 # orthonormal basis of the column span
    # Drop numerically-null directions (rank may be < 2k if the spans overlap).
    rank = torch.linalg.matrix_rank(stacked)
    return Q_C[:, :rank].contiguous()


def transport_generator(Q_src, H_src, Q_C):
    """Lift-Project-Restrict transport of a generator to the covering frame:
        H' = Q_C^H (Q_src H_src Q_src^H) Q_C            (Hermitian-preserving).
    """
    H_global = Q_src @ H_src @ Q_src.mH
    H_prime = Q_C.mH @ H_global @ Q_C
    return 0.5 * (H_prime + H_prime.mH)


def merge_generators(W_A, W_B, k, alpha=0.5, beta=0.5):
    """Generator-space (Lie-algebra) merge of two near-unitary cores.

    Returns:
        Q_C   : covering frame.
        H_C   : alpha H_A' + beta H_B'  (transported, averaged generator).
        U_C   : e^{-i H_C}  (ready for the mixing operator).
    """
    Q_A, _, H_A = transfer_map(W_A, k)
    Q_B, _, H_B = transfer_map(W_B, k)
    Q_C = covering_frame(Q_A, Q_B)
    H_Ap = transport_generator(Q_A, H_A, Q_C)
    H_Bp = transport_generator(Q_B, H_B, Q_C)
    H_C = alpha * H_Ap + beta * H_Bp
    U_C = torch.linalg.matrix_exp(-1j * H_C)
    return Q_C, H_C, U_C
