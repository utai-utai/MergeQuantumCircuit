"""
Metrics shared across the hardware experiments.

  * Distribution distances  — Hellinger fidelity, total-variation distance (Exp III).
  * Parity observable        — Z^{otimes a} SparsePauliOp (Exp IV).
  * Fisher information geometry — classical Fisher of an output distribution via
    the parameter-shift rule, its effective rank, and the normalised effective
    dimension of Abbas et al. (2021) (Exp V).
"""
import numpy as np


# --------------------------------------------------------------------------
# Distribution distances (Experiment III)
# --------------------------------------------------------------------------
def hellinger_fidelity(p, q):
    return float(np.sum(np.sqrt(p * q)) ** 2)


def tvd(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


# --------------------------------------------------------------------------
# Parity observable (Experiment IV)
# --------------------------------------------------------------------------
def parity_op(a):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list([("Z" * a, 1.0)])


# --------------------------------------------------------------------------
# Fisher information geometry (Experiment V)
# --------------------------------------------------------------------------
def fisher_information(prob_fn, theta, n_param):
    """Classical Fisher of p(y|theta) via the parameter-shift rule.

    `prob_fn(theta)` must return the output probability vector for a parameter
    vector.  Averaging over inputs is the caller's responsibility (sum the
    returned matrices).  Returns the (n_param, n_param) Fisher matrix.
    """
    p = prob_fn(theta)
    dp = np.zeros((n_param, len(p)))
    for j in range(n_param):
        tp = theta.copy(); tp[j] += np.pi / 2
        tm = theta.copy(); tm[j] -= np.pi / 2
        dp[j] = 0.5 * (prob_fn(tp) - prob_fn(tm))
    mask = p > 1e-12
    dlog = dp[:, mask] / p[mask]                 # d log p
    return (dlog * p[mask]) @ dlog.T             # sum_y p (dlog)(dlog)^T


def effective_rank(F):
    """Participation ratio of the Fisher spectrum: (sum l)^2 / sum l^2."""
    l = np.clip(np.linalg.eigvalsh(0.5 * (F + F.T)), 0, None)
    s = l.sum()
    return float(s * s / np.sum(l * l)) if s > 0 else 0.0


def effective_dimension(Fs, n_data=1e5, gamma=1.0):
    """Normalised effective dimension (Abbas et al. 2021) from a list of Fisher
    matrices sampled over the parameter manifold."""
    from scipy.special import logsumexp
    d = Fs[0].shape[0]
    tr_mean = np.mean([np.trace(F) for F in Fs])
    Fhat = [d * F / tr_mean for F in Fs]                 # normalise: mean trace = d
    kappa = gamma * n_data / (2 * np.pi * np.log(n_data))
    half_logdets = []
    for F in Fhat:
        _, ld = np.linalg.slogdet(np.eye(d) + kappa * 0.5 * (F + F.T))
        half_logdets.append(0.5 * ld)
    log_mean = logsumexp(half_logdets) - np.log(len(half_logdets))
    return float(2.0 * log_mean / np.log(kappa))
