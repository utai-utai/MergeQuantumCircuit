"""
IBM Quantum access and the shared NISQ plumbing used by Experiments III, IV, V.

One place for: reading credentials from .env, resolving the backend, building a
transpiler pass manager against the real target (free, no device time), running
circuits through SamplerV2, and turning measurement counts into probability
vectors / parity expectations.

Credentials live in ROOT_DIR/.env as IBM_API_KEY + IBM_CRN_STRING (channel
ibm_cloud).  The token/CRN are never printed.
"""
import os
import platform
from datetime import datetime, timezone

import numpy as np

from src import ROOT_DIR, BACKEND_NAME, SHOTS, TRANSPILE_SEED


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
def load_env(path=None):
    path = path or os.path.join(ROOT_DIR, ".env")
    creds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip('"').strip("'")
    return creds


def get_service():
    """QiskitRuntimeService authenticated from the project-local .env."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    creds = load_env()
    token, crn = creds["IBM_API_KEY"], creds["IBM_CRN_STRING"]
    return QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )


def get_backend(name=BACKEND_NAME):
    return get_service().backend(name)


# --------------------------------------------------------------------------
# Transpilation
# --------------------------------------------------------------------------
def make_passmanager(prefer_real=True, name=BACKEND_NAME, optimization_level=3,
                     seed_transpiler=TRANSPILE_SEED):
    """Pass manager + backend tag.  Transpiling against the real target is free
    (no device time) and gives accurate heavy-hex two-qubit-gate counts; falls
    back to AerSimulator when the backend cannot be reached."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    if prefer_real:
        try:
            backend = get_backend(name)
            return generate_preset_pass_manager(optimization_level=optimization_level,
                                                backend=backend,
                                                seed_transpiler=seed_transpiler), backend.name
        except Exception as e:                              # noqa
            print(f"WARNING: real backend unavailable ({e}); using AerSimulator "
                  f"(2q-gate counts will be optimistic)")
    from qiskit_aer import AerSimulator
    return generate_preset_pass_manager(optimization_level=optimization_level,
                                        backend=AerSimulator(),
                                        seed_transpiler=seed_transpiler), "aer"


def backend_metadata(backend):
    """Small JSON-safe provenance record for a hardware submission."""
    import qiskit
    import qiskit_ibm_runtime

    status = backend.status()
    props = None
    try:
        props = backend.properties()
    except Exception:                                      # pragma: no cover - backend dependent
        pass
    last_update = getattr(props, "last_update_date", None)
    return {
        "backend": backend.name,
        "num_qubits": backend.num_qubits,
        "operational": bool(status.operational),
        "status_msg": status.status_msg,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_last_update": (
            last_update.isoformat() if hasattr(last_update, "isoformat") else None
        ),
        "qiskit_version": qiskit.__version__,
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "shots": SHOTS,
        "transpile_seed": TRANSPILE_SEED,
    }


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def run_sampler(isa_circuits, backend, shots=SHOTS):
    """Submit transpiled circuits through SamplerV2 and block for the result."""
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(isa_circuits, shots=shots)
    print(f"Job ID: {job.job_id()} (waiting for result ...)")
    return job.result()


def submit_sampler(isa_circuits, backend, shots=SHOTS, max_per_job=None,
                   mitigation=False, batch_max_time="24h"):
    """Submit transpiled circuits through a Runtime Batch WITHOUT blocking, chunking to
    respect the backend's per-job circuit limit.  Real-device queues run far
    longer than an interactive call can wait, so submission and retrieval are
    split: this returns (job_ids, chunk_sizes); fetch the results later with
    `fetch_results`.  chunk_sizes lets the caller map a flat circuit index back
    to (job, local index).

    mitigation=True turns on device-level error suppression that needs no extra
    circuits: dynamical decoupling (XY4) against idle decoherence plus gate and
    measurement Pauli twirling to convert coherent/readout bias into stochastic
    noise -- the levers that de-bias the deep-circuit Fisher geometry."""
    from qiskit_ibm_runtime import Batch, SamplerV2
    if max_per_job is None:
        max_per_job = getattr(backend, "max_circuits", None) or 300
    job_ids, chunks = [], []
    with Batch(backend=backend, max_time=batch_max_time) as batch:
        batch_id = batch.session_id
        sampler = SamplerV2(mode=batch)
        if mitigation:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            print("  [mitigation] DD=XY4, gate+measurement twirling enabled")
        print(f"  Runtime Batch: {batch_id}")
        for start in range(0, len(isa_circuits), max_per_job):
            chunk = isa_circuits[start:start + max_per_job]
            job = sampler.run(chunk, shots=shots)
            job_ids.append(job.job_id()); chunks.append(len(chunk))
            print(f"  chunk [{start}:{start + len(chunk)}] -> job {job.job_id()}")
    return batch_id, job_ids, chunks


def submit_estimator_zne(pubs, backend, shots=SHOTS, max_per_job=None,
                         noise_factors=(1, 3, 5), extrapolator=("exponential", "linear")):
    """Submit EstimatorV2 PUBs with zero-noise extrapolation, WITHOUT blocking.
    ZNE amplifies noise by digital gate folding at the ISA level (so the folds
    survive — a logical-level fold would be optimised straight back to identity)
    and extrapolates each observable to the zero-noise limit.  Also enables
    readout mitigation + dynamical decoupling.  Returns (job_ids, chunk_sizes)."""
    from qiskit_ibm_runtime import EstimatorV2
    if max_per_job is None:
        max_per_job = getattr(backend, "max_circuits", None) or 300
    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    est.options.resilience.zne_mitigation = True
    est.options.resilience.zne.noise_factors = list(noise_factors)
    est.options.resilience.zne.extrapolator = list(extrapolator)
    est.options.resilience.measure_mitigation = True
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XY4"
    print(f"  [ZNE] noise_factors={list(noise_factors)} extrapolator={list(extrapolator)}, "
          f"readout-mitigation + DD on")
    job_ids, chunks = [], []
    for start in range(0, len(pubs), max_per_job):
        chunk = pubs[start:start + max_per_job]
        job = est.run(chunk)
        job_ids.append(job.job_id()); chunks.append(len(chunk))
        print(f"  chunk [{start}:{start + len(chunk)}] -> job {job.job_id()}")
    return job_ids, chunks


def jobs_status(service, job_ids):
    """{job_id: status-string} for polling a submitted batch."""
    out = {}
    for jid in job_ids:
        st = service.job(jid).status()
        out[jid] = st if isinstance(st, str) else getattr(st, "name", str(st))
    return out


def fetch_results(service, job_ids):
    """Retrieve finished SamplerV2 results by job id, in order (blocks per job
    until each is done)."""
    return [service.job(jid).result() for jid in job_ids]


def get_counts(result, i, creg="meas"):
    """Counts dict for circuit i from a SamplerV2 result (default classical
    register name is 'meas' from QuantumCircuit.measure_all)."""
    return getattr(result[i].data, creg).get_counts()


# --------------------------------------------------------------------------
# Counts post-processing
# --------------------------------------------------------------------------
def counts_to_probs(counts, dim, shots=None):
    p = np.zeros(dim)
    for bitstring, c in counts.items():
        p[int(bitstring.replace(" ", ""), 2)] = c
    shots = shots or sum(counts.values())
    return p / max(shots, 1)


def counts_to_parity(counts, shots=None):
    """<Z^{otimes a}> from counts: +1 for even Hamming weight, -1 for odd."""
    val = 0.0
    for bitstring, c in counts.items():
        val += ((-1) ** bitstring.replace(" ", "").count("1")) * c
    shots = shots or sum(counts.values())
    return val / max(shots, 1)
