#!/bin/bash
#SBATCH --job-name=mcdc-vv-electron
#SBATCH --partition=pbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# LLNL Dane note:
# Submit with your bank/account, for example:
# sbatch -A myBank slurm/submit_mcdc_vv_electron.sh

resolve_paths() {
    local candidate=""
    local normalized=""
    local script_dir=""
    local script_parent=""
    local -a candidates=()

    if [[ -n "${ROOT_DIR:-}" ]]; then
        candidates+=("${ROOT_DIR}")
    fi

    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        candidates+=("${SLURM_SUBMIT_DIR}")
        candidates+=("${SLURM_SUBMIT_DIR}/..")
    fi

    candidates+=("$(pwd)")
    candidates+=("$(pwd)/..")

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    if [[ -n "${script_dir}" ]]; then
        script_parent="$(cd "${script_dir}/.." 2>/dev/null && pwd || true)"
        candidates+=("${script_dir}" "${script_parent}")
    fi

    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        normalized="$(cd "${candidate}" 2>/dev/null && pwd || true)"
        [[ -n "${normalized}" ]] || continue

        if [[ -f "${normalized}/slurm/run_mcdc_vv_electron.py" ]]; then
            ROOT_DIR="${normalized}"
            RUNNER="${ROOT_DIR}/slurm/run_mcdc_vv_electron.py"
            return 0
        fi

        if [[ -f "${normalized}/run_mcdc_vv_electron.py" ]] && [[ "$(basename "${normalized}")" == "slurm" ]]; then
            ROOT_DIR="$(cd "${normalized}/.." && pwd)"
            RUNNER="${normalized}/run_mcdc_vv_electron.py"
            return 0
        fi
    done

    return 1
}

if ! resolve_paths; then
    echo "Runner not found." >&2
    echo "Checked from ROOT_DIR, SLURM_SUBMIT_DIR, pwd, and the script location." >&2
    echo "If needed, submit with ROOT_DIR=/path/to/MCDC-VV-electron sbatch ..." >&2
    exit 1
fi

cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-vv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export SLURM_CPU_BIND="${SLURM_CPU_BIND:-cores}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

JOB_SCRATCH_BASE="${SLURM_TMPDIR:-/tmp/${USER}}"
JOB_SCRATCH="${JOB_SCRATCH_BASE}/mcdc-vv-electron-${SLURM_JOB_ID:-$$}"
mkdir -p "${JOB_SCRATCH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${JOB_SCRATCH}/numba-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${JOB_SCRATCH}/xdg-cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

if [[ -z "${CASE_ROOT:-}" ]]; then
    if [[ -d "${ROOT_DIR}/verification/benchmark/continuous_energy" ]]; then
        CASE_ROOT="${ROOT_DIR}/verification/benchmark/continuous_energy"
    elif [[ -d "${ROOT_DIR}/MCDC-VV-electron/verification/benchmark/continuous_energy" ]]; then
        CASE_ROOT="${ROOT_DIR}/MCDC-VV-electron/verification/benchmark/continuous_energy"
    elif [[ -d "${ROOT_DIR}/MCDC/mcdc-vv" ]]; then
        CASE_ROOT="${ROOT_DIR}/MCDC/mcdc-vv"
    else
        CASE_ROOT="${ROOT_DIR}"
    fi
fi

if [[ -z "${MCDC_ROOT:-}" ]]; then
    if [[ -d "${ROOT_DIR}/../MCDC/mcdc" ]]; then
        MCDC_ROOT="$(cd "${ROOT_DIR}/../MCDC" && pwd)"
    elif [[ -d "${ROOT_DIR}/MCDC/mcdc" ]]; then
        MCDC_ROOT="${ROOT_DIR}/MCDC"
    else
        echo "Correct MCDC repository not found." >&2
        echo "Expected sibling path like: ${ROOT_DIR}/../MCDC" >&2
        echo "Set MCDC_ROOT explicitly if your layout is different." >&2
        exit 1
    fi
fi
if [[ -z "${DATA_LIBRARY:-}" ]]; then
    if [[ -d "${ROOT_DIR}/../mcdc_data" ]]; then
        DATA_LIBRARY="$(cd "${ROOT_DIR}/../mcdc_data" && pwd)"
    elif [[ -d "${ROOT_DIR}/mcdc_data" ]]; then
        DATA_LIBRARY="$(cd "${ROOT_DIR}/mcdc_data" && pwd)"
    else
        DATA_LIBRARY=""
    fi
fi
MODE="${MODE:-numba}"
TARGET="${TARGET:-cpu}"
MPI_EXEC="${MPI_EXEC:-srun}"
MPI_PROCS="${MPI_PROCS:-auto}"
MIN_PARTICLES_PER_RANK="${MIN_PARTICLES_PER_RANK:-2000}"
PROBLEM="${PROBLEM:-}"
MATERIAL="${MATERIAL:-}"
ENERGY="${ENERGY:-}"
ANGLE="${ANGLE:-}"
PARTICLES="${PARTICLES:-10000}"
CACHING="${CACHING:-1}"
CLEAR_CACHE="${CLEAR_CACHE:-0}"
RUN_ONLY="${RUN_ONLY:-0}"
PROCESS_ONLY="${PROCESS_ONLY:-0}"
LIST_ONLY="${LIST_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUNTIME_OUTPUT="${RUNTIME_OUTPUT:-0}"
PROGRESS_BAR="${PROGRESS_BAR:-0}"

export MCDC_VV_MIN_PARTICLES_PER_RANK="${MCDC_VV_MIN_PARTICLES_PER_RANK:-${MIN_PARTICLES_PER_RANK}}"

if [[ ! -d "${CASE_ROOT}" ]]; then
    echo "Case root not found: ${CASE_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${MCDC_ROOT}/mcdc" ]]; then
    echo "Requested MCDC root does not contain the mcdc package: ${MCDC_ROOT}" >&2
    exit 1
fi

if [[ -n "${DATA_LIBRARY}" && ! -d "${DATA_LIBRARY}" ]]; then
    echo "Data library directory not found: ${DATA_LIBRARY}" >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}"
    "${RUNNER}"
    "--case-root" "${CASE_ROOT}"
    "--mcdc-root" "${MCDC_ROOT}"
    "--mode" "${MODE}"
    "--target" "${TARGET}"
    "--mpi-exec" "${MPI_EXEC}"
    "--mpi-procs" "${MPI_PROCS}"
)

if [[ "${CACHING}" == "1" ]]; then
    CMD+=("--caching")
else
    CMD+=("--no-caching")
fi

if [[ "${PROGRESS_BAR}" == "1" ]]; then
    CMD+=("--progress-bar")
else
    CMD+=("--no-progress-bar")
fi

if [[ "${CLEAR_CACHE}" == "1" ]]; then
    CMD+=("--clear-cache")
fi

if [[ "${RUN_ONLY}" == "1" ]]; then
    CMD+=("--run-only")
fi

if [[ "${PROCESS_ONLY}" == "1" ]]; then
    CMD+=("--process-only")
fi

if [[ "${LIST_ONLY}" == "1" ]]; then
    CMD+=("--list")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    CMD+=("--dry-run")
fi

if [[ "${RUNTIME_OUTPUT}" == "1" ]]; then
    CMD+=("--runtime-output")
fi

if [[ -n "${DATA_LIBRARY}" ]]; then
    CMD+=("--data-library" "${DATA_LIBRARY}")
fi

if [[ -n "${PROBLEM}" ]]; then
    CMD+=("--problem" "${PROBLEM}")
fi

if [[ -n "${MATERIAL}" ]]; then
    CMD+=("--material" "${MATERIAL}")
fi

if [[ -n "${ENERGY}" ]]; then
    CMD+=("--energy" "${ENERGY}")
fi

if [[ -n "${ANGLE}" ]]; then
    CMD+=("--angle" "${ANGLE}")
fi

if [[ -n "${PARTICLES}" ]]; then
    CMD+=("--particles" "${PARTICLES}")
fi

if (( $# > 0 )); then
    CMD+=("$@")
fi

echo "Dane job configuration:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  CASE_ROOT=${CASE_ROOT}"
echo "  MCDC_ROOT=${MCDC_ROOT}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  MPI_EXEC=${MPI_EXEC}"
echo "  MPI_PROCS=${MPI_PROCS}"
echo "  MCDC_VV_MIN_PARTICLES_PER_RANK=${MCDC_VV_MIN_PARTICLES_PER_RANK}"
echo "  NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
