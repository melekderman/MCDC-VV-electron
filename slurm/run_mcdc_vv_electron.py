#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.resolve()
DEFAULT_MCDC_ROOT = (WORKSPACE_ROOT / "MCDC").resolve()
DEFAULT_CASE_ROOT_CANDIDATES = (
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "verification" / "benchmark" / "continuous_energy",
    WORKSPACE_ROOT / "MCDC-VV-electron",
    WORKSPACE_ROOT / "MCDC-VV-electron" / "verification" / "benchmark" / "continuous_energy",
    WORKSPACE_ROOT / "MCDC" / "mcdc-vv",
    WORKSPACE_ROOT.parent / "MCDC",
    WORKSPACE_ROOT.parent / "MCDC" / "mcdc-vv",
)
PROCESS_DATA_LIBRARY_ENV = "MCDC_VV_PROCESS_DATA_LIBRARY_DIR"
ADDITIONAL_DATA_LIBRARY_ENVS = ("MCDC_VV_DATA_LIBRARY_DIR", "MCDC_LIB")
MIN_PARTICLES_PER_RANK_ENV = "MCDC_VV_MIN_PARTICLES_PER_RANK"
DEFAULT_RUN_MODE = "numba"
DEFAULT_TARGET = "cpu"
DEFAULT_MPI_PROCS = "auto"
DEFAULT_MIN_PARTICLES_PER_RANK = 3000


@dataclass(frozen=True)
class CaseInfo:
    prefix_parts: tuple[str, ...]
    material: str
    energy_keV: int
    angle_deg: int
    case_dir: Path

    @property
    def problem(self) -> str | None:
        return self.prefix_parts[0] if self.prefix_parts else None

    @property
    def input_path(self) -> Path:
        return self.case_dir / "input.py"

    @property
    def reference_path(self) -> Path:
        return self.case_dir / "reference.npz"

    @property
    def energy_eV(self) -> float:
        return float(self.energy_keV) * 1.0e3

    @property
    def results_dir(self) -> Path:
        return self.case_dir.parents[1] / "results"

    @property
    def label(self) -> str:
        parts = []
        if self.problem:
            parts.append(self.problem)
        parts.append(self.material)
        parts.append(f"{self.energy_keV} keV")
        parts.append(f"{self.angle_deg} deg")
        return " | ".join(parts)


def parse_positive_int(value: str) -> int:
    parsed = float(value)
    if parsed <= 0 or not parsed.is_integer():
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return int(parsed)


def parse_mpi_procs(value: str) -> str | int:
    text = value.strip().lower()
    if text == "auto":
        return "auto"
    return parse_positive_int(value)


def resolve_min_particles_per_rank() -> int:
    raw_value = os.environ.get(MIN_PARTICLES_PER_RANK_ENV)
    if not raw_value:
        return DEFAULT_MIN_PARTICLES_PER_RANK
    try:
        parsed = int(float(raw_value))
    except ValueError as exc:
        raise ValueError(
            f"{MIN_PARTICLES_PER_RANK_ENV} must be a positive integer, got {raw_value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{MIN_PARTICLES_PER_RANK_ENV} must be a positive integer, got {raw_value!r}."
        )
    return parsed


def format_scientific_int(value: int) -> str:
    return f"{float(value):.0e}".replace("+", "")


def normalize_case_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    benchmark_root = path / "verification" / "benchmark" / "continuous_energy"
    if benchmark_root.is_dir():
        return benchmark_root.resolve()
    return path


def resolve_case_root(cli_value: str | None) -> Path:
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value))
    candidates.extend(DEFAULT_CASE_ROOT_CANDIDATES)

    for candidate in candidates:
        normalized = normalize_case_root(candidate)
        if normalized.is_dir():
            return normalized

    tried = "\n".join(f"  - {normalize_case_root(path)}" for path in candidates)
    raise FileNotFoundError(
        "Could not resolve the case root.\n"
        "Use --case-root to point to MCDC-VV-electron or a benchmark directory.\n"
        f"Tried:\n{tried}"
    )


def resolve_mcdc_root(cli_value: str | None, case_root: Path | None = None) -> Path:
    candidates: list[Path] = []

    if cli_value:
        candidates.append(Path(cli_value).expanduser())

    env_value = os.environ.get("MCDC_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    if case_root is not None:
        vv_root = infer_repo_root(case_root)
        candidates.extend([vv_root.parent / "MCDC", vv_root / "MCDC"])

    candidates.extend(
        [
            DEFAULT_MCDC_ROOT,
            WORKSPACE_ROOT.parent / "MCDC",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "mcdc").is_dir():
            return resolved

    tried = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        "Could not resolve the MCDC source root.\n"
        "Use --mcdc-root to point to the repository that contains the `mcdc` package.\n"
        f"Tried:\n{tried}"
    )


def infer_repo_root(case_root: Path) -> Path:
    case_root = case_root.resolve()

    for candidate in (case_root, *case_root.parents):
        if candidate.name == "MCDC-VV-electron":
            return candidate

    if (case_root / "mcdc_data").is_dir():
        return case_root

    if (case_root / "electron-vv-data").is_dir():
        return case_root

    return WORKSPACE_ROOT


def is_valid_data_library_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.h5"))


def discover_cases(case_root: Path) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    energy_pattern = re.compile(r"^E(\d+)keV$")
    angle_pattern = re.compile(r"^th(\d+)deg$")

    for input_path in sorted(case_root.rglob("input.py")):
        rel_parts = input_path.relative_to(case_root).parts
        if len(rel_parts) < 4:
            continue

        prefix_parts = tuple(rel_parts[:-4])
        material, energy_dir, angle_dir, filename = rel_parts[-4:]
        if filename != "input.py":
            continue

        energy_match = energy_pattern.match(energy_dir)
        angle_match = angle_pattern.match(angle_dir)
        if not energy_match or not angle_match:
            continue

        cases.append(
            CaseInfo(
                prefix_parts=prefix_parts,
                material=material,
                energy_keV=int(energy_match.group(1)),
                angle_deg=int(angle_match.group(1)),
                case_dir=input_path.parent,
            )
        )

    return cases


def filter_cases(
    cases: list[CaseInfo],
    problem: str | None = None,
    material: str | None = None,
    energy_keV: float | None = None,
    angle_deg: float | None = None,
) -> list[CaseInfo]:
    selected = cases

    if problem is not None:
        selected = [
            case
            for case in selected
            if case.problem is not None and case.problem.lower() == problem.lower()
        ]

    if material is not None:
        selected = [case for case in selected if case.material.lower() == material.lower()]

    if energy_keV is not None:
        selected = [
            case
            for case in selected
            if math.isclose(case.energy_keV, energy_keV, rel_tol=0.0, abs_tol=1.0e-9)
        ]

    if angle_deg is not None:
        selected = [
            case
            for case in selected
            if math.isclose(case.angle_deg, angle_deg, rel_tol=0.0, abs_tol=1.0e-9)
        ]

    return selected


def _find_input_value(name: str, text: str) -> str:
    match = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*(#.*)?$", text, re.M)
    if not match:
        raise ValueError(f"{name} not found in input.py")
    return match.group(1).strip()


def load_case_metadata(case: CaseInfo) -> dict[str, float | str]:
    text = case.input_path.read_text(encoding="utf-8")
    return {
        "material_symbol": _find_input_value("MATERIAL_SYMBOL", text).strip("\"' "),
        "energy_eV": float(eval(_find_input_value("ENERGY", text), {}, {})),
        "angle_deg": float(eval(_find_input_value("ANGLE", text), {}, {})),
        "csda_range": float(eval(_find_input_value("CSDA_RANGE", text), {}, {})),
        "rho_g_cm3": float(eval(_find_input_value("RHO_G_CM3", text), {}, {})),
        "n_particles": int(float(eval(_find_input_value("N_PARTICLES", text), {}, {}))),
    }


def resolve_data_library_dir(cli_value: str | None, case_root: Path, mcdc_root: Path) -> Path:
    repo_root = infer_repo_root(case_root)
    candidates: list[Path] = []

    if cli_value:
        candidates.append(Path(cli_value).expanduser())

    for env_name in (PROCESS_DATA_LIBRARY_ENV, *ADDITIONAL_DATA_LIBRARY_ENVS):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            repo_root.parent / "mcdc_data",
            repo_root / "mcdc_data",
            case_root / "mcdc_data",
            mcdc_root.parent / "mcdc_data",
            WORKSPACE_ROOT / "mcdc_data",
            repo_root / "electron-vv-data",
            case_root / "electron-vv-data",
            WORKSPACE_ROOT / "electron-vv-data",
            WORKSPACE_ROOT / "PyEEDL" / "mcdc_data",
            mcdc_root / "mcdc-regression_test_data",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        resolved = candidate.resolve()
        if is_valid_data_library_dir(resolved):
            return resolved

    tried = "\n".join(f"  - {path.resolve()}" for path in unique_candidates)
    raise FileNotFoundError(
        "Could not resolve the MCDC electron data library directory.\n"
        "Set --data-library or one of the supported environment variables.\n"
        "Common layouts are sibling `mcdc_data` and legacy `electron-vv-data`.\n"
        f"Tried:\n{tried}"
    )


def build_runtime_env(data_library_dir: Path, mcdc_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_python_path = str(mcdc_root)
    current_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_python_path}{os.pathsep}{current_python_path}"
        if current_python_path
        else repo_python_path
    )
    env[PROCESS_DATA_LIBRARY_ENV] = str(data_library_dir)
    for env_name in ADDITIONAL_DATA_LIBRARY_ENVS:
        env[env_name] = str(data_library_dir)
    return env


def resolve_imported_mcdc_path(
    python_executable: str,
    runtime_env: dict[str, str],
) -> Path:
    probe = (
        "from pathlib import Path; "
        "import mcdc; "
        "print(Path(mcdc.__file__).resolve())"
    )
    completed = subprocess.run(
        [python_executable, "-c", probe],
        env=runtime_env,
        check=True,
        text=True,
        capture_output=True,
    )
    imported_path = completed.stdout.strip()
    if not imported_path:
        raise ValueError("Could not determine the imported MCDC module path.")
    return Path(imported_path).resolve()


def validate_mcdc_import(
    python_executable: str,
    runtime_env: dict[str, str],
    mcdc_root: Path,
) -> Path:
    imported_path = resolve_imported_mcdc_path(python_executable, runtime_env)
    expected_root = mcdc_root.resolve()
    try:
        imported_path.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(
            "The Python environment is not importing MCDC from the requested repository root.\n"
            f"Requested MCDC root: {expected_root}\n"
            f"Imported module path: {imported_path}"
        ) from exc
    return imported_path


def resolve_mpi_exec(cli_value: str | None) -> str | None:
    candidates: list[str] = []

    if cli_value:
        candidates.append(cli_value)

    candidates.extend(["srun", "mpiexec", "mpirun"])

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def resolve_mpi_procs(requested: str | int, particle_count: int, mpi_exec: str | None) -> int:
    if requested == "auto":
        if mpi_exec is None:
            return 1

        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        if slurm_ntasks and slurm_ntasks.isdigit():
            available_slots = int(slurm_ntasks)
        else:
            available_slots = os.cpu_count() or 1

        min_particles_per_rank = resolve_min_particles_per_rank()
        particle_limited_slots = max(1, particle_count // min_particles_per_rank)
        return max(1, min(available_slots, particle_count, particle_limited_slots))

    requested_int = int(requested)
    if requested_int > 1 and mpi_exec is None:
        raise FileNotFoundError(
            "MPI execution was requested but no launcher was found. "
            "Use --mpi-exec or make `srun`, `mpiexec`, or `mpirun` available."
        )
    return max(1, min(requested_int, particle_count))


def build_mcdc_command(
    python_executable: str,
    input_name: str,
    mode: str,
    target: str,
    caching: bool,
    clear_cache: bool,
    progress_bar: bool,
    runtime_output: bool,
    mpi_exec: str | None,
    mpi_procs: int,
) -> list[str]:
    command: list[str] = []

    if mpi_procs > 1:
        if mpi_exec is None:
            raise FileNotFoundError(
                "MPI execution was requested but no launcher was found. "
                "Use --mpi-exec or make `srun`, `mpiexec`, or `mpirun` available."
            )
        command.extend([mpi_exec, "-n", str(mpi_procs)])

    command.extend(
        [
            python_executable,
            input_name,
            "--mode",
            mode,
            "--target",
            target,
        ]
    )

    command.append("--caching" if caching else "--no_caching")
    if clear_cache:
        command.append("--clear_cache")
    command.append("--progress_bar" if progress_bar else "--no-progress_bar")
    if runtime_output:
        command.append("--runtime_output")

    return command


def replace_assignment(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{name}\s*=\s*)(.+?)(\s*(#.*)?$)", re.M)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{value}{match.group(3)}"

    updated, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace assignment for {name}")
    return updated


def build_runtime_input(case: CaseInfo, particle_override: int | None) -> str:
    text = case.input_path.read_text(encoding="utf-8")
    text = replace_assignment(text, "MATERIAL_SYMBOL", repr(case.material))
    text = replace_assignment(text, "ENERGY", f"{case.energy_eV:.12g}")
    text = replace_assignment(text, "ANGLE", f"{float(case.angle_deg):.12g}")

    if particle_override is not None:
        text = replace_assignment(text, "N_PARTICLES", str(particle_override))

    return text


def snapshot_h5_files(case_dir: Path) -> dict[str, int]:
    return {path.name: path.stat().st_mtime_ns for path in case_dir.glob("*.h5") if path.is_file()}


def resolve_h5_output(case_dir: Path, before_run: dict[str, int] | None = None) -> Path:
    candidates = [path for path in case_dir.glob("*.h5") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No HDF5 output found in {case_dir}")

    if before_run is not None:
        fresh = [path for path in candidates if before_run.get(path.name) != path.stat().st_mtime_ns]
        if fresh:
            return max(fresh, key=lambda path: path.stat().st_mtime_ns)

    answer_path = case_dir / "answer.h5"
    if answer_path.exists():
        return answer_path

    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run_case(
    case: CaseInfo,
    particle_override: int | None,
    data_library_dir: Path,
    *,
    mcdc_root: Path,
    mode: str,
    target: str,
    caching: bool,
    clear_cache: bool,
    progress_bar: bool,
    runtime_output: bool,
    mpi_exec: str | None,
    mpi_procs_request: str | int,
) -> Path:
    runtime_input = build_runtime_input(case, particle_override)
    env = build_runtime_env(data_library_dir, mcdc_root)
    before_run = snapshot_h5_files(case.case_dir)
    metadata = load_case_metadata(case)
    particle_count = particle_override or int(metadata["n_particles"])
    mpi_procs = resolve_mpi_procs(mpi_procs_request, particle_count, mpi_exec)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_runtime_input.py",
        prefix="tmp_",
        dir=case.case_dir,
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(runtime_input)
        temp_input_path = Path(handle.name)

    try:
        command = build_mcdc_command(
            python_executable=sys.executable,
            input_name=temp_input_path.name,
            mode=mode,
            target=target,
            caching=caching,
            clear_cache=clear_cache,
            progress_bar=progress_bar,
            runtime_output=runtime_output,
            mpi_exec=mpi_exec,
            mpi_procs=mpi_procs,
        )
        subprocess.run(command, cwd=case.case_dir, env=env, check=True)
    finally:
        temp_input_path.unlink(missing_ok=True)

    return resolve_h5_output(case.case_dir, before_run=before_run)


def read_h5_dataset(file_handle, paths: list[str]):
    import numpy as np

    for path in paths:
        if path in file_handle:
            return np.atleast_1d(file_handle[path][()]).astype(float)
    raise KeyError(f"Could not find any of the datasets: {', '.join(paths)}")


def read_h5_scalar_int(file_handle, paths: list[str]) -> int | None:
    for path in paths:
        if path in file_handle:
            return int(file_handle[path][()])
    return None


def load_reference_series(reference_path: Path):
    import numpy as np

    with np.load(reference_path) as ref:
        data = {
            "theo_fmr": ref["fmr_theo_tiger"].astype(float),
            "theo_edep": ref["edep_theo_tiger"].astype(float),
        }

        if "fmr_exp_lw_A" in ref and "edep_exp_lw_A" in ref:
            data["exp_A_fmr"] = ref["fmr_exp_lw_A"].astype(float)
            data["exp_A_edep"] = ref["edep_exp_lw_A"].astype(float)
        if "fmr_exp_lw_B" in ref and "edep_exp_lw_B" in ref:
            data["exp_B_fmr"] = ref["fmr_exp_lw_B"].astype(float)
            data["exp_B_edep"] = ref["edep_exp_lw_B"].astype(float)
        if "fmr_exp_lw" in ref and "edep_exp_lw" in ref:
            data["exp_fmr"] = ref["fmr_exp_lw"].astype(float)
            data["exp_edep"] = ref["edep_exp_lw"].astype(float)

    return data


def process_case(
    case: CaseInfo,
    h5_path: Path | None = None,
    particle_override: int | None = None,
) -> Path:
    try:
        import h5py
        import matplotlib as mpl
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Figure generation requires `numpy`, `h5py`, and `matplotlib` "
            "in the active Python environment."
        ) from exc

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    if not case.reference_path.exists():
        raise FileNotFoundError(f"reference.npz not found in {case.case_dir}")

    if h5_path is None:
        h5_path = resolve_h5_output(case.case_dir)

    metadata = load_case_metadata(case)
    effective_particles = particle_override or metadata["n_particles"]
    reference_data = load_reference_series(case.reference_path)

    with h5py.File(h5_path, "r") as file:
        z = read_h5_dataset(file, ["tallies/edep/grid/z", "tallies/mesh_tally_0/grid/z"])
        edep = read_h5_dataset(
            file,
            [
                "tallies/edep/energy_deposition/mean",
                "tallies/mesh_tally_0/edep/mean",
            ],
        )
        h5_particles = read_h5_scalar_int(file, ["input_deck/setting/N_particle"])
        if h5_particles is not None:
            effective_particles = h5_particles

    dz = z[1:] - z[:-1]
    z_centers = 0.5 * (z[:-1] + z[1:])
    total_thickness = float(metadata["csda_range"]) / float(metadata["rho_g_cm3"])
    mcdc_fmr = z_centers / total_thickness
    edep_mcdc = edep / float(metadata["rho_g_cm3"]) / dz / 1.0e6

    energy_mev = case.energy_keV / 1000.0
    angle_deg = float(metadata["angle_deg"])
    case.results_dir.mkdir(parents=True, exist_ok=True)
    out_name = (
        f"fig_{case.material}_{energy_mev:g}MeV_"
        f"th{case.angle_deg}_{format_scientific_int(int(effective_particles))}.png"
    )
    out_path = case.results_dir / out_name

    plt.figure(figsize=(14, 9), constrained_layout=True)
    mpl.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 29,
            "axes.labelsize": 25,
            "legend.fontsize": 22,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
        }
    )

    plt.minorticks_on()
    plt.grid(
        True, which="major", linestyle="-", linewidth=0.8, color="#3B3A3AFF", alpha=0.5
    )
    plt.grid(
        True, which="minor", linestyle=":", linewidth=0.6, color="#8A8686", alpha=0.7
    )

    plt.plot(
        mcdc_fmr,
        edep_mcdc,
        label="MCDC Simulation",
        marker="o",
        markersize=9,
        linewidth=3.0,
    )
    plt.plot(
        reference_data["theo_fmr"],
        reference_data["theo_edep"],
        label="TIGER Theoretical",
        marker="s",
        markersize=9,
        linewidth=3.0,
    )

    if "exp_A_fmr" in reference_data:
        plt.plot(
            reference_data["exp_A_fmr"],
            reference_data["exp_A_edep"],
            label="Lockwood Experimental A",
            marker="D",
            markersize=9,
            linewidth=3.0,
        )
    if "exp_B_fmr" in reference_data:
        plt.plot(
            reference_data["exp_B_fmr"],
            reference_data["exp_B_edep"],
            label="Lockwood Experimental B",
            marker="*",
            markersize=9,
            linewidth=3.0,
        )
    if "exp_fmr" in reference_data:
        plt.plot(
            reference_data["exp_fmr"],
            reference_data["exp_edep"],
            label="Lockwood Experimental",
            marker="D",
            markersize=9,
            linewidth=3.0,
        )

    plt.xlim(0, 1)
    plt.xlabel("Fraction of Mean Range", labelpad=15)
    plt.ylabel("Energy Deposition (MeV/g/cm²)", labelpad=15)
    plt.title(
        f"Energy Deposition of {energy_mev:g} MeV Electrons in "
        f"{metadata['material_symbol']} at {angle_deg:g}° Incidence",
        pad=20,
    )
    plt.legend()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()

    return out_path


def print_case_table(cases: list[CaseInfo], case_root: Path) -> None:
    for case in cases:
        rel_dir = case.case_dir.relative_to(case_root)
        print(f"- {case.label} -> {rel_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MCDC VV electron benchmark cases and generate result figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case-root",
        help=(
            "Path to the case directory root. This can point either to "
            "`MCDC-VV-electron`, `verification/benchmark/continuous_energy`, "
            "or the legacy `MCDC/mcdc-vv` directory."
        ),
    )
    parser.add_argument(
        "--mcdc-root",
        help="Path to the MCDC repository that contains the `mcdc` package.",
    )
    parser.add_argument(
        "--mode",
        choices=["python", "numba", "numba_debug"],
        default=DEFAULT_RUN_MODE,
        help="MCDC execution mode.",
    )
    parser.add_argument(
        "--target",
        choices=["cpu", "gpu"],
        default=DEFAULT_TARGET,
        help="MCDC execution target.",
    )
    parser.add_argument(
        "--mpi-procs",
        type=parse_mpi_procs,
        default=DEFAULT_MPI_PROCS,
        help="MPI process count. Use `auto` to respect Slurm task count or local CPU availability.",
    )
    parser.add_argument(
        "--mpi-exec",
        help="MPI launcher to use, for example `srun`, `mpiexec`, or `mpirun`.",
    )
    parser.add_argument(
        "--caching",
        dest="caching",
        action="store_true",
        help="Enable MCDC/Numba caching for faster repeated runs.",
    )
    parser.add_argument(
        "--no-caching",
        dest="caching",
        action="store_false",
        help="Disable MCDC/Numba caching.",
    )
    parser.set_defaults(caching=True)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear MCDC/Numba caches before the run.",
    )
    parser.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action="store_true",
        help="Show the MCDC progress bar during transport.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable the MCDC progress bar for cleaner logs.",
    )
    parser.set_defaults(progress_bar=False)
    parser.add_argument(
        "--runtime-output",
        action="store_true",
        help="Ask MCDC to emit extra runtime datasets.",
    )
    parser.add_argument(
        "--problem",
        help="Filter cases by the first path component before the material directory.",
    )
    parser.add_argument(
        "--material",
        help="Filter cases by material name, for example Al.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        help="Filter cases by incident energy in keV, for example 500 or 1000.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        help="Filter cases by incident angle in degrees, for example 0 or 60.",
    )
    parser.add_argument(
        "--particles",
        type=parse_positive_int,
        help="Override the particle count for selected runs. Scientific notation is allowed.",
    )
    parser.add_argument(
        "--data-library",
        help="Path to the processed MCDC electron data library directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching cases and exit.",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Run selected cases but skip figure generation.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip simulation and only generate figures from existing HDF5 outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved configuration without running anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run_only and args.process_only:
        parser.error("--run-only and --process-only cannot be used together.")

    case_root = resolve_case_root(args.case_root)
    mcdc_root = resolve_mcdc_root(args.mcdc_root, case_root)
    cases = discover_cases(case_root)
    selected = filter_cases(
        cases,
        problem=args.problem,
        material=args.material,
        energy_keV=args.energy,
        angle_deg=args.angle,
    )

    if not selected:
        print("No matching cases found.\nAvailable cases:")
        print_case_table(cases, case_root)
        return 1

    data_library_dir = None
    mpi_exec = resolve_mpi_exec(args.mpi_exec)
    imported_mcdc_path = None
    if args.dry_run and not args.process_only:
        data_library_dir = resolve_data_library_dir(args.data_library, case_root, mcdc_root)

    if args.list or args.dry_run:
        resolved_mpi_procs = None
        if not args.process_only:
            if data_library_dir is None:
                data_library_dir = resolve_data_library_dir(args.data_library, case_root, mcdc_root)
            runtime_env = build_runtime_env(data_library_dir, mcdc_root)
            imported_mcdc_path = validate_mcdc_import(
                python_executable=sys.executable,
                runtime_env=runtime_env,
                mcdc_root=mcdc_root,
            )
            first_case = selected[0]
            first_metadata = load_case_metadata(first_case)
            first_particles = args.particles or int(first_metadata["n_particles"])
            resolved_mpi_procs = resolve_mpi_procs(args.mpi_procs, first_particles, mpi_exec)

        print("Selected cases:")
        print_case_table(selected, case_root)
        print(f"\nCase root: {case_root}")
        print(f"MCDC root: {mcdc_root}")
        print(f"Run mode: {args.mode}")
        print(f"Target: {args.target}")
        print(f"MPI launcher: {mpi_exec or '<not found>'}")
        print(f"MPI processes: {args.mpi_procs}")
        if resolved_mpi_procs is not None:
            print(f"Resolved MPI processes: {resolved_mpi_procs}")
        print(f"Caching: {args.caching}")
        print(f"Progress bar: {args.progress_bar}")
        if data_library_dir is not None:
            print(f"Data library: {data_library_dir}")
        if imported_mcdc_path is not None:
            print(f"Validated MCDC import: {imported_mcdc_path}")
        if args.particles is not None:
            print(f"Particle override: {args.particles}")
        if args.dry_run or args.list:
            return 0

    figure_paths: list[Path] = []
    h5_paths: list[Path] = []

    if not args.process_only and data_library_dir is None:
        data_library_dir = resolve_data_library_dir(args.data_library, case_root, mcdc_root)

    if not args.process_only:
        if imported_mcdc_path is None:
            runtime_env = build_runtime_env(data_library_dir, mcdc_root)
            imported_mcdc_path = validate_mcdc_import(
                python_executable=sys.executable,
                runtime_env=runtime_env,
                mcdc_root=mcdc_root,
            )
        print(f"Validated MCDC import: {imported_mcdc_path}")

    for index, case in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {case.label}")

        h5_path = None
        if not args.process_only:
            metadata = load_case_metadata(case)
            particle_count = args.particles or int(metadata["n_particles"])
            mpi_procs = resolve_mpi_procs(args.mpi_procs, particle_count, mpi_exec)
            print(f"Using case root: {case_root}")
            print(f"Using MCDC root: {mcdc_root}")
            print(f"Using data library: {data_library_dir}")
            print(f"Using mode: {args.mode}")
            print(f"Using target: {args.target}")
            print(f"Using MPI launcher: {mpi_exec or '<none>'}")
            print(f"Using MPI processes: {mpi_procs}")
            if imported_mcdc_path is not None:
                print(f"Using imported MCDC module: {imported_mcdc_path}")
            print(f"Using caching: {args.caching}")
            print(f"Running case in {case.case_dir}")
            h5_path = run_case(
                case,
                args.particles,
                data_library_dir,
                mcdc_root=mcdc_root,
                mode=args.mode,
                target=args.target,
                caching=args.caching,
                clear_cache=args.clear_cache,
                progress_bar=args.progress_bar,
                runtime_output=args.runtime_output,
                mpi_exec=mpi_exec,
                mpi_procs_request=args.mpi_procs,
            )
            h5_paths.append(h5_path)
            print(f"Produced HDF5: {h5_path}")

        if not args.run_only:
            process_input = h5_path if h5_path is not None else resolve_h5_output(case.case_dir)
            print(f"Generating figure from {process_input}")
            figure_path = process_case(case, h5_path=process_input, particle_override=args.particles)
            figure_paths.append(figure_path)
            print(f"Saved figure: {figure_path}")

    print("\nCompleted.")
    if h5_paths:
        print(f"HDF5 files: {len(h5_paths)}")
    if figure_paths:
        print(f"Figures: {len(figure_paths)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        KeyError,
        ModuleNotFoundError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
