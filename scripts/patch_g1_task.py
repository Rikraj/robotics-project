#!/usr/bin/env python3
"""Idempotently add baseline/domain-randomization controls to Isaac Lab 3.0 G1 flat cfg.

Run from anywhere:
  python scripts/patch_g1_task.py --isaaclab-root ~/workspace/IsaacLab --mode baseline

The selected mode is read at environment construction time from G1_DR_MODE, so one
patched checkout can be reused for both conditions without rewriting source each run.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

MARKER_START = "# BEGIN HUMANOID_DR_PROJECT_PATCH"
MARKER_END = "# END HUMANOID_DR_PROJECT_PATCH"

PATCH_LINES = [
    MARKER_START,
    "import os as _humanoid_dr_os",
    "_humanoid_dr_mode = _humanoid_dr_os.environ.get('G1_DR_MODE', 'baseline').strip().lower()",
    "if _humanoid_dr_mode not in {'baseline', 'randomized'}:",
    "    raise ValueError(f\"G1_DR_MODE must be 'baseline' or 'randomized', got {_humanoid_dr_mode!r}\")",
    "# Hold unrelated dynamics fixed so the comparison isolates contact friction and pushes.",
    "self.events.add_base_mass = None",
    "self.events.base_com = None",
    "# Randomize/materialize only the G1 ankle-roll (foot contact) collision geometry.",
    "self.events.physics_material.params['asset_cfg'].body_names = '.*_ankle_roll_link'",
    "if _humanoid_dr_mode == 'baseline':",
    "    self.events.physics_material.params['static_friction_range'] = (1.0, 1.0)",
    "    self.events.physics_material.params['dynamic_friction_range'] = (1.0, 1.0)",
    "    self.events.push_robot = None",
    "else:",
    "    self.events.physics_material.params['static_friction_range'] = (0.5, 1.25)",
    "    self.events.physics_material.params['dynamic_friction_range'] = (0.5, 1.25)",
    "    # Newton uses a single friction coefficient; dynamic friction/buckets are PhysX-only.",
    "    self.events.physics_material.params['num_buckets'] = 64",
    "    self.events.push_robot.interval_range_s = (5.0, 5.0)",
    "    self.events.push_robot.params['velocity_range'] = {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}",
    MARKER_END,
]


def find_target(root: Path) -> Path:
    candidates = list(root.glob("source/isaaclab_tasks/isaaclab_tasks/**/g1/flat_env_cfg.py"))
    candidates = [p for p in candidates if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one G1 flat_env_cfg.py under source/isaaclab_tasks; "
            f"found {len(candidates)}. Confirm this is the Isaac Lab 3.0.0 source checkout."
        )
    return candidates[0]


def patch_file(path: Path, dry_run: bool = False) -> bool:
    source = path.read_text(encoding="utf-8")
    if MARKER_START in source:
        print(f"Patch already present: {path}")
        return False

    tree = ast.parse(source, filename=str(path))
    target_class = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1FlatEnvCfg"),
        None,
    )
    if target_class is None:
        raise RuntimeError("G1FlatEnvCfg class not found; refusing to patch an unknown file.")
    method = next(
        (n for n in target_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__post_init__"),
        None,
    )
    if method is None:
        raise RuntimeError("G1FlatEnvCfg.__post_init__ not found; source layout differs from Isaac Lab 3.0.0.")

    super_call = None
    for node in method.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "__post_init__"
                and isinstance(call.func.value, ast.Call)
                and isinstance(call.func.value.func, ast.Name)
                and call.func.value.func.id == "super"
            ):
                super_call = node
                break
    if super_call is None:
        raise RuntimeError("Could not locate super().__post_init__() in G1FlatEnvCfg; refusing to patch.")

    lines = source.splitlines()
    insert_at = super_call.end_lineno
    indent = " " * (super_call.col_offset)
    block = [indent + line if line else "" for line in PATCH_LINES]
    new_lines = lines[:insert_at] + block + lines[insert_at:]
    updated = "\n".join(new_lines) + ("\n" if source.endswith("\n") else "")
    ast.parse(updated, filename=str(path))  # syntax-check before touching original

    if dry_run:
        print(f"Dry run: would patch {path}")
        return True

    backup = path.with_suffix(path.suffix + ".humanoid_dr.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Patched: {path}\nBackup:  {backup}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, required=True, help="Isaac Lab 3.0.0 repository root")
    parser.add_argument("--mode", choices=["baseline", "randomized"], default="baseline", help="Informational only; runtime mode is selected by G1_DR_MODE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.isaaclab_root.expanduser().resolve()
    if not (root / "pyproject.toml").exists() or not (root / "source/isaaclab_tasks").exists():
        raise SystemExit(f"Not an Isaac Lab source root: {root}")
    target = find_target(root)
    changed = patch_file(target, args.dry_run)
    print("No source change required." if not changed else "Patch check complete.")


if __name__ == "__main__":
    main()
