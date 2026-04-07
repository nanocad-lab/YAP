from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN19_ROOT = REPO_ROOT / "D2W" / "input" / "design_19"


def _extract_chiplet_instances(text: str) -> list[tuple[str, str]]:
    """Return (instance_name, reference_name) in the original file order."""
    try:
        chiplet_inst_block = text.split("ChipletInst:\n", 1)[1].split("Stack:\n", 1)[0]
    except IndexError as exc:
        raise ValueError("Could not locate ChipletInst/Stack sections") from exc

    instances: list[tuple[str, str]] = []
    pattern = re.compile(
        r"^  (?P<instance>[^:\n]+):\n"
        r"    reference: (?P<reference>[^\n]+)\n"
        r"    is_master: true\n?",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(chiplet_inst_block):
        instances.append((match.group("instance"), match.group("reference")))

    if not instances:
        raise ValueError("No chiplet instances found")
    return instances


def rebuild_connection_section(text: str) -> str:
    try:
        prefix = text.split("Connection:\n", 1)[0].rstrip()
    except IndexError as exc:
        raise ValueError("Could not locate Connection section") from exc

    instances = _extract_chiplet_instances(text)
    connection_lines = ["Connection:"]
    conn_idx = 1
    for instance_name, reference_name in instances:
        if reference_name == "interposer":
            continue
        connection_lines.extend(
            [
                f"  Conn{conn_idx}:",
                f"    top: {instance_name}.regions.From_interposer",
                f"    bot: interposer_0.regions.To_{reference_name}",
            ]
        )
        conn_idx += 1

    return prefix + "\n" + "\n".join(connection_lines) + "\n"


def validate_stack_config(path: Path) -> None:
    data = yaml.safe_load(path.read_text())
    chiplet_inst = data["ChipletInst"]
    connections = data["Connection"]
    expected = sum(1 for inst in chiplet_inst.values() if inst["reference"] != "interposer")
    if len(connections) != expected:
        raise ValueError(f"{path} has {len(connections)} connections, expected {expected}")
    for conn_name, conn in connections.items():
        if "top" not in conn or "bot" not in conn:
            raise ValueError(f"{path} has incomplete {conn_name}")


def repair_all_design19_stack_configs() -> list[Path]:
    repaired_paths: list[Path] = []
    for path in sorted(DESIGN19_ROOT.rglob("generated_stack_config.3dbx")):
        repaired_text = rebuild_connection_section(path.read_text())
        path.write_text(repaired_text)
        validate_stack_config(path)
        repaired_paths.append(path)
    return repaired_paths


if __name__ == "__main__":
    repaired = repair_all_design19_stack_configs()
    print(f"Repaired {len(repaired)} generated_stack_config.3dbx files under {DESIGN19_ROOT}")
    print("Representative paths:")
    for path in repaired[:5]:
        print(f"  {path.relative_to(REPO_ROOT)}")
