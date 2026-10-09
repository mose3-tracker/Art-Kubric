#!/usr/bin/env python
"""Strip <inertial> blocks from URDFs in place.

Infinigen-Articulated exports inertia tensors that SAPIEN rejects; without them SAPIEN
computes inertia from the collision geometry.

    python tools/sanitize_urdf_inertials.py <dir> [...]
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize(urdf_path: Path) -> bool:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    removed = 0
    for link in root.iter("link"):
        for inertial in link.findall("inertial"):
            link.remove(inertial)
            removed += 1
    if removed:
        tree.write(urdf_path, xml_declaration=True, encoding="unicode")
    return bool(removed)


def main() -> None:
    changed = total = 0
    for d in sys.argv[1:]:
        for urdf in Path(d).rglob("*.urdf"):
            total += 1
            changed += sanitize(urdf)
    print(f"Sanitized {changed}/{total} URDFs")


if __name__ == "__main__":
    main()
