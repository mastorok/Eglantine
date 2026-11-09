#!/usr/bin/env python3
"""Reads the guest profile of the sampler (EDEN_NX2_SAMPLE with CPUCore_* threads).

  guest_hot.py <profil.txt> [--journal journal.txt] [--objdump llvm-objdump] [--core CPUCore_4]
               [--blocks 8]

The sampler writes <profil.txt>.guest.txt (where each guest core was: threads, 256-byte blocks,
addresses) and <profil.txt>.guest_code.txt (the code around the hottest blocks). This prints, per
core, the hottest blocks as "module+offset" (module bases from the journal's "Module <name> loaded
at 0x..." lines) followed by the disassembly of each block with the sample count of every address
that was hit.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def modules(journal):
    found = []
    if journal:
        for line in Path(journal).read_text(errors="replace").splitlines():
            m = re.search(r"Module (\S+) loaded at 0x([0-9a-fA-F]+) \(size 0x([0-9a-fA-F]+)\)", line)
            if m:
                found.append((m.group(1), int(m.group(2), 16), int(m.group(3), 16)))
    return found


def where(address, mods):
    for name, base, size in mods:
        if base <= address < base + size:
            return f"{name}+{address - base:#x}"
    return f"{address:#x}"


def parse_guest(path):
    cores, current = {}, None
    section = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("=== "):
            current = line[4:].split(":")[0]
            cores[current] = {"head": line[4:], "threads": [], "blocks": [], "pcs": []}
            continue
        if line.startswith("-- "):
            section = line[3:].split(" ")[0]
            continue
        if current is None or not line.strip():
            continue
        parts = line.split()
        if section == "guest" and parts[0].isdigit():
            cores[current]["threads"].append((int(parts[0]), line.strip()))
        elif section == "256-byte":
            cores[current]["blocks"].append((int(parts[0], 16), int(parts[1]), parts[2]))
        elif section == "guest":
            pass
        elif section == "guest" or section == "guest":
            pass
        if section == "guest" and parts[0].startswith("0x"):
            cores[current]["pcs"].append((int(parts[0], 16), int(parts[1])))
    return cores


def parse_code(path):
    code, current, start = {}, None, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("=== "):
            current = line[4:].strip()
            code[current] = {}
        elif line.startswith("# block"):
            pass
        elif current and re.match(r"0x[0-9a-f]+: [0-9a-f]+$", line):
            addr, data = line.split(": ")
            code[current][int(addr, 16)] = bytes.fromhex(data)
    return code


def disassemble(objdump, base, data):
    """Assembles the words as .inst directives (so every word keeps its place) and disassembles."""
    clang = "clang"
    with tempfile.TemporaryDirectory() as tmp:
        asm = Path(tmp) / "code.s"
        obj = Path(tmp) / "code.o"
        words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data) - 3, 4)]
        asm.write_text(".text\n" + "".join(f".inst {w:#010x}\n" for w in words))
        built = subprocess.run([clang, "--target=aarch64-linux-gnu", "-c", "-x", "assembler",
                                str(asm), "-o", str(obj)], capture_output=True, text=True)
        if built.returncode != 0:
            return built.stderr
        out = subprocess.run([objdump, "-d", f"--adjust-vma={base:#x}", str(obj)],
                             capture_output=True, text=True)
        return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--journal")
    ap.add_argument("--objdump", default="llvm-objdump")
    ap.add_argument("--clang", default="clang")
    ap.add_argument("--core")
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()
    mods = modules(args.journal)
    guest = parse_guest(args.profile + ".guest.txt")
    code = parse_code(args.profile + ".guest_code.txt")
    for core, info in guest.items():
        if args.core and core != args.core:
            continue
        print("=" * 100)
        print(info["head"])
        for _, text in info["threads"][:4]:
            print("  thread:", text)
        hits = dict(info["pcs"])
        for block, count, pct in info["blocks"][: args.blocks]:
            print("-" * 100)
            print(f"block {where(block, mods)}  {count} samples ({pct})")
            chunks = code.get(core, {})
            start = block - 0x80
            data = b"".join(chunks.get(start + o, b"") for o in range(0, 512, 32))
            if len(data) != 512:
                print("  (code not captured)")
                continue
            for line in disassemble(args.objdump, start, data).splitlines():
                m = re.match(r"\s*([0-9a-f]+):\s", line)
                mark = ""
                if m:
                    n = hits.get(int(m.group(1), 16))
                    if n:
                        mark = f"   <== {n} samples"
                if m or line.strip().startswith("0000"):
                    print(line + mark)


if __name__ == "__main__":
    sys.exit(main())
