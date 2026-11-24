"""Build a metadata manifest from reviewed, local Git snapshots.

Usage: python3 -I import_github.py /path/to/clones > github-manifest.json
Each clone uses its GitHub repository basename. No checkout or network is required.
Use --batch-dir to also write admin-sized batches grouped by game.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import unicodedata
from urllib.parse import unquote
import zipfile
import zlib


@dataclass(frozen=True)
class Source:
    repository: str
    commit: str
    credit: str
    license: str = "NOASSERTION"


SOURCES = (
    Source("Kenji-NX/switch-pchtxt-mods", "13f0e6559136ead857e7c66d53f1ef1775d339f4", "KeatonTheBot"),
    Source("Fl4sh9174/Switch-Emulator-Ultrawide-FPS-Mods",
           "5eed21701e8091389e2e746f073f694ea2ad4ec6", "Fl4sh9174"),
    Source("StevensND/ue4-emuswitch-60fps", "53c69bf93c2f79a6a0e6e7a46f26ea0f0010f06c", "StevensND"),
    Source("ChanseyIsTheBest/NX-60FPS-RES-GFX-Cheats",
           "a2755da3eea706734d459349370757fbe2375cfc", "ChanseyIsTheBest"),
    Source("StevensND/switch-port-mods", "774e7f7c85561ddd865c1cab555f2a335054d522", "StevensND"),
)
TITLE_ID = r"0100[0-9A-F]{12}"
HEX = r"[0-9A-F]"
TITLE = re.compile(TITLE_ID, re.I | re.ASCII)
TITLE_DIR = re.compile(rf"(?:.+?\s+)?\[({TITLE_ID})\]", re.I | re.ASCII)
TITLE_LABEL = re.compile(r"(?im)^\s*\**Title\s*ID\**\s*:\s*\**([0-9a-f]{16})\b")
GAME_LABEL = re.compile(r"(?im)^\s*\*{2}(?:Game Title|Title)\**\s*:\s*([^\n]+)")
NSOBID = re.compile(rb"@nsobid-([0-9a-f]+)", re.I)
PCHTXT_TOKEN = re.compile(rb"""\\.|["'/#]""")
PCHTXT_RECORD = re.compile(
    rb"""(?:0x)?[0-9a-f]+(?:[ \t]*(?:"(?:\\.|[^"\\])+"|'(?:\\.|[^'\\])+')"""
    rb"""|(?:[ \t]+[0-9a-f]+)*[ \t]+[0-9a-f]{2,})""", re.I,
)
SWITCH_TWO = re.compile(r"\b(?:switch[ _-]*2|nx2|nsw2|sw2)\b", re.I)
SCARLET_VIOLET = {"0100A3D008C5C000": "pokemon scarlet", "01008F6008C5E000": "pokemon violet"}
HARDWARE_ONLY = re.compile(r"\b(?:IPS|not emulat\w*|only for switch|real hardware only)\b", re.I)
VERSION = re.compile(r"v?\d+(?:\.\d+)+", re.I | re.ASCII)
CHEAT_ROW = re.compile(
    rf"^\s*\|?\s*\[(.+?)\]\((titles/[^ )]+\.txt)\)\s*\|"
    rf"\s*\[({TITLE_ID})\]\(titles/\3\)\s*\|", re.I | re.ASCII,
)
HANDLE = re.compile(r"@?([\w][\w.\-’']{0,35})", re.ASCII)
PATCH_EXTENSIONS = {".zip", ".pchtxt", ".ips", ".ips32"}
INSTALL_SCOPES = {"exefs", "cheats", "romfs", "romfs_ext", "romfslite"}
ZIP_VERSION = re.compile(r"(?i)(?<!\w)(?:v\.?\d+(?:\.\d+)+[a-z]?|\d+(?:\.\d+){2,}(?=\s*\]?$))(?!\w)")
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_ZIP_MEMBERS = 1024
MAX_PATCH_BYTES = 64 * 1024
CHEAT_FILE = re.compile(rf"({HEX}{{16}})\.txt", re.I | re.ASCII)
CHEAT_TAG = re.compile(r"\[([^\[\]{}]{1,160})\]")
CHEAT_CODES = re.compile(r"[0-9A-Fa-f]{8}(?:[ \t]+[0-9A-Fa-f]{8})*")
CHEAT_DEPENDENCY = re.compile(
    r"\b(?:master|enabler|required|requires|(?:enable|activate)\b.*\bfirst|must (?:enable|activate)"
    r"|for use with)\b|^\s*must\s+be\s+on\b",
    re.I,
)
PROHIBITION = re.compile(
    r"(?i)(?:don.t have permission|do not|don.t|no permission|not allowed|prohibit\w*)"
    r"[^\n]{0,100}\b(?:use|integrat\w*|download\w*|install\w*)\b[^\n]{0,80}"
    r"\b(?:other|external|third[ -]party)\s+(?:tools?|clients?|installers?|downloaders?|apps?)\b",
)
ZIP_PATCH_VERSION = re.compile(r"\d+(?:\.\d+){1,3}(?:-\d+(?:\.\d+){1,3})*\.pchtxt", re.I)


class ImportFailure(Exception):
    pass


class Snapshot:
    def __init__(self, root, source):
        self.path = root / source.repository.split("/")[1]
        self.source = source
        origin = self.git("config", "--local", "--get", "remote.origin.url").decode().strip()
        expected = f"https://github.com/{source.repository}"
        if origin.removesuffix(".git").lower() != expected.lower():
            raise ImportFailure(f"{self.path}: expected origin {expected}")
        if not re.fullmatch(r"[a-f0-9]{40}", source.commit):
            raise ImportFailure(f"{source.repository}: a full lowercase commit SHA is required")
        if self.git("cat-file", "-t", source.commit).strip() != b"commit":
            raise ImportFailure(f"{source.repository}: {source.commit} is not a commit")
        timestamp = int(self.git("show", "-s", "--no-show-signature", "--format=%ct", source.commit))
        self.updated = datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")
        self.files = {}
        for entry in self.git("ls-tree", "-rlz", "--full-tree", source.commit).split(b"\0"):
            if not entry:
                continue
            header, path = entry.split(b"\t", 1)
            mode, kind, oid, size = header.split()
            if mode in (b"100644", b"100755") and kind == b"blob":
                self.files[path.decode("utf-8")] = (oid.decode("ascii"), int(size))

    def git(self, *args):
        try:
            return subprocess.run(
                ["git", "--no-replace-objects", "-C", self.path, *args],
                check=True, capture_output=True, timeout=15,
                env=os.environ | {"GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:
            raise ImportFailure(f"{self.path}: git {args[0]} failed") from e

    def size(self, path):
        return self.files[path][1]

    def data(self, path, limit=MAX_METADATA_BYTES):
        oid, size = self.files[path]
        if size > limit:
            raise ImportFailure(f"{self.source.repository}: blob exceeds its inspection limit")
        return self.git("cat-file", "blob", oid)

    def text(self, path):
        return metadata_text(self.data(path))


def metadata_text(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


@dataclass(frozen=True)
class Payload:
    path: str
    size_bytes: int
    sha256: str
    kind: str
    member: str = ""
    tag: str = ""

    def record(self):
        result = {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256, "kind": self.kind}
        if self.member:
            result["member"] = self.member
        if self.tag:
            result["tag"] = self.tag
        return result


@dataclass
class Listing:
    title: str
    game: str
    name: str
    kind: str
    author: str = ""
    name_priority: int = 0
    builds: dict = field(default_factory=dict)
    payloads: dict = field(default_factory=dict)

    def add_build(self, build, target, payload):
        if build in self.builds and self.payloads[build] != payload:
            payload = None
        self.builds[build] = target
        self.payloads[build] = payload


def clean(text):
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c.isspace())
    return " ".join(unicodedata.normalize("NFC", text).split())


def variant_key(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def matches_edition(title, game):
    if SWITCH_TWO.search(game):
        return False
    if expected := SCARLET_VIOLET.get(title):
        return game.casefold().replace("é", "e") == expected
    return True


def safe_path(path):
    return (len(path.encode("utf-8")) <= 512 and not any(c in path for c in "\\:%?#")
            and not any(part in ("", ".", "..") for part in path.split("/"))
            and not any(unicodedata.category(c).startswith("C") for c in path))


def pchtxt_lines(data):
    for line in re.split(rb"[\r\n]+", data):
        quote = None
        for match in PCHTXT_TOKEN.finditer(line):
            token = match[0]
            if token in (b'"', b"'"):
                if quote is None:
                    quote = token
                elif token == quote:
                    quote = None
            elif quote is None and token in (b"/", b"#"):
                line = line[:match.start()]
                break
        line = line.strip()
        if len(line) > 2 and line.startswith(b"@"):
            yield line[1:2].lower(), line
        elif line:
            yield b"", line


def pchtxt_build(data):
    headers = []
    for directive, line in pchtxt_lines(data):
        if directive == b"s":
            break
        if directive == b"n":
            headers.append(NSOBID.fullmatch(line))
    if len(headers) != 1 or headers[0] is None:
        return None
    build = headers[0][1].decode("ascii").upper()
    if len(build) not in (16, 32, 40, 64) or not build.strip("0"):
        return None
    # IPSwitchCompiler right-pads the NSO header ID; it is not a main-build prefix.
    return build.ljust(64, "0")


def add_variant(rows, omissions, title, game, name, kind, author="", priority=0):
    if not matches_edition(title, game):
        omissions["unresolved title or edition"] += 1
        return None
    row = rows.setdefault((title, name.casefold()), Listing(title, game, name, kind, author, priority))
    if author and not row.author:
        row.author = author
    return row


def loose_pchtxt(snap, path, row, omissions, allowed=True):
    data = snap.data(path, MAX_PATCH_BYTES) if snap.size(path) <= MAX_PATCH_BYTES else b""
    if build := pchtxt_build(data):
        payload = None
        if allowed and safe_path(path) and has_patch_records(data):
            payload = Payload(path, len(data), hashlib.sha256(data).hexdigest(), "pchtxt")
        row.add_build(build, "unknown", payload)
    else:
        omissions["unusable pchtxt build evidence"] += 1


def has_patch_records(data):
    if b"\0" not in data:
        enabled = False
        for directive, line in pchtxt_lines(data):
            if directive == b"s":
                break
            if directive in (b"e", b"d"):
                enabled = directive == b"e"
            elif enabled and not directive and PCHTXT_RECORD.fullmatch(line):
                return True
    return False


def add_archive_builds(row, records, allowed=True):
    for build, target, payload in records:
        row.add_build(build, target, payload if allowed else None)


def root_allows_downloads(snap, omissions):
    try:
        restricted = any(PROHIBITION.search(snap.text(path)) for path in snap.files
                         if path.lower() == "readme.md")
    except (ImportFailure, UnicodeError):
        omissions["unreadable root README"] += 1
        return False
    if restricted:
        omissions["repository restricts third-party tools"] += 1
    return not restricted


def kind_for(name):
    if re.search(r"cheat", name, re.I):
        return "cheats"
    if re.search(r"fps|framerate", name, re.I):
        return "fps"
    if re.search(
        r"resolution|\d+p\b|\d+k\b|\d+x\d+|aspect ratio|ultrawide|\b(?:aa|dof|fxaa|ssaa|hdr|ssao|ssr|lod|fov)\b"
        r"|graphics?|bloom|shadow blur|outline|clouds|fog|filter|minimap|motion blur|tint|flare|lens|heat wave",
        name, re.I,
    ):
        return "graphics"
    return "gameplay"


def display_variant(text):
    text = clean(text)
    text = re.sub(r"(?i)(\d+)\s*fps\b", r"\1 FPS", text)
    if text.casefold() in {"uncap fps", "fps unlocked"}:
        return "Uncap FPS"
    return text


def cheat_build(filename):
    match = CHEAT_FILE.fullmatch(filename)
    if not match or not match[1].strip("0") or match[1] not in (match[1].lower(), match[1].upper()):
        return None
    return match[1].upper()


def cheat_sections(data):
    sections = {}
    current = ""
    blocked = data.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"))
    try:
        text = metadata_text(data)
    except UnicodeError:
        return {}
    for line in re.split(r"[\r\n]", text):
        line = line.strip(" \t\v\f")
        if not line:
            continue
        if match := CHEAT_TAG.fullmatch(line):
            current = match[1]
            blocked |= (current in sections or not current.strip() or "\ufffd" in current
                        or len(current.encode("utf-16-le")) > 320
                        or any(unicodedata.category(c).startswith("C") for c in current)
                        or CHEAT_DEPENDENCY.search(current) is not None)
            sections[current] = 0
        elif not current or not CHEAT_CODES.fullmatch(line):
            blocked = True
        else:
            sections[current] += len(line.split())
            blocked |= sections[current] > 0x100
    return {name: count > 0 and not blocked for name, count in sections.items()}


def zip_variants(snap, path, omissions, outer_name=None, *, bracketed=False):
    if snap.size(path) > MAX_ZIP_BYTES:
        omissions["ZIPs above inspection limit"] += 1
        return {}
    data = snap.data(path, MAX_ZIP_BYTES)
    archive_end = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if archive_end < 0 or len(data) - archive_end < 22:
        omissions["unreadable ZIPs"] += 1
        return {}
    disk, directory_disk, disk_count, count, directory_size, _, comment = struct.unpack_from(
        "<4H2IH", data, archive_end + 4,
    )
    if disk or directory_disk or disk_count != count or archive_end + 22 + comment != len(data):
        omissions["unreadable ZIPs"] += 1
        return {}
    if count > MAX_ZIP_MEMBERS or directory_size > MAX_ZIP_MEMBERS * 2048:
        omissions["ZIPs above inspection limit"] += 1
        return {}
    variants = {}
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) != count:
                omissions["unreadable ZIPs"] += 1
                return {}
            duplicates = Counter(info.filename for info in members)
            clean_archive = not comment and safe_path(path)
            buckets = {}
            for info in members:
                if info.is_dir():
                    continue
                path_parts = info.filename.split("/")
                if (info.orig_filename != info.filename
                        or info.flag_bits & 1 or duplicates[info.filename] != 1
                        or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)):
                    clean_archive = False
                    omissions["unsafe or duplicate ZIP members"] += 1
                    continue
                if not safe_path(info.filename):
                    clean_archive = False
                    omissions["unsafe or duplicate ZIP members"] += 1
                    if "\\" in info.filename or any(part in ("", ".", "..") for part in path_parts):
                        continue
                filename = path_parts[-1]
                folders = [part.lower() for part in path_parts[:-1]]
                component = not INSTALL_SCOPES.isdisjoint(folders)
                if (re.match(r"(?i)^(?:readme|licen[cs]e|copying|(?:mods?\s+)?instructions?)\b", filename)
                        and info.file_size <= MAX_PATCH_BYTES):
                    try:
                        with archive.open(info) as note:
                            clean_archive &= not PROHIBITION.search(metadata_text(note.read(MAX_PATCH_BYTES + 1)))
                    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError, UnicodeError, zlib.error):
                        clean_archive = False
                if not component:
                    continue
                if len(folders) == 1 and outer_name:
                    name, label, scope = outer_name, "", folders[0]
                elif (len(folders) >= 2 and folders[0] not in INSTALL_SCOPES
                      and folders[1] in INSTALL_SCOPES and (not bracketed or path_parts[0].startswith("["))):
                    label = path_parts[0]
                    if not clean(label) or SWITCH_TWO.search(label) or HARDWARE_ONLY.search(label):
                        continue
                    name = display_variant(ZIP_VERSION.sub("", label).replace("][", " ")
                                           .replace("[", "").replace("]", ""))
                    scope = folders[1]
                else:
                    clean_archive = False
                    continue
                if not name:
                    clean_archive = False
                    continue
                buckets.setdefault((name, label), []).append((info, scope, filename, len(folders) <= 2))
            labeled = any(label for _, label in buckets)
            clean_archive &= not labeled or all(label for _, label in buckets)
            sha256 = hashlib.sha256(data).hexdigest() if clean_archive and buckets else ""
            for (name, label), members in buckets.items():
                records = []
                standalone = clean_archive and (len(members) == 1 or all(
                    scope == "exefs" and leaf and ZIP_PATCH_VERSION.fullmatch(filename)
                    for _, scope, filename, leaf in members
                ))
                for info, scope, filename, leaf in members:
                    if not leaf or scope not in ("exefs", "cheats"):
                        continue
                    build = cheat_build(filename) if scope == "cheats" else None
                    is_patch = scope == "exefs" and filename.lower().endswith(".pchtxt")
                    if not build and not is_patch:
                        continue
                    contents = b""
                    if info.file_size <= MAX_PATCH_BYTES and info.compress_type in (
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                    ) and (is_patch or standalone):
                        try:
                            with archive.open(info) as item:
                                contents = item.read(MAX_PATCH_BYTES + 1)
                        except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error):
                            pass
                    if is_patch:
                        build = pchtxt_build(contents)
                        if not build:
                            omissions["unusable ZIP pchtxt build evidence"] += 1
                            continue
                        payload = Payload(path, len(data), sha256, "pchtxt", info.filename) if (
                            standalone and has_patch_records(contents)
                        ) else None
                        records.append((build, "unknown", payload))
                    else:
                        sections = cheat_sections(contents)
                        tags = [tag for tag, valid in sections.items() if valid]
                        tag = tags[0] if len(sections) == len(tags) == 1 else ""
                        if not tag and len(tags) > 1:
                            selected = [tag for tag in tags if display_variant(tag).casefold() == name.casefold()]
                            tag = selected[0] if len(selected) == 1 else ""
                        payload = Payload(path, len(data), sha256, "cheat", info.filename, tag) if tag else None
                        records.append((build, "main", payload))
                if records or ((label or not labeled) and any(
                    scope == "romfs" or filename in ("main.npdm", "subsdk3", "subsdk9")
                    for _, scope, filename, _ in members
                )):
                    variants.setdefault(name, []).extend(records)
    except (OSError, RuntimeError, zipfile.BadZipFile, ValueError, zlib.error):
        omissions["unreadable ZIPs"] += 1
        return {}
    return variants


def kenji(snap, omissions):
    rows = {}
    allowed = root_allows_downloads(snap, omissions)
    components = defaultdict(list)
    for path in snap.files:
        parts = path.split("/")
        for i in range(3, len(parts) - 1):
            if parts[i].lower() in INSTALL_SCOPES:
                components[tuple(parts[:i])].append(parts[i:])
                break
    for path in snap.files:
        parts = path.split("/")
        ext = Path(parts[-1]).suffix.lower()
        if ext not in PATCH_EXTENSIONS:
            continue
        if len(parts) < 3 or not TITLE.fullmatch(parts[1]) or SWITCH_TWO.search(path):
            omissions["unresolved title or edition"] += 1
            continue
        game = re.sub(r"^Legend of Zelda, The - ", "The Legend of Zelda: ", clean(parts[0]))
        if ext == ".pchtxt" and len(parts) >= 5 and parts[-2].lower() == "exefs":
            name = display_variant(" ".join(parts[2:-2]))
        elif ext == ".zip" and len(parts) == 3:
            name = display_variant(Path(parts[-1]).stem)
        else:
            omissions["unresolved variant"] += 1
            continue
        row = add_variant(rows, omissions, parts[1].upper(), game, name, kind_for(name))
        if not row:
            continue
        if ext == ".pchtxt":
            siblings = components.get(tuple(parts[:-2]), ())
            standalone = bool(siblings) and (len(siblings) == 1 or all(
                len(item) == 2 and item[0].lower() == "exefs" and ZIP_PATCH_VERSION.fullmatch(item[1])
                for item in siblings
            ))
            loose_pchtxt(snap, path, row, omissions, allowed and standalone)
        else:
            variants = zip_variants(snap, path, omissions, name)
            if len(variants) == 1:
                variant, builds = next(iter(variants.items()))
                add_archive_builds(row, builds, allowed and variant.casefold() == name.casefold())
            else:
                omissions["ZIPs without resolved variants"] += 1
    return rows.values()


def fl4sh(snap, omissions):
    rows = {}
    allowed = root_allows_downloads(snap, omissions)
    for path in snap.files:
        if not path.lower().endswith(".zip"):
            continue
        match = re.fullmatch(rf"([^/]+?)\s*\[({TITLE_ID})\](?:\s*\[(?:mods|[A-Z]{{2,3}})\])*\.zip",
                             path, re.I | re.ASCII)
        if not match or SWITCH_TWO.search(path):
            omissions["unresolved title or edition"] += 1
            continue
        title = match[2].upper()
        if title.endswith("800") or title == "01005EA01C0FC000":
            omissions["update ID or bundled title ambiguity"] += 1
            continue
        game = clean(re.sub(r"\s*\[(?:EU|EUR|JP|JPN|US|USA)\]$", "", match[1], flags=re.I))
        if not matches_edition(title, game):
            omissions["unresolved title or edition"] += 1
            continue
        variants = zip_variants(snap, path, omissions, bracketed=True)
        if not variants:
            omissions["ZIPs without resolved variants"] += 1
            variants = {"Emulator mod pack": []}
        for name, builds in variants.items():
            kind = "graphics" if name == "Emulator mod pack" else kind_for(name)
            row = add_variant(rows, omissions, title, game, name, kind)
            add_archive_builds(row, builds, allowed)
    return rows.values()


def handles(text):
    people = []
    for token in re.split(r",\s*|\s+and\s+", text.strip(), flags=re.I):
        match = HANDLE.fullmatch(token)
        if not match or match[1].lower() in {"unknown", "anonymous", "unknown-author"}:
            return []
        people.append(match[1])
    return people


def cheat_tag(tag):
    name = display_variant(tag)
    if (SWITCH_TWO.search(name) or re.search(r"\b(?:master|enabler|default|restore|re[ -]?enable)\b", name, re.I)
            or re.search(r"[/#]?[0-9A-F]{8,}|\bhttps?\b", name, re.I)):
        return None
    if re.search(r"resolution|\b(?:rrs|drs)\b|\d+p\b|\d+x\d+", name, re.I):
        return name, "graphics"
    if re.search(r"\b(?:\d+\s*fps|fps|framerate)\b", name, re.I):
        return name, "fps"
    if re.search(r"graphics|\bgfx\b|render|shadow|bloom|distortion|vignette|film grain|chrom ab|anisotrop"
                 r"|adaptive exposure|lens flare|motion blur|fog|lighting|distance scale|sharpen"
                 r"|\b(?:aa|fxaa|smaa|taa|msaa|ssa[o0]|dof|hdr|ssr|lod|fov)\b", name, re.I):
        return name, "graphics"
    return None


def chansey(snap, omissions):
    rows = {}
    allowed = root_allows_downloads(snap, omissions)
    cheats = {}
    for path in snap.files:
        match = re.fullmatch(rf"titles/({TITLE_ID})/cheats/([^/]+)", path, re.I | re.ASCII)
        if match and (build := cheat_build(match[2])):
            cheats.setdefault(match[1].upper(), []).append((build, path))
        elif path.startswith("titles/") and "/cheats/" in path:
            omissions["unusable cheat filename"] += 1

    seen = {}
    for line in snap.text("GAMES.md").splitlines():
        match = CHEAT_ROW.match(line)
        if not match:
            continue
        game, path, raw_title = match.groups()
        title = raw_title.upper()
        path = unquote(path)
        labels = path.split("/")
        if (len(labels) != 3 or labels[1].upper() != title or path not in snap.files
                or game != Path(labels[2]).stem or not matches_edition(title, clean(game)) or not cheats.get(title)):
            omissions["unresolved title or edition"] += 1
            continue
        if title in seen:
            if seen[title] != clean(game):
                raise ImportFailure(f"{snap.source.repository}: conflicting game names for {title}")
            continue
        seen[title] = clean(game)
        try:
            note = snap.text(path).lstrip().split("\n", 1)[0].strip()
        except UnicodeError:
            note = ""
        credit = re.fullmatch(r"By\s+(.+)", note, re.I)
        authors = handles(credit[1]) if credit else []
        for build, cheat_path in cheats[title]:
            if snap.size(cheat_path) > MAX_PATCH_BYTES:
                omissions["cheat files above inspection limit"] += 1
                continue
            data = snap.data(cheat_path, MAX_PATCH_BYTES)
            sections = cheat_sections(data)
            tags = [(classified, tag, usable) for tag, usable in sections.items()
                    if (classified := cheat_tag(tag))]
            if not tags:
                omissions["cheat files without supported tags"] += 1
            sha256 = hashlib.sha256(data).hexdigest() if allowed and tags and safe_path(cheat_path) else ""
            for (name, kind), tag, usable in tags:
                key = (title, name.casefold())
                row = rows.setdefault(key, Listing(title, clean(game), name, kind, ", ".join(authors), 1))
                # Atmosphère documents cheats/<build_id>.txt as the first eight bytes of main.
                payload = Payload(cheat_path, len(data), sha256, "cheat", tag=tag) if sha256 and usable else None
                row.add_build(build, "main", payload)
    omissions["cheat directories absent from game index"] += len(cheats.keys() - seen.keys())
    return rows.values()


def readme_credits(doc, name):
    lines = doc.splitlines()
    if match := re.search(r"(?im)^\s*-\s*\*\*Made by\s+\[([^\]]+)\]\(https?://[^)]+\)\*\*\s*$", doc):
        return ", ".join(handles(match[1]))
    for i, line in enumerate(lines):
        match = re.fullmatch(r"\s*(?:##\s+|\*\*)?Credits\**\s*:?\s*(.*?)\s*", line, re.I)
        if not match:
            continue
        credit_lines = [match[1]] if match[1] else []
        if not credit_lines:
            for following in lines[i + 1:]:
                following = following.strip()
                if not following:
                    if credit_lines:
                        break
                    continue
                if following.startswith("#"):
                    break
                credit_lines.append(following.removeprefix("- "))
                if not following.startswith("- "):
                    break
        credits = []
        for item in credit_lines:
            item = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", item).strip()
            for person in item.split(","):
                specific = re.fullmatch(r"\s*([^()]+?)\s*\((.+)\)\s*", person)
                if specific:
                    role = specific[2].removesuffix(" Mod")
                    if "mod" in specific[2].lower() and "author" not in role.lower():
                        if display_variant(role).casefold() != name.casefold():
                            continue
                    person = specific[1]
                credits.extend(handles(person))
        return ", ".join(dict.fromkeys(credits))
    return ""


def stevens_variant(tail):
    folders = [p for p in tail[:-1] if p.lower() not in
               {"fps", "zip folder", "exefs", "resolution", "resolutions", "resolution mods"}
               and not VERSION.fullmatch(p) and not re.match(r"\d+(?:\.\d+)+ - FLASH STILL NEED TO UPDATE", p)]
    if folders:
        if len(folders) > 1 and re.fullmatch(r"\[.+\s+v\d+(?:\.\d+)+\]", folders[-1], re.I):
            folders.pop()
        if len(folders) == 1 and folders[0].lower() == "mods":
            return None
        name = display_variant(" ".join(folders).replace(" PCHTXT", ""))
        return name
    if any(p.lower() in {"resolution", "resolutions", "resolution mods"} for p in tail[:-1]):
        return "Resolution mods"
    filename = tail[-1].lower()
    if re.search(r"uncap[ _-]*fps", filename):
        return "Uncap FPS"
    if re.search(r"(?<!\d)60[ _-]*fps\b", filename):
        return "60 FPS"
    if re.search(r"disable[_ -]dr\.", filename):
        return "Disable Dynamic Resolution"
    if re.fullmatch(r"\[disable dof-blur v\d+(?:\.\d+)+\]\.zip", filename):
        return "Disable DOF-BLUR"
    if filename == "resolution mods.zip":
        return "Resolution mods"
    return None


def stevens(snap, omissions):
    rows = {}
    root_allowed = root_allows_downloads(snap, omissions)
    readmes = {
        tuple(path.split("/")[:-1]): snap.text(path)
        for path in snap.files if "/" in path and path.split("/")[-1].lower() == "readme.md"
    }
    for path in snap.files:
        parts = path.split("/")
        ext = Path(parts[-1]).suffix.lower()
        if ext not in PATCH_EXTENSIONS:
            continue
        match = TITLE_DIR.fullmatch(parts[1]) if len(parts) >= 3 else None
        game, name = clean(parts[0]), None
        if not match:
            if parts[0] != "Pokemon Brilliant Diamond Shining Pearl" or len(parts) != 3:
                omissions["unresolved title or edition"] += 1
                continue
            edition = re.fullmatch(r"(brilliantdiamond|shiningpearl)(60fps|uncap)\.zip", parts[-1])
            if not edition:
                omissions["unresolved title or edition"] += 1
                continue
            game = {"brilliantdiamond": "Pokémon Brilliant Diamond",
                    "shiningpearl": "Pokémon Shining Pearl"}[edition[1]]
            documented = re.search(rf"\*\*{game} TID\*\*:\s*({TITLE_ID})", readmes.get((parts[0],), ""))
            if not documented:
                omissions["unresolved title or edition"] += 1
                continue
            title = documented[1]
            name = "60 FPS" if edition[2] == "60fps" else "Uncap FPS"
        else:
            title = match[1].upper()
        if SWITCH_TWO.search(path):
            omissions["unresolved title or edition"] += 1
            continue
        if ext in {".ips", ".ips32"} or any(HARDWARE_ONLY.search(part) for part in parts[2:]):
            omissions["hardware-only or unverified IPS layout"] += 1
            continue
        if title == "0100F2C0115B6000":
            omissions["mixed third-party collection with obsolete packs"] += 1
            continue
        documents = [readmes[prefix] for n in range(len(parts) - 1, 0, -1)
                     if (prefix := tuple(parts[:n])) in readmes]
        if any(ids and title not in {id_.upper() for id_ in ids}
               for doc in documents if (ids := TITLE_LABEL.findall(doc))):
            omissions["conflicting README title ID"] += 1
            continue
        if title == "010009D018A06000":
            omissions["conflicting region guidance"] += 1
            continue
        if name is None:
            name = stevens_variant(parts[2:])
        variants = zip_variants(snap, path, omissions, name) if ext == ".zip" else {}
        if not name and not variants:
            omissions["unresolved variant"] += 1
            continue
        documented_name = next((m[1].strip("* ") for doc in documents if (m := GAME_LABEL.search(doc))), "")
        if documented_name:
            game = clean(documented_name)
        if parts[0] == "Sonic x Shadow Generations":
            game += " (Shadow)" if title == "01005EA01C0FC000" else " (Sonic)"
        allowed = root_allowed and not any(PROHIBITION.search(doc) for doc in documents)
        if not allowed:
            omissions["assets restricted by third-party tool terms"] += 1
        outer_name = "Dynamic FPS BETA" if name == "Mods Dynamic FPS" else name
        if not variants:
            variants = {outer_name: []}
            if ext == ".zip":
                omissions["ZIPs without resolved variants"] += 1
        elif len(variants) == 1 and name:
            variant, builds = next(iter(variants.items()))
            allowed &= variant.casefold() in {name.casefold(), outer_name.casefold()}
            variants = {outer_name: builds}
        for variant, builds in variants.items():
            author = next((credit for doc in documents if (credit := readme_credits(doc, variant))), "")
            if title == "01003AE01AA76000":
                author = "Masagrator, StevensND"
            row = add_variant(rows, omissions, title, game, variant, kind_for(variant), author,
                              2 if documented_name else 0)
            if row:
                if ext == ".pchtxt":
                    loose_pchtxt(snap, path, row, omissions, allowed)
                else:
                    add_archive_builds(row, builds, allowed)
    return rows.values()


def build_manifest(root, sources=SOURCES, parsers=(kenji, fl4sh, stevens, chansey, stevens)):
    games = {}
    stats = {}
    for source, parse in zip(sources, parsers, strict=True):
        snap = Snapshot(root, source)
        omitted = Counter()
        rows = list(parse(snap, omitted))
        if not rows:
            raise ImportFailure(f"{source.repository}: no supported listings at {source.commit}")
        for row in rows:
            key = variant_key(row.name)
            identity = f"{source.repository.lower()}\0{row.title}\0{row.name.casefold()}".encode()
            digest = hashlib.sha256(identity).hexdigest()[:16]
            source_slug = variant_key(source.repository.split("/")[0])
            slug = f"{source_slug}-{key}"[:42].rstrip("-")
            mod_id = f"gh-{slug}-{digest}"
            if parse is chansey:
                summary = f"{row.name} cheat for {row.game}."
                description = "The source targets Atmosphère on hardware. Emulator support is not established."
            elif parse is fl4sh:
                summary = f"{row.name} for {row.game}."
                description = "This mod is distributed in a game pack. Check the source for required options."
            else:
                summary = f"{row.name} for {row.game}."
                description = "Check the source for supported versions, credits, and notes."
            mods, game, priority = games.setdefault(row.title, ({}, row.game, row.name_priority))
            if row.name_priority > priority:
                games[row.title] = (mods, row.game, row.name_priority)
            if mod_id in mods:
                raise ImportFailure(f"duplicate listing identity: {mod_id}")
            mods[mod_id] = {
                "id": mod_id, "name": row.name, "summary": summary, "description": description,
                "author": row.author or (source.credit if parse is kenji
                                         else f"{source.credit} (repository maintainer)"),
                "type": row.kind, "repository": source.repository, "commit": source.commit,
                "license": source.license, "source_url": f"https://github.com/{source.repository}/tree/{source.commit}",
                "updated_at": snap.updated,
                "builds": [
                    {"id": build, "target": row.builds[build]}
                    | ({"download": row.payloads[build].record()} if row.payloads[build] else {})
                    for build in sorted(row.builds)
                ],
            }
        stats[source.repository] = {
            "commit": source.commit, "license": source.license,
            "games": len({row.title for row in rows}), "listings": len(rows),
            "builds": sum(len(row.builds) for row in rows),
            "downloadable_builds": sum(bool(payload) for row in rows for payload in row.payloads.values()),
            "without_builds": sum(not row.builds for row in rows),
            "omissions": {reason: count for reason, count in sorted(omitted.items()) if count},
        }
    return {"games": [{"title_id": title, "name": game, "mods": [mods[id_] for id_ in sorted(mods)]}
                      for title, (mods, game, _) in sorted(games.items())]}, stats


def write_batches(manifest, directory, limit=8 << 20):
    header, footer = b'{"games":[', b']}\n'
    batches = []
    pending = []
    size = len(header) + len(footer)
    for game in manifest["games"]:
        encoded = json.dumps(game, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) + len(header) + len(footer) > limit:
            raise ImportFailure(f"{game['title_id']}: one game exceeds the admin request limit")
        if pending and size + len(encoded) + 1 > limit:
            batches.append(header + b",".join(pending) + footer)
            pending = []
            size = len(header) + len(footer)
        size += len(encoded) + bool(pending)
        pending.append(encoded)
    if pending:
        batches.append(header + b",".join(pending) + footer)
    directory.mkdir(mode=0o700)
    for i, batch in enumerate(batches, 1):
        with (directory / f"github-{i:03}.json").open("xb") as out:
            out.write(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clones", type=Path, help="directory containing the five local Git clones")
    parser.add_argument("--batch-dir", type=Path, help="new directory for admin upload batches")
    args = parser.parse_args()
    try:
        manifest, stats = build_manifest(args.clones)
        if args.batch_dir is not None:
            write_batches(manifest, args.batch_dir)
    except (ImportFailure, UnicodeError, ValueError, OSError) as error:
        parser.exit(1, f"import failed: {error}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)
    sys.stdout.buffer.write((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
