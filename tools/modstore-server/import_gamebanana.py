import argparse
import hashlib
import http.client
import json
import os
import re
import stat
import struct
import sys
import tempfile
import time
import unicodedata
import zipfile
import zlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ORIGIN = "https://gamebanana.com"
API = ORIGIN + "/apiv11"
USER_AGENT = "Lichii Mod Store catalog importer (+https://github.com/zortos293/Lichii)"
SORTS = ("Generic_NewAndUpdated", "Generic_MostLiked")
MIB = 1024 * 1024
METADATA_TTL = 24 * 60 * 60
UNAVAILABLE_RETRY = 24 * 60 * 60
ROW_FLAGS = (
    "_bIsPrivate", "_bIsTrashed", "_bIsWithheld", "_bIsFlagged", "_bIsObsolete",
    "_bIsArchived", "_bHasContentRatings", "_aContentRatings", "_bIsNsfw", "_bIsAdult",
)
BUILD_ID = r"(?:[0-9A-F]{16}|[0-9A-F]{64})"
IPS_NAME = re.compile(rf"({BUILD_ID})\.ips(?:32)?", re.IGNORECASE | re.ASCII)
CHEAT_NAME = re.compile(r"([0-9A-F]{16})\.txt", re.IGNORECASE | re.ASCII)
NSOBID = re.compile(rb"^[ \t]*@nsobid-([0-9A-F]{16}|[0-9A-F]{64})[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)
PATCH_ROOTS = {"exefs", "exefs_patches"}
PATCH_TEXT_MAX = 64 * 1024
INSPECTABLE = {".zip", ".rar", ".7z", ".ips", ".ips32", ".pchtxt"}
CDN_HOST = re.compile(r"(?:files2?|archive|filecache[0-9]{1,3})\.gamebanana\.com")
REDIRECT = (301, 302, 303, 307, 308)
KIRBY_ORIGINAL = "01004D300C5AE000"
KIRBY_EDITION_SPLIT = "2025-08-27T00:00:00Z"
KIRBY_SWITCH_1 = re.compile(rf"(?<![A-Z0-9])(?:switch[ _-]*1|{KIRBY_ORIGINAL})(?![A-Z0-9])", re.IGNORECASE)
GAMES = {
    "040074A01BF12000": ("Donkey Kong Bananza", 23092),
    "0400E88025B1A000": ("Star Fox", 25232),
    "04007EF01F8AE000": ("Fire Emblem: Fortune's Weave", 26274),
    "0400C3F00006E000": ("Mario Kart World", 22152),
    "04008E001F098000": ("Nintendo Switch Sports Resort", 26399),
    "0400A3A01EE4C000": ("Pokémon Pokopia", 26262),
    "0400380020056000": ("Raiders", 26183),
    "0400486024DEC000": ("The Legend of Zelda: Ocarina of Time", 26242),
    "0400245021D2C000": ("Yoshi and the Mysterious Book", 26326),
    "0100EA80032EA000": ("New Super Mario Bros. U Deluxe", 6886),
    "0100152000022000": ("Mario Kart 8 Deluxe", 6507),
    "01007EF00011E000": ("The Legend of Zelda: Breath of the Wild", 6386),
    "0100F2C0115B6000": ("The Legend of Zelda: Tears of the Kingdom", 7617),
    "01006D0017F7A000": ("Mario & Luigi: Brothership", 20508),
    "010051F0207B2000": ("Tomodachi Life: Living the Dream", 23911),
    "01006BB00C6F0000": ("The Legend of Zelda: Link's Awakening", 7562),
    "01001F5010DFA000": ("Pokémon Legends: Arceus", 15053),
    "0100F43008C44000": ("Pokémon Legends: Z-A", 23582),
    KIRBY_ORIGINAL: ("Kirby and the Forgotten Land", 15908),
}
SWITCH_2 = re.compile(r"(?<![A-Z0-9])(?:switch[ _-]*2|nsw?2|sw2|nx2)(?![A-Z0-9])", re.IGNORECASE)
WII_U = re.compile(r"(?<![A-Z0-9])(?:wii[ _-]*u|cemu)(?![A-Z0-9])", re.IGNORECASE)
SWITCH_2_ONLY = re.compile(r"\b(?:switch[ _-]*2[ _-]+(?:edition[ _-]+)?only|(?:only|requires)[ _-]+(?:on[ _-]+)?(?:nintendo[ _-]+)?switch[ _-]*2)\b", re.IGNORECASE)
TYPE_KEYWORDS = (
    ("fps", r"\b(?:fps|frame ?rates?|performance|[0-9]+ ?fps)\b"),
    ("translation", r"\b(?:translations?|locali[sz]ations?|language packs?)\b"),
    ("cheats", r"\b(?:cheats?|trainers?)\b"),
    ("graphics", r"\b(?:graphics?|visuals?|shaders?|textures?|skins?|models?|reshade|ui|hud)\b"),
)


class ImportFailure(Exception):
    pass


class Blocked(ImportFailure):
    pass


class Cooldown(Blocked):
    pass


class Unavailable(Exception):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def write_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as out:
        pending = Path(out.name)
        try:
            out.write(data)
            out.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


class Upstream:
    def __init__(self, root, offline=False):
        self.root = root
        self.offline = offline
        self.opener = build_opener(NoRedirects())
        self.next_request = 0.0
        self.cooldown = root / "retry-after.json"
        self.missing_feeds = 0
        self.missing_profiles = 0

    def stop(self, reason, retry_after):
        now = time.time()
        delay = None
        if retry_after:
            try:
                delay = int(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    delay = when.replace(tzinfo=when.tzinfo or timezone.utc).timestamp() - now
                except (ValueError, OverflowError, OSError):
                    pass
        until = now + (3600 if delay is None else min(253402300799 - now, max(1, delay)))
        write_cache(self.cooldown, json.dumps({"until": until, "reason": reason}).encode())
        raise Blocked(f"{reason}; stopped (Retry-After: {retry_after or 'not supplied'})")

    def request(self, url, accept):
        if self.offline:
            raise ImportFailure("offline cache miss")
        if self.cooldown.exists():
            state = json.loads(self.cooldown.read_bytes())
            remaining = state["until"] - time.time()
            if remaining > 0:
                raise Cooldown(f"upstream cooldown active for {int(remaining) + 1}s: {state['reason']}")
        while (delay := self.next_request - time.monotonic()) > 0:
            time.sleep(delay)
        self.next_request = time.monotonic() + 1
        request = Request(url, headers={
            "User-Agent": USER_AGENT, "Accept": accept, "Accept-Encoding": "identity",
        })
        try:
            response = self.opener.open(request, timeout=20)
        except HTTPError as error:
            response = error
        code = response.code
        retry_after = response.headers.get("Retry-After")
        if (code in (401, 403, 429, 451) or retry_after
                or response.headers.get("WWW-Authenticate")
                or response.headers.get("cf-mitigated", "").lower() == "challenge"):
            response.close()
            self.stop(f"GameBanana access blocked (HTTP {code})", retry_after)
        return response

    def metadata(self, endpoint, missing_ok=False):
        url = API + endpoint
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.root / "metadata" / f"{key}.json"
        if path.is_file() and (self.offline or time.time() - path.stat().st_mtime < METADATA_TTL):
            data = path.read_bytes()
            return None if missing_ok and data == b"null" else json_object(data, endpoint)
        if self.offline:
            if missing_ok:
                self.missing_profiles += 1
            else:
                self.missing_feeds += 1
            return None
        with self.request(url, "application/json") as response:
            if missing_ok and response.code in (404, 410):
                write_cache(path, b"null")
                return None
            if response.code in REDIRECT:
                self.stop(f"{endpoint}: API redirected instead of returning JSON", None)
            if response.code != 200:
                raise ImportFailure(f"{endpoint}: HTTP {response.code}")
            content_type = response.headers.get_content_type()
            data = response.read(2 * MIB + 1)
        if len(data) > 2 * MIB:
            raise ImportFailure(f"{endpoint}: JSON response exceeds 2 MiB")
        if not data.strip() or content_type != "application/json" or data.lstrip().startswith(b"<"):
            ordinary = False
            if missing_ok and content_type == "text/html":
                page = endpoint.removeprefix("/Mod/").removesuffix("/ProfilePage")
                ordinary = ordinary_profile_html(data, f"{ORIGIN}/mods/{page}")
            if missing_ok and (not data.strip() or ordinary):
                reason = "ordinary HTML profile page" if ordinary else "empty response"
                print(f"{endpoint}: {reason} instead of JSON, skipped", file=sys.stderr)
                write_cache(path, b"null")
                return None
            self.stop(f"{endpoint}: expected JSON from GameBanana, got {content_type}", None)
        value = json_object(data, endpoint)
        write_cache(path, data)
        return value

    def archive(self, url):
        if not re.fullmatch(r"https://gamebanana\.com/dl/[1-9][0-9]{0,18}", url):
            raise Unavailable("not an official GameBanana file link")
        seen = set()
        for _ in range(5):
            if url in seen:
                raise Unavailable("file redirects loop")
            seen.add(url)
            response = self.request(url, "application/octet-stream, */*")
            if response.code in REDIRECT:
                with response:
                    url = urljoin(url, response.headers.get("Location", ""))
                if re.search(r"/(?:sign-?in|log-?in|authenticate)(?:[/?#]|$)", url, re.IGNORECASE):
                    self.stop("GameBanana archive redirected to authentication", None)
                if not archive_cdn(url):
                    raise Unavailable("redirect is not to a vetted GameBanana file host")
                continue
            if response.code == 200 and response.headers.get_content_type() == "text/html":
                response.close()
                self.stop("GameBanana archive request returned HTML instead of a file", None)
            if response.code == 200 and archive_cdn(url):
                return response
            with response:
                if response.code >= 500:
                    raise ImportFailure(f"GameBanana file service HTTP {response.code}")
                raise Unavailable(f"no public CDN file (HTTP {response.code})")
        raise Unavailable("too many file redirects")


def json_object(data, context):
    try:
        value = json.loads(data)
    except (ValueError, RecursionError) as error:
        raise ImportFailure(f"{context}: invalid JSON") from error
    if not isinstance(value, dict) or "_sError" in value or "_aErrors" in value:
        raise ImportFailure(f"{context}: unexpected JSON response")
    return value


def ordinary_profile_html(data, page):
    if re.search(
        rb"challenge-platform|cf-chl-|just a moment|checking your browser|"
        rb"<title[^>]*>\s*(?:log\s*in|sign\s*in|access denied|verify you are human)\b",
        data, re.IGNORECASE,
    ):
        return False

    class PageMarkers(HTMLParser):
        canonical = None
        social_url = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "link" and attrs.get("rel") == "canonical":
                self.canonical = attrs.get("href")
            if tag == "meta" and attrs.get("property") == "og:url":
                self.social_url = attrs.get("content")

    markers = PageMarkers()
    markers.feed(data.decode("utf-8", errors="replace"))
    markers.close()
    return markers.canonical == markers.social_url == page


def archive_cdn(url):
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    path = unquote(parts.path)
    return (parts.scheme == "https" and CDN_HOST.fullmatch(parts.netloc) is not None
            and not parts.fragment and path.startswith("/mods/")
            and "\\" not in path and not any(part in (".", "..") for part in path.split("/"))
            and not any(ord(char) < 32 or ord(char) == 127 for char in url + path))


def row(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return (item for item in value if isinstance(item, dict)) if isinstance(value, list) else ()


def positive_int(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value):
        value = int(value)
    return value if type(value) is int and 0 < value < 2**63 else None


def timestamp(value):
    seconds = positive_int(value)
    if seconds is None or seconds > 253402300799:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def control(char):
    return unicodedata.category(char) in ("Cc", "Cf", "Cs")


def single_line(value, limit):
    if not isinstance(value, str):
        return ""
    cleaned = "".join(" " if char.isspace() else char for char in value if not control(char) or char.isspace())
    return " ".join(cleaned.split())[:limit].rstrip()


class PlainText(HTMLParser):
    hidden = frozenset({"script", "style", "iframe", "template", "object", "svg"})
    breaks = frozenset({"br", "div", "p", "li", "ul", "ol", "blockquote", "pre", "hr",
                        "h1", "h2", "h3", "h4", "h5", "h6", "td", "tr"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = []

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden:
            self.suppressed.append(tag)
        elif not self.suppressed and tag in self.breaks:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.suppressed:
            if tag == self.suppressed[-1]:
                self.suppressed.pop()
        elif tag in self.breaks:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def plain_text(value, limit):
    if not isinstance(value, str):
        return ""
    parser = PlainText()
    parser.feed(value)
    parser.close()
    lines = (single_line(line, limit) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)[:limit].rstrip()


def excluded(record):
    return any(record.get(key) for key in ROW_FLAGS)


def credits(profile):
    result = []
    for group in records(profile.get("_aCredits")):
        group_name = single_line(group.get("_sGroupName"), 100)
        for person in records(group.get("_aAuthors")):
            name = single_line(person.get("_sName"), 100)
            if not name:
                continue
            role = single_line(person.get("_sRole"), 100)
            if role and group_name and role.casefold() != group_name.casefold():
                role = (group_name + ": " + role)[:100].rstrip()
            url = person.get("_sProfileUrl")
            if not isinstance(url, str) or not re.fullmatch(
                r"https://gamebanana\.com/members/[1-9][0-9]{0,18}", url
            ):
                url = ""
            credit = {"name": name, "role": role or group_name or "Contributor", "url": url}
            if credit not in result:
                result.append(credit)
                if len(result) == 8:
                    return result
    return result


def previews(profile):
    result = []
    for item in records(row(profile.get("_aPreviewMedia")).get("_aImages")):
        base = item.get("_sBaseUrl")
        name = item.get("_sFile")
        if (base == "https://images.gamebanana.com/img/ss/mods" and isinstance(name, str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,180}\.(?:jpe?g|png|gif|webp)", name, re.IGNORECASE)):
            url = f"{base}/{name}"
            if url not in result:
                result.append(url)
                if len(result) == 8:
                    break
    return result


def public_file(item):
    file_id = positive_int(item.get("_idRow"))
    name = item.get("_sFile")
    size = positive_int(item.get("_nFilesize"))
    uploaded = timestamp(item.get("_tsDateAdded"))
    url = item.get("_sDownloadUrl")
    if (excluded(item) or item.get("_bIsTainted") or item.get("_bIsMalicious")
            or item.get("_sAnalysisState") != "done" or item.get("_sAnalysisResult") != "ok"
            or item.get("_sAvState") != "done" or item.get("_sAvResult") != "clean"
            or not (file_id and size and uploaded) or url != f"{ORIGIN}/dl/{file_id}"
            or not isinstance(name, str) or not 0 < len(name) <= 240
            or not name.strip() or name in (".", "..") or "/" in name or "\\" in name
            or any(control(char) for char in name)):
        return None
    return {"id": file_id, "name": name, "size_bytes": size, "uploaded_at": uploaded,
            "download_url": url, "sha256": None, "build_ids": []}


def mod_entry(profile, mod_id, game_id, inspector, title_id):
    category_row = row(profile.get("_aCategory"))
    parent_category = single_line(row(profile.get("_aSuperCategory")).get("_sName"), 160)
    category = single_line(category_row.get("_sName"), 160) or parent_category
    name = single_line(profile.get("_sName"), 120)
    updated = max(filter(None, (timestamp(profile.get(key)) for key in (
        "_tsDateUpdated", "_tsDateModified", "_tsDateAdded",
    ))), default=None)
    page = f"{ORIGIN}/mods/{mod_id}"
    tagline = plain_text(profile.get("_sDescription"), 240)
    description = plain_text(profile.get("_sText"), 700)
    tags = profile.get("_aTags")
    tags = " ".join(single_line(row(tag).get("_sName") if isinstance(tag, dict) else tag, 80)
                    for tag in tags) if isinstance(tags, list) else ""
    edition_labels = f"{name} {tagline} {category} {parent_category} {tags}"
    kirby_original_listing = title_id == KIRBY_ORIGINAL and bool(
        KIRBY_SWITCH_1.search(f"{edition_labels} {description}")
    )
    wrong_editions = []
    if title_id.startswith("0100"):
        wrong_editions.append(SWITCH_2)
    if game_id == 6386:
        wrong_editions.append(WII_U)
    if (excluded(profile) or excluded(category_row)
            or str(profile.get("_sInitialVisibility")).lower() != "show"
            or positive_int(profile.get("_idRow")) != mod_id
            or positive_int(row(profile.get("_aGame")).get("_idRow")) != game_id
            or profile.get("_sProfileUrl") != page or not (name and category and updated)
            or any(pattern.search(edition_labels) for pattern in wrong_editions)
            or (SWITCH_2 in wrong_editions and SWITCH_2_ONLY.search(description))):
        return None
    credited = credits(profile)
    if not credited:
        return None
    authors = []
    for credit in credited:
        author = credit["name"]
        if author not in authors and len(", ".join([*authors, author])) <= 100:
            authors.append(author)
    files = []
    seen_files = set()
    for item in records(profile.get("_aFiles")):
        file = public_file(item)
        if not file or file["id"] in seen_files:
            continue
        file_description = plain_text(item.get("_sDescription"), 240)
        if (any(pattern.search(file["name"]) for pattern in wrong_editions)
                or (SWITCH_2 in wrong_editions and SWITCH_2_ONLY.search(file_description))
                or (title_id == KIRBY_ORIGINAL and not kirby_original_listing
                    and file["uploaded_at"] >= KIRBY_EDITION_SPLIT
                    and not KIRBY_SWITCH_1.search(f"{file['name']} {file_description}"))):
            continue
        seen_files.add(file["id"])
        if inspector:
            inspector.inspect(file, item.get("_sMd5Checksum"))
        files.append(file)
        if len(files) == 8:
            break
    if not files:
        return None
    summary = single_line(tagline or description or name, 240)
    mod_type = next((kind for kind, regex in TYPE_KEYWORDS
                     if re.search(regex, f"{category} {parent_category}", re.IGNORECASE)), "gameplay")
    if mod_type == "gameplay" and re.search(r"\b[0-9]+\s*fps\b", name, re.IGNORECASE):
        mod_type = "fps"
    return {"id": mod_id, "name": name, "summary": summary,
            "description": description or tagline or summary, "author": ", ".join(authors),
            "credits": credited, "type": mod_type, "category": category, "page_url": page,
            "preview_urls": previews(profile), "updated_at": updated, "files": files}


def nsobids(data):
    return {match.decode("ascii").upper() for match in NSOBID.findall(data.removeprefix(b"\xef\xbb\xbf"))}


def zip_patch_ids(archive):
    end = archive.seek(0, os.SEEK_END)
    start = max(0, end - (65535 + 22 + 20))
    archive.seek(start)
    footer = archive.read()
    marker = footer.rfind(b"PK\x05\x06")
    if (marker < 0 or len(footer) - marker < 22
            or (marker >= 20 and footer[marker - 20:marker - 16] == b"PK\x06\x07")):
        return []
    _, disk, central_disk, disk_count, count, size, offset, comment = struct.unpack_from(
        "<4s4H2LH", footer, marker
    )
    if (disk or central_disk or disk_count != count or count > 4096 or size > 2 * MIB
            or offset + size != start + marker or marker + 22 + comment != len(footer)):
        return []
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if any(entry.flag_bits & 1 for entry in entries):
                raise Unavailable("ZIP has encrypted entries")
            if len(entries) != count:
                return []
            ids = set()
            metadata_left = 4 * PATCH_TEXT_MAX
            for entry in entries:
                path = entry.orig_filename.replace("\\", "/").split("/")
                file_type = stat.S_IFMT(entry.external_attr >> 16)
                if (len(path) < 2 or entry.is_dir() or entry.external_attr & 0x10
                        or file_type not in (0, stat.S_IFREG)
                        or any(part in ("", ".", "..") or ":" in part for part in path)
                        or any(control(char) for char in entry.orig_filename)):
                    continue
                name = path[-1]
                if path[-2].lower() == "cheats":
                    match = CHEAT_NAME.fullmatch(name)
                    if match:
                        ids.add(match[1].upper())
                if not PATCH_ROOTS.intersection(part.lower() for part in path[:-1]):
                    continue
                match = IPS_NAME.fullmatch(name)
                if match:
                    ids.add(match[1].upper())
                if (name.lower().endswith(".pchtxt")
                        and 0 < entry.file_size <= min(PATCH_TEXT_MAX, metadata_left)
                        and entry.compress_size <= PATCH_TEXT_MAX
                        and entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    metadata_left -= entry.file_size
                    try:
                        with source.open(entry) as member:
                            data = member.read(entry.file_size)
                        if len(data) == entry.file_size:
                            ids.update(nsobids(data))
                    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error):
                        continue
            return sorted(ids)[:32]
    except (zipfile.BadZipFile, NotImplementedError, UnicodeError, ValueError):
        pass
    return []


class Inspector:
    def __init__(self, upstream, max_mib, max_files):
        self.upstream = upstream
        self.max_bytes = max_mib * MIB
        self.remaining = max_files

    def inspect(self, file, md5):
        file_id = file["id"]
        suffix = Path(file["name"]).suffix.lower()
        if suffix not in INSPECTABLE:
            print(f"file {file_id}: archive type not inspected", file=sys.stderr)
            return
        if not isinstance(md5, str) or not re.fullmatch("[0-9A-Fa-f]{32}", md5):
            print(f"file {file_id}: no upstream MD5; SHA/IDs skipped", file=sys.stderr)
            return
        md5 = md5.lower()
        cache = self.upstream.root / "inspection" / f"{file_id}-{md5}.json"
        evidence = None
        if cache.is_file():
            evidence = json_object(cache.read_bytes(), f"file {file_id} inspection")
            sha = evidence.get("sha256")
            ids = evidence.get("build_ids")
            if (sha is not None and (not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha))
                    or not isinstance(ids, list) or len(ids) > 32 or any(
                        not isinstance(build_id, str) or not re.fullmatch(BUILD_ID, build_id)
                        for build_id in ids
                    ) or sha is None and ids):
                raise ImportFailure(f"file {file_id}: invalid cached inspection")
            if sha is None:
                if evidence.get("unavailable") is True and time.time() - cache.stat().st_mtime < UNAVAILABLE_RETRY:
                    print(f"file {file_id}: recently unavailable; SHA/IDs skipped", file=sys.stderr)
                    return
                evidence = None
        if evidence is None:
            if self.upstream.offline or not self.remaining or file["size_bytes"] > self.max_bytes:
                print(f"file {file_id}: offline or over archive budget; SHA/IDs skipped", file=sys.stderr)
                return
            self.remaining -= 1
            try:
                evidence = self.download(file, md5, suffix)
            except Unavailable as error:
                write_cache(cache, b'{"sha256":null,"build_ids":[],"unavailable":true}')
                print(f"file {file_id}: {error}; SHA/IDs skipped", file=sys.stderr)
                return
            write_cache(cache, json.dumps(evidence).encode())
        file.update(sha256=evidence["sha256"], build_ids=evidence["build_ids"])
        if not file["build_ids"]:
            print(f"file {file_id}: SHA-256 available; no patch build IDs", file=sys.stderr)

    def download(self, file, md5, suffix):
        expected = file["size_bytes"]
        sha = hashlib.sha256()
        checksum = hashlib.md5(usedforsecurity=False)
        with self.upstream.archive(file["download_url"]) as response:
            content_type = response.headers.get_content_type()
            length = response.headers.get("Content-Length")
            binary = content_type.startswith("application/") and not content_type.endswith("json")
            if (not (binary or suffix == ".pchtxt" and content_type == "text/plain")
                    or response.headers.get("Content-Encoding", "identity").lower() != "identity"):
                raise Unavailable("response is not a public binary download")
            if length is not None and positive_int(length) != expected:
                raise Unavailable("response size does not match the API")
            with tempfile.TemporaryFile(dir=self.upstream.root) as archive:
                remaining = expected
                deadline = time.monotonic() + 120
                while remaining >= 0:
                    if time.monotonic() > deadline:
                        raise ImportFailure(f"file {file['id']}: download timed out")
                    part = response.read1(min(64 * 1024, remaining + 1))
                    if not part:
                        break
                    remaining -= len(part)
                    if remaining < 0:
                        raise Unavailable("download exceeds the API size")
                    sha.update(part)
                    checksum.update(part)
                    archive.write(part)
                if remaining != 0 or checksum.hexdigest() != md5:
                    raise Unavailable("download size or MD5 does not match the API")
                match = IPS_NAME.fullmatch(file["name"])
                if suffix == ".zip":
                    ids = zip_patch_ids(archive)
                elif suffix == ".pchtxt" and expected <= PATCH_TEXT_MAX:
                    archive.seek(0)
                    ids = sorted(nsobids(archive.read(PATCH_TEXT_MAX)))[:32]
                else:
                    ids = [match[1].upper()] if match else []
        return {"sha256": sha.hexdigest(), "build_ids": ids}


def candidate_ids(upstream, game_id, per_feed):
    feeds = []
    for sort in SORTS:
        query = urlencode({"_aFilters[Generic_Game]": game_id, "_nPage": 1,
                           "_nPerpage": per_feed, "_sSort": sort})
        result = upstream.metadata("/Mod/Index?" + query)
        if result is None:
            continue
        feed = result.get("_aRecords")
        if not isinstance(feed, list):
            raise ImportFailure(f"GameBanana game {game_id}: missing records in {sort} feed")
        feeds.append(feed[:per_feed])
    seen = set()
    for pair in zip_longest(*feeds):
        for item in pair:
            item = row(item)
            visibility = item.get("_sInitialVisibility")
            mod_id = positive_int(item.get("_idRow"))
            if (mod_id is not None and mod_id not in seen and not excluded(item)
                    and (visibility is None or str(visibility).lower() == "show")):
                seen.add(mod_id)
                yield mod_id


def manifest(upstream, title_ids, per_feed, limit, inspector):
    games = []
    for title_id in title_ids:
        name, game_id = GAMES[title_id]
        mods = []
        for mod_id in candidate_ids(upstream, game_id, per_feed):
            profile = upstream.metadata(f"/Mod/{mod_id}/ProfilePage", missing_ok=True)
            if profile and (entry := mod_entry(profile, mod_id, game_id, inspector, title_id)):
                mods.append(entry)
                if len(mods) == limit:
                    break
        if mods:
            games.append({"title_id": title_id, "name": name, "mods": mods})
        print(f"{title_id}: selected {len(mods)} mods for {name}", file=sys.stderr)
    return {"games": games}


def main():
    parser = argparse.ArgumentParser(description="Build a GameBanana mod-store import manifest.")
    parser.add_argument("--scratch", required=True, type=Path, help="private cache outside the checkout")
    parser.add_argument("--title-id", action="append", type=str.upper, choices=GAMES, metavar="TITLE_ID",
                        help="select a title; repeat for more (default: all mapped games)")
    parser.add_argument("--per-feed", type=int, default=12, choices=range(1, 21), metavar="1-20",
                        help="candidate cap per feed (default: 12)")
    parser.add_argument("--limit", type=int, default=20, choices=range(1, 21), metavar="1-20",
                        help="mod cap per game (default: 20)")
    parser.add_argument("--offline", action="store_true", help="use cached metadata only")
    parser.add_argument("--inspect-files", action="store_true", help="opt in to bounded archive inspection")
    parser.add_argument("--max-file-mib", type=int, choices=range(8, 33), default=8, metavar="8-32",
                        help="archive size cap when inspecting files (default: 8 MiB)")
    parser.add_argument("--max-files", type=int, choices=range(1, 33), default=8, metavar="1-32",
                        help="new archive downloads per run (default: 8)")
    args = parser.parse_args()
    try:
        root = args.scratch.expanduser().resolve()
        checkout = Path(__file__).resolve().parents[2]
        if root.is_relative_to(checkout):
            raise ImportFailure("--scratch must be outside the checkout")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        upstream = Upstream(root, args.offline)
        inspector = Inspector(upstream, args.max_file_mib, args.max_files) if args.inspect_files else None
        selected = dict.fromkeys(args.title_id or GAMES)
        result = manifest(upstream, selected, args.per_feed, args.limit, inspector)
        if args.offline:
            print(f"offline: skipped {upstream.missing_feeds} uncached feeds and "
                  f"{upstream.missing_profiles} uncached profiles", file=sys.stderr)
        if not result["games"]:
            raise ImportFailure("no eligible GameBanana listings for the selected titles")
        payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
        if len(payload) + 1 > 8 * MIB:
            raise ImportFailure("manifest exceeds the server's 8 MiB limit; select fewer titles")
        sys.stdout.buffer.write(payload + b"\n")
    except (ImportFailure, OSError, http.client.HTTPException, ValueError, KeyError) as error:
        print(f"import stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
