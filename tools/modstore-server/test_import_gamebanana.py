import hashlib
import io
import json
import os
import stat
import tempfile
import time
import unittest
import zipfile
from contextlib import redirect_stderr
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import import_gamebanana as gb

TITLE = "0100000000000001"
GAME = 987
MOD = 123
FILE = 456


def file_row(file_id=FILE, **fields):
    return {
        "_idRow": file_id,
        "_sFile": "example.zip",
        "_nFilesize": 123,
        "_tsDateAdded": 1704067200,
        "_sDownloadUrl": f"{gb.ORIGIN}/dl/{file_id}",
        "_sAnalysisState": "done",
        "_sAnalysisResult": "ok",
        "_sAvState": "done",
        "_sAvResult": "clean",
    } | fields


def profile(mod_id=MOD, **fields):
    return {
        "_idRow": mod_id,
        "_aGame": {"_idRow": GAME},
        "_sInitialVisibility": "Show",
        "_sName": "Example",
        "_sDescription": "<i>A précis</i>",
        "_sText": "<p>First &amp; second</p><script>hidden</script>\u202e<b>Next</b>",
        "_sProfileUrl": f"{gb.ORIGIN}/mods/{mod_id}",
        "_tsDateUpdated": 1704067200,
        "_tsDateModified": 1704153600,
        "_aCategory": {"_sName": "Frame Rate"},
        "_aCredits": [{
            "_sGroupName": "Key Authors",
            "_aAuthors": [
                {"_sName": "One", "_sProfileUrl": f"{gb.ORIGIN}/members/42"},
                {"_sName": "Two", "_sRole": "Code", "_sProfileUrl": "https://example.com"},
            ],
        }],
        "_aPreviewMedia": {"_aImages": [
            {"_sBaseUrl": "https://images.gamebanana.com/img/ss/mods", "_sFile": "sample.jpg"},
            {"_sBaseUrl": "https://images.gamebanana.com/img/ss/mods", "_sFile": "../out.jpg"},
            {"_sBaseUrl": "https://other.example/preview", "_sFile": "sample.jpg"},
        ]},
        "_aFiles": [file_row()],
    } | fields


class MetadataTests(unittest.TestCase):
    def test_mod_contract_and_sanitized_metadata(self):
        entry = gb.mod_entry(profile(), MOD, GAME, None, TITLE)
        self.assertEqual(entry, {
            "id": MOD, "name": "Example", "summary": "A précis",
            "description": "First & second\nNext", "updated_at": "2024-01-02T00:00:00Z",
            "page_url": f"{gb.ORIGIN}/mods/{MOD}", "type": "fps", "category": "Frame Rate",
            "author": "One, Two",
            "credits": [
                {"name": "One", "role": "Key Authors", "url": f"{gb.ORIGIN}/members/42"},
                {"name": "Two", "role": "Key Authors: Code", "url": ""},
            ],
            "preview_urls": ["https://images.gamebanana.com/img/ss/mods/sample.jpg"],
            "files": [{"id": FILE, "name": "example.zip", "size_bytes": 123,
                       "uploaded_at": "2024-01-01T00:00:00Z", "download_url": f"{gb.ORIGIN}/dl/{FILE}",
                       "sha256": None, "build_ids": []}],
        })

    def test_exclusions_and_array_caps(self):
        for fields in (
            {"_bHasContentRatings": True}, {"_sInitialVisibility": "Hide"},
            {"_sInitialVisibility": None}, {"_bIsWithheld": True}, {"_bIsPrivate": True},
            {"_bIsArchived": True}, {"_bIsObsolete": True}, {"_aGame": {"_idRow": GAME + 1}},
            {"_idRow": MOD + 1}, {"_aCredits": []}, {"_aFiles": []},
        ):
            with self.subTest(fields=fields):
                self.assertIsNone(gb.mod_entry(profile(**fields), MOD, GAME, None, TITLE))
        unsafe = [
            file_row(FILE + 20, _bIsArchived=True), file_row(FILE + 21, _sAvResult="infected"),
            file_row(FILE + 22, _bIsTainted=True), file_row(FILE + 23, _sAvState="pending"),
            file_row(FILE + 24, _sFile="bad\x00.zip"), file_row(FILE + 25, _sDownloadUrl="https://other.example/"),
        ]
        candidates = unsafe + [file_row()] + [file_row(FILE + offset) for offset in range(10)]
        entry = gb.mod_entry(profile(_aFiles=candidates), MOD, GAME, None, TITLE)
        self.assertEqual([item["id"] for item in entry["files"]], list(range(FILE, FILE + 8)))
        self.assertIsNone(gb.mod_entry(profile(_aFiles=unsafe), MOD, GAME, None, TITLE))

    def test_switch_one_does_not_inherit_explicit_switch_two_files(self):
        wrong = file_row(FILE + 1, _sFile="adjustment_switch2.zip")
        entry = gb.mod_entry(profile(_aFiles=[wrong, file_row()]), MOD, GAME, None, TITLE)
        self.assertEqual([item["id"] for item in entry["files"]], [FILE])
        self.assertIsNone(gb.mod_entry(
            profile(_sName="Switch 2 Edition example"), MOD, GAME, None, TITLE
        ))

    def test_kirby_shared_hub_uses_per_file_original_edition_evidence(self):
        split = 1756252800
        candidates = [
            file_row(FILE, _tsDateAdded=split - 1),
            file_row(FILE + 1, _tsDateAdded=split),
            file_row(FILE + 2, _tsDateAdded=split, _sFile=f"{gb.KIRBY_ORIGINAL}.zip"),
            file_row(FILE + 3, _tsDateAdded=split, _sDescription="For Switch 1"),
            file_row(FILE + 4, _tsDateAdded=split - 1, _sDescription="Requires Switch 2"),
        ]
        profile_fields = {"_aGame": {"_idRow": 15908}, "_aFiles": candidates}
        entry = gb.mod_entry(profile(**profile_fields), MOD, 15908, None, gb.KIRBY_ORIGINAL)
        self.assertEqual([item["id"] for item in entry["files"]], [FILE, FILE + 2, FILE + 3])
        self.assertIsNone(gb.mod_entry(
            profile(**profile_fields, _sName="Switch 2 Edition update"), MOD, 15908, None, gb.KIRBY_ORIGINAL,
        ))
        profile_fields["_aFiles"] = [candidates[1]]
        entry = gb.mod_entry(
            profile(**profile_fields, _sName="Switch 1 mod"), MOD, 15908, None, gb.KIRBY_ORIGINAL,
        )
        self.assertEqual([item["id"] for item in entry["files"]], [FILE + 1])

    def test_offline_cli_interleaves_feeds_and_uses_cached_profiles(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(gb.GAMES, {TITLE: ("Example Game", GAME)}):
            scratch = Path(directory)
            endpoints = {}
            for sort, ids in zip(gb.SORTS, ([MOD, MOD + 1], [MOD + 2, MOD])):
                query = gb.urlencode({"_aFilters[Generic_Game]": GAME, "_nPage": 1,
                                      "_nPerpage": 2, "_sSort": sort})
                endpoints["/Mod/Index?" + query] = {"_aRecords": [{"_idRow": mod_id} for mod_id in ids]}
            for mod_id in (MOD, MOD + 2):
                endpoints[f"/Mod/{mod_id}/ProfilePage"] = profile(mod_id)
            for endpoint, response in endpoints.items():
                key = hashlib.sha256((gb.API + endpoint).encode()).hexdigest()
                gb.write_cache(scratch / "metadata" / f"{key}.json", json.dumps(response).encode())
            with io.TextIOWrapper(io.BytesIO(), encoding="utf-8") as stdout, redirect_stderr(io.StringIO()):
                args = ["script", "--scratch", directory, "--title-id", TITLE, "--offline",
                        "--per-feed", "2", "--limit", "2"]
                with patch.object(gb.sys, "argv", args), patch.object(gb.sys, "stdout", stdout):
                    self.assertEqual(gb.main(), 0)
                data = json.loads(stdout.buffer.getvalue())
            self.assertEqual(len(data["games"]), 1)
            game = data["games"][0]
            self.assertEqual((game["title_id"], game["name"]), (TITLE, "Example Game"))
            self.assertEqual([mod["id"] for mod in game["mods"]], [MOD, MOD + 2])

    def test_offline_partial_run_omits_empty_titles_and_reports_misses(self):
        second = "0100000000000002"
        with tempfile.TemporaryDirectory() as directory, patch.dict(gb.GAMES, {
            TITLE: ("Example Game", GAME), second: ("Second Game", GAME + 1),
        }):
            scratch = Path(directory)
            query = gb.urlencode({"_aFilters[Generic_Game]": GAME, "_nPage": 1,
                                  "_nPerpage": 2, "_sSort": gb.SORTS[0]})
            responses = {
                f"/Mod/Index?{query}": {"_aRecords": [{"_idRow": MOD}, {"_idRow": MOD + 1}]},
                f"/Mod/{MOD}/ProfilePage": profile(),
            }
            for endpoint, response in responses.items():
                key = hashlib.sha256((gb.API + endpoint).encode()).hexdigest()
                gb.write_cache(scratch / "metadata" / f"{key}.json", json.dumps(response).encode())
            gb.write_cache(scratch / "retry-after.json", b'{"until":253402300799,"reason":"429"}')
            stderr = io.StringIO()
            with io.TextIOWrapper(io.BytesIO(), encoding="utf-8") as stdout, redirect_stderr(stderr):
                args = ["script", "--scratch", directory, "--title-id", TITLE, "--title-id", second,
                        "--offline", "--per-feed", "2"]
                with patch.object(gb.sys, "argv", args), patch.object(gb.sys, "stdout", stdout), patch.object(
                    gb.Upstream, "request", side_effect=AssertionError("offline network request")
                ):
                    self.assertEqual(gb.main(), 0)
                data = json.loads(stdout.buffer.getvalue())
            self.assertEqual([game["title_id"] for game in data["games"]], [TITLE])
            self.assertIn("selected 0 mods", stderr.getvalue())
            self.assertIn("skipped 3 uncached feeds and 1 uncached profiles", stderr.getvalue())


class Response(io.BytesIO):
    def __init__(self, code=200, body=b"", headers=None):
        super().__init__(body)
        self.code = code
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value


class UpstreamTests(unittest.TestCase):
    def test_only_empty_or_self_identifying_html_can_be_skipped_for_profiles(self):
        page = f"{gb.ORIGIN}/mods/{MOD}"
        markers = (f'<link rel="canonical" href="{page}">'
                   f'<meta content="{page}" property="og:url">').encode()
        verified = b"<html><head>" + markers + b"</head></html>"
        endpoint = f"/Mod/{MOD}/ProfilePage"
        cases = (
            (b"", {}, True),
            (verified, {}, True),
            (b"<html>unknown page</html>", {}, False),
            (verified.replace(b"<head>", b"<head><title>Log in</title>"), {}, False),
            (verified, {"cf-mitigated": "challenge"}, False),
        )
        for body, headers, skipped in cases:
            with self.subTest(body=body, headers=headers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                upstream = gb.Upstream(root)
                response = Response(body=body, headers={"Content-Type": "text/html"} | headers)
                with patch.object(upstream.opener, "open", return_value=response) as opened:
                    if skipped:
                        with redirect_stderr(io.StringIO()):
                            self.assertIsNone(upstream.metadata(endpoint, missing_ok=True))
                        self.assertIsNone(upstream.metadata(endpoint, missing_ok=True))
                        self.assertEqual([path.read_bytes() for path in (root / "metadata").iterdir()], [b"null"])
                        self.assertFalse(upstream.cooldown.exists())
                    else:
                        with self.assertRaises(gb.Blocked):
                            upstream.metadata(endpoint, missing_ok=True)
                        self.assertTrue(upstream.cooldown.exists())
                        self.assertFalse((root / "metadata").exists())
                    opened.assert_called_once()

    def test_shared_pacing_and_redirect_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            upstream = gb.Upstream(Path(directory))
            now = [50.0]
            opened = []
            location = "https://files.gamebanana.com/mods/example.zip"
            responses = [
                Response(body=b'{"_aRecords":[]}', headers={"Content-Type": "application/json"}),
                Response(302, headers={"Location": location}),
                Response(),
                Response(302, headers={"Location": "https://files.gamebanana.com.evil.example/mods/f"}),
            ]

            def open_request(request, timeout):
                self.assertEqual(request.get_header("User-agent"), gb.USER_AGENT)
                opened.append((request.full_url, now[0]))
                return responses.pop(0)

            def sleep(duration):
                now[0] += duration

            with patch.object(gb.time, "monotonic", side_effect=lambda: now[0]), patch.object(
                gb.time, "sleep", side_effect=sleep
            ), patch.object(upstream.opener, "open", side_effect=open_request):
                upstream.metadata("/Mod/Index")
                upstream.metadata("/Mod/Index")
                with upstream.archive(f"{gb.ORIGIN}/dl/{FILE}"):
                    pass
                with self.assertRaises(gb.Unavailable):
                    upstream.archive(f"{gb.ORIGIN}/dl/{FILE + 1}")
            self.assertEqual([value for _, value in opened], [50.0, 51.0, 52.0, 53.0])
            self.assertEqual(opened[2][0], location)

    def test_retry_after_persists_and_stops_a_second_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gb.time, "time", return_value=1000):
            root = Path(directory)
            upstream = gb.Upstream(root)
            response = Response(429, headers={"Retry-After": "90"})
            with patch.object(upstream.opener, "open", return_value=response) as opened:
                with self.assertRaisesRegex(gb.Blocked, "Retry-After: 90"):
                    upstream.metadata("/Mod/Index")
                opened.assert_called_once()
            self.assertTrue(response.closed)
            self.assertEqual(json.loads(upstream.cooldown.read_bytes())["until"], 1090)
            next_run = gb.Upstream(root)
            with patch.object(next_run.opener, "open") as opened:
                with self.assertRaisesRegex(gb.Blocked, "cooldown"):
                    next_run.metadata("/Mod/Index")
                opened.assert_not_called()

    def test_index_queries_are_game_scoped_and_deduplicate(self):
        class Fixtures:
            def metadata(self, endpoint):
                parts = urlsplit(endpoint)
                self_query = parse_qs(parts.query)
                self_test.assertEqual(parts.path, "/Mod/Index")
                self_test.assertEqual(self_query["_aFilters[Generic_Game]"], [str(GAME)])
                self_test.assertEqual(self_query["_nPage"], ["1"])
                self_test.assertEqual(self_query["_nPerpage"], ["3"])
                if self_query["_sSort"] == [gb.SORTS[0]]:
                    feed = [{"_idRow": 1}, {"_idRow": 2, "_sInitialVisibility": "Hide"}, {"_idRow": 3}]
                else:
                    self_test.assertEqual(self_query["_sSort"], [gb.SORTS[1]])
                    feed = [{"_idRow": 3}, {"_idRow": 4}, {"_idRow": 5, "_bHasContentRatings": True}]
                return {"_aRecords": feed}

        self_test = self
        self.assertEqual(list(gb.candidate_ids(Fixtures(), GAME, 3)), [1, 3, 4])


class InspectionTests(unittest.TestCase):
    def test_zip_names_only_and_hash_results_are_cached(self):
        payload = io.BytesIO()
        build_id = "a1b2c3d4e5f6a7b8"
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(f"exefs_patches/example/{build_id}.ips", b"")
            archive.writestr(build_id + ".txt", b"")
        raw = payload.getvalue()
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            upstream = gb.Upstream(root)
            inspector = gb.Inspector(upstream, 8, 1)
            original = gb.public_file(file_row(_nFilesize=len(raw)))
            file = original.copy()
            with patch.object(upstream, "archive", return_value=Response(
                body=raw, headers={"Content-Type": "application/zip", "Content-Length": str(len(raw))}
            )) as download, patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("read member")):
                inspector.inspect(file, md5)
                again = original.copy()
                inspector.inspect(again, md5)
                download.assert_called_once_with(file["download_url"])
            self.assertEqual(file, again)
            self.assertEqual(file["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(file["build_ids"], [build_id.upper()])
            self.assertEqual([path.name for path in root.iterdir()], ["inspection"])
            self.assertEqual(json.loads(next((root / "inspection").iterdir()).read_bytes()),
                             {"sha256": file["sha256"], "build_ids": file["build_ids"]})

    def test_patch_directives_and_strict_zip_paths_use_bounded_metadata(self):
        raw = io.BytesIO()
        long_id = "ab12" * 16
        cheat_id = "1234" * 4
        ips_id = "8888" * 4
        bogus = "ffff" * 4
        directive = f"@nsobid-{long_id}\r\n".encode()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(4):
                archive.writestr(f"mods/example/ExeFS/demo{index}.pchtxt",
                                 directive.ljust(gb.PATCH_TEXT_MAX, b"\n"))
            archive.writestr("exefs/over-budget.pchtxt", f"@nsobid-{bogus}\n")
            archive.writestr(f"contents/example/cheats/{cheat_id}.txt", b"")
            archive.writestr(f"exefs_patches/example/{ips_id}.ips32", b"")
            archive.writestr(f"misc/{bogus}.ips", b"")
            archive.writestr(f"exefs/../exefs/{bogus}.ips", b"")
            archive.writestr(f"/exefs/{bogus}.ips", b"")
            archive.writestr(f"cheats/{bogus}.txt.exe", b"")
            symlink = zipfile.ZipInfo(f"exefs/{bogus}.ips")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(symlink, b"ignored")
        self.assertEqual(gb.zip_patch_ids(raw), sorted([long_id.upper(), cheat_id, ips_id]))

    def test_direct_patch_text_reads_only_the_build_id_directive(self):
        build_id = "abcd1234" * 2
        raw = f"\ufeff    @nsobid-{build_id}\r\n".encode()
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            upstream = gb.Upstream(Path(directory))
            file = gb.public_file(file_row(_sFile="example.pchtxt", _nFilesize=len(raw)))
            with patch.object(upstream, "archive", return_value=Response(
                body=raw, headers={"Content-Type": "application/octet-stream"}
            )):
                gb.Inspector(upstream, 8, 1).inspect(file, md5)
            self.assertEqual(file["build_ids"], [build_id.upper()])

    def test_budget_and_md5_failure_leave_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            upstream = gb.Upstream(Path(directory))
            inspector = gb.Inspector(upstream, 8, 1)
            too_large = gb.public_file(file_row(_nFilesize=8 * gb.MIB + 1))
            small = gb.public_file(file_row(_nFilesize=3))
            md5 = "0" * 32
            with patch.object(upstream, "archive", return_value=Response(
                body=b"abc", headers={"Content-Type": "application/octet-stream"}
            )) as download:
                inspector.inspect(too_large, md5)
                inspector.inspect(small, md5)
                inspector.inspect(small, md5)
                download.assert_called_once()
            for file in (too_large, small):
                self.assertIsNone(file["sha256"])
                self.assertEqual(file["build_ids"], [])

    def test_unavailable_entries_yield_the_next_budget_then_expire(self):
        raw = b"archive fixture"
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            upstream = gb.Upstream(root)
            first = gb.public_file(file_row(_nFilesize=len(raw)))
            second = gb.public_file(file_row(FILE + 1, _nFilesize=len(raw)))
            with patch.object(upstream, "archive", side_effect=gb.Unavailable("file removed")):
                gb.Inspector(upstream, 8, 1).inspect(first, md5)
            cache = root / "inspection" / f"{FILE}-{md5}.json"
            self.assertTrue(json.loads(cache.read_text())["unavailable"])
            with patch.object(upstream, "archive", return_value=Response(
                body=raw, headers={"Content-Type": "application/zip"}
            )) as download:
                inspector = gb.Inspector(upstream, 8, 1)
                inspector.inspect(first, md5)
                inspector.inspect(second, md5)
                download.assert_called_once_with(second["download_url"])
            self.assertIsNone(first["sha256"])
            self.assertEqual(second["sha256"], hashlib.sha256(raw).hexdigest())
            expired = time.time() - gb.UNAVAILABLE_RETRY - 1
            os.utime(cache, (expired, expired))
            with patch.object(upstream, "archive", return_value=Response(
                body=raw, headers={"Content-Type": "application/zip"}
            )) as download:
                gb.Inspector(upstream, 8, 1).inspect(first, md5)
                download.assert_called_once_with(first["download_url"])
            self.assertEqual(first["sha256"], second["sha256"])

    def test_retries_after_transient_failure_and_ignores_legacy_null_cache(self):
        raw = b"archive fixture"
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        for error in (gb.Blocked("rate limited"), gb.ImportFailure("HTTP 503"), TimeoutError()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                upstream = gb.Upstream(root)
                file = gb.public_file(file_row(_nFilesize=len(raw)))
                cache = root / "inspection" / f"{FILE}-{md5}.json"
                with patch.object(upstream, "archive", side_effect=error), self.assertRaises(type(error)):
                    gb.Inspector(upstream, 8, 1).inspect(file, md5)
                self.assertFalse(cache.exists())
                gb.write_cache(cache, b'{"sha256":null,"build_ids":[]}')
                with patch.object(upstream, "archive", return_value=Response(
                    body=raw, headers={"Content-Type": "application/zip"}
                )) as download, redirect_stderr(io.StringIO()):
                    gb.Inspector(upstream, 8, 1).inspect(file, md5)
                    download.assert_called_once_with(file["download_url"])
                self.assertEqual(file["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(json.loads(cache.read_text()), {"sha256": file["sha256"], "build_ids": []})


if __name__ == "__main__":
    unittest.main()
