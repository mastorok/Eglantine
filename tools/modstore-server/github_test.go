package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const (
	testGitHubToken = "test-admin"
	testGitHubRoute = "/admin/v1/github"
	testGitHubRef   = "1234567890abcdef1234567890abcdef12345678"
	testGitHubURL   = "https://github.com/Example-Org/mods.catalog/tree/" + testGitHubRef
	testGitHubTitle = "0100000000000001"
	testSharedTitle = "0100000000000002"
)

func gitHubRequest(t *testing.T, h http.Handler, method, path, body string, status int, features ...string) []byte {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer "+testGitHubToken)
	r.Header.Set("Content-Type", "application/json")
	for _, feature := range features {
		r.Header.Add("X-Lichii-Catalog-Features", feature)
	}
	resp := httptest.NewRecorder()
	h.ServeHTTP(resp, r)
	if resp.Code != status {
		t.Fatalf("%s %s (%v): got %d, want %d: %s", method, path, features, resp.Code, status, resp.Body.String())
	}
	return resp.Body.Bytes()
}

func gitHubFixture(id, name string) githubModInput {
	return githubModInput{
		githubMetadata: githubMetadata{
			Repository: "Example-Org/mods.catalog",
			Commit:     testGitHubRef,
			License:    "NOASSERTION",
			Builds: []githubBuild{
				{ID: "1234567890ABCDEF", Target: "main"},
				{ID: strings.Repeat("A", 64), Target: "unknown"},
			},
		},
		ID: id, Name: name, Summary: "A GitHub listing", Description: "Link to the original project.",
		Author: "Example Org", Type: "fps", SourceURL: testGitHubURL,
		UpdatedAt: githubTimestamp{time.Date(2026, time.November, 22, 12, 11, 10, 0, time.FixedZone("CET", 3600))},
	}
}

func gitHubBody(t *testing.T, games ...githubGameInput) string {
	t.Helper()
	body, err := json.Marshal(githubManifest{Games: games})
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func TestGitHubCatalogUpgradeAndOptIn(t *testing.T) {
	dir := t.TempDir()
	oldDB, err := sql.Open("sqlite", filepath.Join(dir, "catalog.sqlite3"))
	if err != nil {
		t.Fatal(err)
	}
	const legacyView = `CREATE VIEW catalog_targets AS
	    SELECT mod_id, title_id, status='approved' AS available FROM versions
	    UNION ALL
	    SELECT mod_id, title_id, 1 AS available FROM gamebanana`
	const legacySchema = `
	    CREATE TABLE games (title_id TEXT PRIMARY KEY, name TEXT NOT NULL) STRICT;
	    CREATE TABLE mods (
	        id TEXT PRIMARY KEY, name TEXT NOT NULL, summary TEXT NOT NULL, description TEXT NOT NULL,
	        author TEXT NOT NULL, type TEXT NOT NULL CHECK (type IN ('graphics','fps','cheats','translation','gameplay')),
	        demo INTEGER NOT NULL CHECK (demo IN (0,1)), preview_sha TEXT, preview_type TEXT,
	        status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected')), updated_at INTEGER NOT NULL
	    ) STRICT;
	    CREATE TABLE versions (
	        mod_id TEXT NOT NULL REFERENCES mods(id), id TEXT NOT NULL, title_id TEXT NOT NULL REFERENCES games(title_id),
	        game_version TEXT NOT NULL, build_id TEXT NOT NULL, min_lichii_build TEXT NOT NULL, sha256 TEXT NOT NULL,
	        size_bytes INTEGER NOT NULL, unpacked_size_bytes INTEGER NOT NULL,
	        status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected')), created_at INTEGER NOT NULL,
	        PRIMARY KEY (mod_id,id)
	    ) STRICT;
	    CREATE TABLE gamebanana (
	        mod_id TEXT PRIMARY KEY REFERENCES mods(id), title_id TEXT NOT NULL REFERENCES games(title_id),
	        page_url TEXT NOT NULL, preview_url TEXT, metadata TEXT NOT NULL
	    ) STRICT;
	`
	if _, err := oldDB.Exec(legacySchema + legacyView); err != nil {
		oldDB.Close()
		t.Fatal(err)
	}
	_, err = oldDB.Exec(`INSERT INTO games (title_id,name) VALUES (?, 'B Existing game'), ('0100000000000003', 'C Existing game');
	    INSERT INTO mods (id,name,summary,description,author,type,demo,status,updated_at)
	        VALUES ('old-mod','B Existing mod','Old summary','Old description','Old author','fps',0,'approved',1),
	               ('other-old-mod','C Existing mod','Old summary','Old description','Old author','fps',0,'approved',1);
	    INSERT INTO versions VALUES ('old-mod','v1',?,'1.0','ABCD','build-1','abcd',1,1,'approved',1),
	                                ('other-old-mod','v1','0100000000000003','1.0','ABCD','build-1','abcd',1,1,'approved',1);`,
		testSharedTitle, testSharedTitle)
	if closeErr := oldDB.Close(); err != nil || closeErr != nil {
		t.Fatalf("creating the legacy catalog: %v; close: %v", err, closeErr)
	}
	s, err := openStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.db.Close() })
	var upgradedView string
	if err := s.db.QueryRow("SELECT sql FROM sqlite_schema WHERE name='catalog_targets'").Scan(&upgradedView); err != nil || upgradedView != legacyView {
		t.Fatalf("legacy catalog view changed: %q (%v)", upgradedView, err)
	}
	pub, admin := s.publicHandler(), s.adminHandler(testGitHubToken)
	alone := gitHubFixture("gh-alone", "A GitHub mod")
	shared := gitHubFixture("gh-shared", "D GitHub mod")
	shared.Builds = []githubBuild{}
	body := gitHubBody(t, githubGameInput{testGitHubTitle, "A GitHub-only game", []githubModInput{alone}},
		githubGameInput{testSharedTitle, "Do not replace the existing name", []githubModInput{shared}})
	gitHubRequest(t, pub, "PUT", testGitHubRoute, body, 404)
	imported := parse[map[string]int](t, gitHubRequest(t, admin, "PUT", testGitHubRoute, body, 200))
	if imported["games"] != 2 || imported["mods"] != 2 {
		t.Fatalf("unexpected import counts: %v", imported)
	}

	get := func(path string, status int, features ...string) []byte {
		return gitHubRequest(t, pub, "GET", publicPrefix+path, "", status, features...)
	}
	oldFirst := parse[testGamesPage](t, get("/games?limit=1", 200))
	oldLast := parse[testGamesPage](t, get("/games?limit=1&offset=1", 200))
	newFirst := parse[testGamesPage](t, get("/games?limit=1", 200, "github-v1"))
	newNext := parse[testGamesPage](t, get("/games?limit=1&offset=1", 200, "github-v1"))
	if !reflect.DeepEqual(oldFirst.Games, []game{{testSharedTitle, "B Existing game", 1}}) ||
		oldFirst.Next == nil || *oldFirst.Next != 1 || len(oldLast.Games) != 1 || oldLast.Games[0].TitleID != "0100000000000003" || oldLast.Next != nil ||
		!reflect.DeepEqual(newFirst.Games, []game{{testGitHubTitle, "A GitHub-only game", 1}}) || newFirst.Next == nil || *newFirst.Next != 1 ||
		!reflect.DeepEqual(newNext.Games, []game{{testSharedTitle, "B Existing game", 2}}) || newNext.Next == nil || *newNext.Next != 2 {
		t.Fatalf("unexpected catalog pages: old %+v %+v; new %+v %+v", oldFirst, oldLast, newFirst, newNext)
	}
	for _, tc := range []struct {
		path     string
		features []string
		ids      []string
		next     bool
	}{
		{"/mods?sort=name&limit=1", nil, []string{"old-mod"}, true},
		{"/mods?sort=name&limit=1&offset=1", nil, []string{"other-old-mod"}, false},
		{"/mods?sort=name&limit=1", []string{"github-v1"}, []string{"gh-alone"}, true},
		{"/mods?sort=name&limit=1&offset=1", []string{"github-v1"}, []string{"old-mod"}, true},
		{"/mods?title_id=" + testGitHubTitle, nil, []string{}, false},
		{"/mods?title_id=" + testGitHubTitle, []string{"github-v1"}, []string{"gh-alone"}, false},
		{"/mods?title_id=" + testSharedTitle + "&sort=name", []string{"github-v1"}, []string{"old-mod", "gh-shared"}, false},
	} {
		page := parse[testModsPage](t, get(tc.path, 200, tc.features...))
		ids := make([]string, 0, len(page.Mods))
		for _, m := range page.Mods {
			ids = append(ids, m.ID)
			if strings.HasPrefix(m.ID, "gh-") && (m.Source != "github" || m.SourceURL != testGitHubURL || m.PreviewURL != nil) {
				t.Fatalf("unexpected GitHub summary: %+v", m)
			}
		}
		if !reflect.DeepEqual(ids, tc.ids) || (page.Next != nil) != tc.next {
			t.Fatalf("%s (%v): got %+v, want %v", tc.path, tc.features, page, tc.ids)
		}
	}

	for _, features := range [][]string{nil, {"GitHub-v1"}, {"github-v1, future"}, {"github-v1", "github-v1"}} {
		get("/mods/gh-alone", 404, features...)
	}
	detail := parse[modDetail](t, get("/mods/gh-alone", 200, "github-v1"))
	if detail.Source != "github" || detail.SourceURL != testGitHubURL || !reflect.DeepEqual(detail.TitleIDs, []string{testGitHubTitle}) ||
		detail.PreviewURL != nil || detail.Versions == nil || len(detail.Versions) != 0 || detail.External != nil ||
		detail.Status != "" || detail.Description != alone.Description || detail.UpdatedAt != "2026-11-22T11:11:10.000Z" ||
		!reflect.DeepEqual(detail.GitHub, &alone.githubMetadata) {
		t.Fatalf("unexpected GitHub detail: %+v (%+v)", detail, detail.GitHub)
	}
	oldDetail := parse[modDetail](t, get("/mods/old-mod", 200, "github-v1"))
	if len(oldDetail.Versions) != 1 || oldDetail.Source != "submission" || oldDetail.GitHub != nil {
		t.Fatalf("old submission changed under opt-in: %+v", oldDetail)
	}
	get("/mods/gh-alone/versions/v1/download", 404, "github-v1")
	get("/mods/gh-alone/preview", 404, "github-v1")
}

func TestGitHubManifestValidation(t *testing.T) {
	api := newTestAPI(t)
	fixture := gitHubFixture("gh-fixture", "Fixture")
	body := gitHubBody(t, githubGameInput{testGitHubTitle, "Fixture game", []githubModInput{fixture}})
	api.put("/github", body, 200)
	for _, tc := range []struct{ old, replacement string }{
		{fixture.ID, "non-github-id"},
		{fixture.ID, "gh-" + strings.Repeat("a", 61)},
		{testGitHubTitle, "010000000000000Z"},
		{fixture.Repository, "Example-Org/.."},
		{fixture.Repository, "Example-Org/%2E%2E"},
		{fixture.Repository, "Example-Org@example.net/mods"},
		{testGitHubRef, strings.ToUpper(testGitHubRef)},
		{testGitHubRef, "main"},
		{testGitHubURL, "https://raw.githubusercontent.com/Example-Org/mods.catalog/" + testGitHubRef},
		{testGitHubURL, "https://github.com.example.net/Example-Org/mods.catalog/tree/" + testGitHubRef},
		{testGitHubURL, testGitHubURL + "/subpath"},
		{testGitHubURL, testGitHubURL + "?raw=1"},
		{testGitHubURL, testGitHubURL + "#readme"},
		{testGitHubURL, strings.Replace(testGitHubURL, "/tree/", "/archive/", 1) + ".zip"},
		{testGitHubURL, strings.Replace(testGitHubURL, fixture.Repository, "Other/repo", 1)},
		{"NOASSERTION", "MIT"},
		{`"type":"fps"`, `"type":"translation"`},
		{"2026-11-22T12:11:10+01:00", "2026-11-22T12:11:10"},
		{"2026-11-22T12:11:10+01:00", "2026-11-22T2:11:10+01:00"},
		{"2026-11-22T12:11:10+01:00", "2026-11-22T12:11:10+24:00"},
		{"1234567890ABCDEF", "1234567890abcdef"},
		{"1234567890ABCDEF", "0000000000000000"},
		{strings.Repeat("A", 64), "1234567890ABCDEF"},
		{strings.Repeat("A", 64), strings.Repeat("A", 63)},
		{`"target":"main"`, `"target":"sdk"`},
		{`"license":"NOASSERTION"`, `"license":"NOASSERTION","download_url":"https://example.com/archive.zip"`},
		{`"builds":[{"id":"1234567890ABCDEF","target":"main"},{"id":"` + strings.Repeat("A", 64) + `","target":"unknown"}]`, `"builds":null`},
	} {
		t.Run(tc.replacement, func(t *testing.T) {
			bad := strings.ReplaceAll(body, tc.old, tc.replacement)
			if bad == body {
				t.Fatalf("missing fixture value %q", tc.old)
			}
			api.put("/github", bad, 400)
		})
	}
	for i := 0; i < 128; i++ {
		fixture.Builds = append(fixture.Builds, githubBuild{ID: fmt.Sprintf("%016X", i+1), Target: "main"})
	}
	tooMany := gitHubBody(t, githubGameInput{testGitHubTitle, "Fixture game", []githubModInput{fixture}})
	api.put("/github", tooMany, 400)
	duplicate := gitHubBody(t, githubGameInput{testGitHubTitle, "Fixture game", []githubModInput{gitHubFixture("gh-fixture", "First"), gitHubFixture("gh-fixture", "Second")}})
	api.put("/github", duplicate, 400)
}

func TestGitHubPinnedDownloadMetadata(t *testing.T) {
	api := newTestAPI(t)
	pub := api.store.publicHandler()
	fixture := gitHubFixture("gh-direct", "Pinned files")
	body := gitHubBody(t, githubGameInput{testGitHubTitle, "Fixture game", []githubModInput{fixture}})
	digest := strings.Repeat("ab", 32)
	direct := map[string]any{
		"path": "mods/60 FPS [US] & EU/日本.PCHTXT", "size_bytes": 2 << 20, "sha256": digest, "kind": "pchtxt",
	}
	projection := map[string]any{
		"path": "archives/60 FPS [US].ZIP", "size_bytes": 64 << 20, "sha256": digest, "kind": "cheat",
		"member": "cheats/1234567890ABCDEF.TXT", "tag": "60 FPS",
	}
	withDownloads := func(first, second map[string]any) string {
		t.Helper()
		result := body
		for _, entry := range []struct {
			target   string
			download map[string]any
		}{{"main", first}, {"unknown", second}} {
			if entry.download == nil {
				continue
			}
			metadata, err := json.Marshal(entry.download)
			if err != nil {
				t.Fatal(err)
			}
			field := `"target":"` + entry.target + `"`
			result = strings.Replace(result, field, field+`,"download":`+string(metadata), 1)
		}
		return result
	}
	api.put("/github", withDownloads(projection, direct), 200)
	detailPath := "/api/v1/mods/gh-direct"
	gitHubRequest(t, pub, "GET", detailPath, "", 404)
	detail := parse[struct {
		GitHub *struct {
			Builds []struct {
				Download json.RawMessage `json:"download"`
			} `json:"builds"`
		} `json:"github"`
		Versions []version `json:"versions"`
	}](t, gitHubRequest(t, pub, "GET", detailPath, "", 200, "github-v1"))
	if detail.GitHub == nil || len(detail.GitHub.Builds) != 2 || detail.Versions == nil || len(detail.Versions) != 0 {
		t.Fatalf("unexpected pinned download details: %+v", detail)
	}
	for i, want := range []map[string]any{projection, direct} {
		fields := parse[map[string]json.RawMessage](t, detail.GitHub.Builds[i].Download)
		if len(fields) != len(want) {
			t.Fatalf("unexpected download %d: %s", i, detail.GitHub.Builds[i].Download)
		}
		for key, value := range want {
			encoded, err := json.Marshal(value)
			if err != nil || string(encoded) != string(fields[key]) {
				t.Fatalf("download %d %s: got %s, want %s (%v)", i, key, fields[key], encoded, err)
			}
		}
	}
	gitHubRequest(t, pub, "GET", detailPath+"/versions/1234567890ABCDEF/download", "", 404, "github-v1")
	api.put("/github", body, 200)
	if details := gitHubRequest(t, pub, "GET", detailPath, "", 200, "github-v1"); strings.Contains(string(details), `"download":`) {
		t.Fatalf("reimport retained removed downloads: %s", details)
	}

	for _, tc := range []struct {
		field string
		value any
		zip   bool
	}{
		{"path", "/mods/file.pchtxt", false},
		{"path", "mods//file.pchtxt", false},
		{"path", "mods/../file.pchtxt", false},
		{"path", "mods/./file.pchtxt", false},
		{"path", "mods/file.pchtxt/", false},
		{"path", "mods/file\n.pchtxt", false},
		{"path", "mods/file\\name.pchtxt", false},
		{"path", "mods/%2e%2e/file.pchtxt", false},
		{"path", "https://example.org/file.pchtxt", false},
		{"path", "mods/file?query.pchtxt", false},
		{"path", "mods/file#fragment.pchtxt", false},
		{"path", "not-a-patch.ips", false},
		{"path", "missing-member.zip", false},
		{"path", "not-an-archive.txt", true},
		{"path", strings.Repeat("é", 253) + ".pchtxt", false},
		{"size_bytes", 0, false},
		{"size_bytes", -1, false},
		{"size_bytes", 1.5, false},
		{"size_bytes", (2 << 20) + 1, false},
		{"size_bytes", (64 << 20) + 1, true},
		{"sha256", strings.ToUpper(digest), false},
		{"sha256", digest[:63], false},
		{"kind", "exe", false},
		{"member", "cheats/file.txt", false},
		{"member", "", true},
		{"member", nil, true},
		{"member", "../cheats/file.txt", true},
		{"member", "cheats/file.exe", true},
		{"member", "cheats/1234567890ABCDFE.txt", true},
		{"member", "cheats/file?.txt", true},
		{"tag", "60 FPS", false},
		{"tag", "[60 FPS]", true},
		{"tag", " ", true},
		{"tag", "{Master Code}", true},
		{"tag", "60 FPS\nOther", true},
		{"tag", "60 [FPS]", true},
		{"tag", "60\tFPS", true},
		{"tag", strings.Repeat("a", 161), true},
		{"tag", strings.Repeat("𐌀", 81), true},
		{"tag", "60\u200b FPS", true},
		{"tag", "60\ufffd FPS", true},
		{"tag", "", true},
		{"tag", nil, true},
		{"url", "https://example.org/file.pchtxt", false},
	} {
		t.Run(fmt.Sprintf("%s=%v", tc.field, tc.value), func(t *testing.T) {
			base := direct
			if tc.zip {
				base = projection
			}
			bad := make(map[string]any, len(base)+1)
			for key, value := range base {
				bad[key] = value
			}
			bad[tc.field] = tc.value
			if tc.zip {
				api.put("/github", withDownloads(bad, nil), 400)
			} else {
				api.put("/github", withDownloads(nil, bad), 400)
			}
		})
	}
	api.put("/github", strings.Replace(body, `"target":"main"`, `"target":"main","download":null`, 1), 400)
	api.put("/github", withDownloads(map[string]any{}, nil), 400)
	api.put("/github", withDownloads(direct, nil), 400)
	api.put("/github", withDownloads(nil, projection), 400)
	api.put("/github", strings.Replace(withDownloads(projection, nil), `"target":"main"`, `"target":"unknown"`, 1), 400)
	untaggedZIP := map[string]any{
		"path": projection["path"], "size_bytes": projection["size_bytes"], "sha256": digest,
		"kind": "cheat", "member": projection["member"],
	}
	api.put("/github", withDownloads(untaggedZIP, nil), 400)
	cheat := map[string]any{"path": "cheats/1234567890abcdef.TxT", "size_bytes": 1, "sha256": digest, "kind": "cheat"}
	api.put("/github", withDownloads(cheat, nil), 200)
	cheat["path"] = "cheats/1234567890abcdfe.txt"
	api.put("/github", withDownloads(cheat, nil), 400)
	for _, suffix := range []string{".ips", ".IPS32"} {
		patch := map[string]any{"path": "patches/" + strings.Repeat("a", 64) + suffix,
			"size_bytes": 1, "sha256": digest, "kind": "ips"}
		api.put("/github", withDownloads(nil, patch), 200)
		patch["path"] = "patches/" + strings.Repeat("b", 64) + suffix
		api.put("/github", withDownloads(nil, patch), 400)
	}
}

func TestGitHubImportsKeepModerationAndProtectSource(t *testing.T) {
	api := newTestAPI(t)
	existing := gitHubFixture("gh-existing", "First import")
	omitted := gitHubFixture("gh-omitted", "Omitted from the second import")
	group := githubGameInput{testGitHubTitle, "Fixture game", []githubModInput{existing, omitted}}
	api.put("/github", gitHubBody(t, group), 200)
	api.put("/mods/gh-existing/status", `{"status":"rejected"}`, 204)
	existing.Name = "Updated listing"
	existing.Commit = strings.Repeat("e", 40)
	existing.SourceURL = "https://github.com/Example-Org/mods.catalog/tree/" + existing.Commit
	existing.Builds = []githubBuild{}
	group.Mods = []githubModInput{existing}
	api.put("/github", gitHubBody(t, group), 200)
	pub, admin := api.store.publicHandler(), api.store.adminHandler(testGitHubToken)
	gitHubRequest(t, pub, "GET", "/api/v1/mods/gh-existing", "", 404, "github-v1")
	gitHubRequest(t, pub, "GET", "/api/v1/mods/gh-omitted", "", 200, "github-v1")
	detail := parse[modDetail](t, gitHubRequest(t, admin, "GET", "/admin/v1/mods/gh-existing", "", 200))
	if detail.Status != "rejected" || detail.Name != existing.Name || detail.SourceURL != existing.SourceURL ||
		!reflect.DeepEqual(detail.GitHub, &existing.githubMetadata) || detail.Versions == nil || len(detail.Versions) != 0 {
		t.Fatalf("unexpected reimport result: %+v (%+v)", detail, detail.GitHub)
	}
	api.put("/mods/gh-existing", `{"name":"Not GitHub","summary":"Bad","description":"Bad","author":"Bad","type":"fps"}`, 409)
	api.request("PUT", api.admin, "/admin/v1/mods/gh-existing/preview", api.token, "image/png", nil, 409)
	api.request("PUT", api.admin, "/admin/v1/mods/gh-existing/versions/v1", api.token, "application/zip", nil, 409)
	api.request("GET", api.admin, "/admin/v1/mods/gh-existing/versions/v1/download", api.token, "", nil, 404)

	api.put("/mods/gh-old-submission", `{"name":"Own submission","summary":"Existing","description":"Existing","author":"Example","type":"fps"}`, 204)
	newMod := gitHubFixture("gh-rollback", "Must not persist")
	collision := gitHubFixture("gh-old-submission", "Must not replace a submission")
	conflictingRepo := existing
	conflictingRepo.Repository = "Other/repo"
	conflictingRepo.SourceURL = "https://github.com/Other/repo/tree/" + existing.Commit
	for _, bad := range [][]githubGameInput{
		{{testGitHubTitle, "Fixture", []githubModInput{newMod, collision}}},
		{{testSharedTitle, "Different title", []githubModInput{newMod, existing}}},
		{{testGitHubTitle, "Fixture", []githubModInput{newMod, conflictingRepo}}},
	} {
		api.put("/github", gitHubBody(t, bad...), 409)
		gitHubRequest(t, admin, "GET", "/admin/v1/mods/gh-rollback", "", 404)
	}
	var gameExists bool
	if err := api.store.db.QueryRowContext(context.Background(), "SELECT EXISTS (SELECT 1 FROM games WHERE title_id=?)", testSharedTitle).
		Scan(&gameExists); err != nil || gameExists {
		t.Fatalf("failed import leaked a game: %v (%v)", gameExists, err)
	}
}
