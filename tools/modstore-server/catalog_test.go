package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPublicListIndependentOfDetailAndAssetLimits(t *testing.T) {
	api := newTestAPI(t)
	pub := api.store.publicHandler()
	for range cap(catalogDetailSlots) {
		catalogDetailSlots <- struct{}{}
	}
	for range cap(api.store.assetTransfers) {
		api.store.assetTransfers <- struct{}{}
	}
	defer func() {
		for range cap(catalogDetailSlots) {
			<-catalogDetailSlots
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	for _, route := range []string{"/api/v1/games", "/api/v1/mods"} {
		w := httptest.NewRecorder()
		pub.ServeHTTP(w, httptest.NewRequestWithContext(ctx, "GET", route, nil))
		if w.Code != http.StatusOK || w.Header().Get("ETag") == "" {
			t.Fatalf("blocked list %s got %d %v", route, w.Code, w.Header())
		}
	}
	for _, route := range []string{"/api/v1/mods/test/preview", "/api/v1/mods/test/versions/v1/download"} {
		w := catalogRequest(pub, "GET", route)
		if w.Code != http.StatusServiceUnavailable || w.Header().Get("Cache-Control") != "no-store" {
			t.Fatalf("saturated asset %s got %d %v", route, w.Code, w.Header())
		}
	}
}

func checkCatalogPage[T comparable](t *testing.T, got []T, next *int, want []T, wantNext int) {
	t.Helper()
	if got == nil || !slices.Equal(got, want) {
		t.Fatalf("items = %+v, want %+v as an array", got, want)
	}
	if wantNext == 0 && next != nil {
		t.Fatalf("next_offset = %d, want null", *next)
	}
	if wantNext != 0 && (next == nil || *next != wantNext) {
		t.Fatalf("next_offset does not equal %d", wantNext)
	}
}

func checkCatalogMods(t *testing.T, body []byte, next int, ids ...string) []mod {
	t.Helper()
	page := parse[testModsPage](t, body)
	if page.Mods == nil {
		t.Fatal("mods must be an array")
	}
	got := make([]string, len(page.Mods))
	for i, m := range page.Mods {
		got[i] = m.ID
	}
	checkCatalogPage(t, got, page.Next, ids, next)
	return page.Mods
}

func TestPublicCatalogOrderAndSearch(t *testing.T) {
	api := newTestAPI(t)
	fixtures := []struct {
		title, id, name string
		day             int
	}{
		{"0100000000000004", "gh-four", "éclair", 2},
		{"0100000000000002", "gh-two", "Alpha", 1},
		{"0100000000000003", "gh-three", "Élan", 2},
		{"0100000000000001", "gh-one", "alpha", 1},
	}
	var games []githubGameInput
	for _, f := range fixtures {
		m := gitHubFixture(f.id, f.name)
		m.UpdatedAt = githubTimestamp{time.Date(2026, time.November, f.day, 0, 0, 0, 0, time.UTC)}
		m.Description = "Hidden Route_100% notes"
		if f.id == "gh-four" {
			m.Type = "graphics"
		}
		games = append(games, githubGameInput{f.title, f.name, []githubModInput{m}})
	}
	api.put("/github", gitHubBody(t, games...), http.StatusOK)
	pub := api.store.publicHandler()
	get := func(t *testing.T, path string) []byte {
		t.Helper()
		return gitHubRequest(t, pub, "GET", publicPrefix+path, "", http.StatusOK, "github-v1")
	}

	first := parse[testGamesPage](t, get(t, "/games?limit=2"))
	checkCatalogPage(t, first.Games, first.Next, []game{
		{fixtures[3].title, "alpha", 1}, {fixtures[1].title, "Alpha", 1},
	}, 2)
	last := parse[testGamesPage](t, get(t, "/games?limit=2&offset=2"))
	checkCatalogPage(t, last.Games, last.Next, []game{
		{fixtures[2].title, "Élan", 1}, {fixtures[0].title, "éclair", 1},
	}, 0)
	empty := parse[testGamesPage](t, get(t, "/games?offset=4"))
	checkCatalogPage(t, empty.Games, empty.Next, nil, 0)

	byDescription := "sort=name&type=fps&q=" + url.QueryEscape(" ROUTE_100% ") + "&limit=2"
	for _, tc := range []struct {
		query string
		next  int
		ids   []string
	}{
		{"sort=name", 0, []string{"gh-one", "gh-two", "gh-three", "gh-four"}},
		{"sort=updated", 0, []string{"gh-four", "gh-three", "gh-one", "gh-two"}},
		{"", 0, []string{"gh-four", "gh-three", "gh-one", "gh-two"}},
		{byDescription, 2, []string{"gh-one", "gh-two"}},
		{byDescription + "&offset=2", 0, []string{"gh-three"}},
		{"q=" + url.QueryEscape("ROUTE%"), 0, nil},
		{"q=" + url.QueryEscape("ÉLAN"), 0, []string{"gh-three"}},
		{"q=" + url.QueryEscape("élan"), 0, nil},
		{"q=github+listing", 0, []string{"gh-four", "gh-three", "gh-one", "gh-two"}},
		{"q=example+org", 0, []string{"gh-four", "gh-three", "gh-one", "gh-two"}},
	} {
		t.Run(tc.query, func(t *testing.T) {
			body := get(t, "/mods?"+tc.query)
			checkCatalogMods(t, body, tc.next, tc.ids...)
			if bytes.Contains(body, []byte(`"description":`)) {
				t.Fatal("list exposed the detail-only description")
			}
		})
	}
}

func TestPublicCatalogFeatureIsolationAfterImport(t *testing.T) {
	api := newTestAPI(t)
	pub := api.store.publicHandler()
	const shared = "0400C3F00006E000"
	const sharedMods = publicPrefix + "/mods?title_id=0400c3f00006e000&sort=name"
	const detailPath = "/mods/gh-circuit"
	get := func(path string, status int, features ...string) []byte {
		t.Helper()
		return gitHubRequest(t, pub, "GET", path, "", status, features...)
	}
	check := func(sharedCount, next int, features []string, ids ...string) {
		t.Helper()
		body := get(publicPrefix+"/games?limit=1", http.StatusOK, features...)
		page := parse[testGamesPage](t, body)
		checkCatalogPage(t, page.Games, page.Next, []game{{shared, "Mario Kart World", sharedCount}}, next)
		for _, m := range checkCatalogMods(t, get(sharedMods, http.StatusOK, features...), 0, ids...) {
			if !slices.Equal(m.TitleIDs, []string{shared}) || m.Status != "" {
				t.Fatalf("unexpected public title or moderation status: %+v", m)
			}
		}
	}

	optIn := []string{"github-v1"}
	for _, features := range [][]string{nil, optIn} {
		empty := parse[testGamesPage](t, get(publicPrefix+"/games?limit=1", http.StatusOK, features...))
		checkCatalogPage(t, empty.Games, empty.Next, nil, 0)
		checkCatalogMods(t, get(sharedMods, http.StatusOK, features...), 0)
	}
	get(publicPrefix+"/mods/gb-123", http.StatusNotFound)
	api.put("/gamebanana", gameBananaSample, http.StatusOK)
	check(1, 0, nil, "gb-123")
	check(1, 0, optIn, "gb-123")
	get(publicPrefix+"/mods/gb-123", http.StatusOK)
	get(publicPrefix+detailPath, http.StatusNotFound, optIn...)

	patch := gitHubFixture("gh-circuit", "A circuit patch")
	only := gitHubFixture("gh-only", "Only on GitHub")
	api.put("/github", gitHubBody(t,
		githubGameInput{shared, "Do not rename the shared game", []githubModInput{patch}},
		githubGameInput{testGitHubTitle, "Z GitHub-only game", []githubModInput{only}},
	), http.StatusOK)

	check(2, 1, optIn, patch.ID, "gb-123")
	check(1, 0, nil, "gb-123")
	check(1, 0, []string{"unknown"}, "gb-123")
	check(2, 1, optIn, patch.ID, "gb-123")
	onlyPage := parse[testGamesPage](t, get(publicPrefix+"/games?limit=1&offset=1", http.StatusOK, optIn...))
	checkCatalogPage(t, onlyPage.Games, onlyPage.Next, []game{{testGitHubTitle, "Z GitHub-only game", 1}}, 0)

	get(publicPrefix+detailPath, http.StatusNotFound)
	detail := parse[modDetail](t, get(publicPrefix+detailPath, http.StatusOK, optIn...))
	if detail.Source != "github" || detail.GitHub == nil || detail.Description != patch.Description || detail.Status != "" {
		t.Fatalf("unexpected opted-in detail: %+v", detail)
	}
	get(publicPrefix+detailPath, http.StatusNotFound)
	admin, _ := api.request("GET", api.admin, adminPrefix+detailPath, api.token, "", nil, http.StatusOK)
	if private := parse[modDetail](t, admin); private.Status != "approved" || private.GitHub == nil {
		t.Fatalf("admin detail lost GitHub visibility without an opt-in header: %+v", private)
	}
}

func TestPublicCatalogRefreshAfterModeration(t *testing.T) {
	api := newTestAPI(t)
	const (
		id           = "catalog-card"
		path         = "/mods/" + id
		title        = "040074A01BF12000"
		otherTitle   = "040077201BE50000"
		metadata     = `{"name":"Catalog Card","summary":"A test card","description":"Initial text.","author":"Example","type":"fps"}`
		approvedJSON = `{"status":"approved"}`
		rejectedJSON = `{"status":"rejected"}`
	)
	api.put("/games/"+title, `{"name":"First game"}`, http.StatusNoContent)
	api.put("/games/"+otherTitle, `{"name":"Other game"}`, http.StatusNoContent)
	api.put(path, metadata, http.StatusNoContent)
	var archive bytes.Buffer
	if _, err := writeDemoZIP(&archive, id); err != nil {
		t.Fatal(err)
	}
	for _, upload := range []struct{ version, title string }{{"v1", title}, {"v2", title}, {"other", otherTitle}} {
		q := url.Values{
			"title_id": {upload.title}, "game_version": {"1.0"}, "build_id": {strings.Repeat("A", 64)},
			"min_lichii_build": {"build-1"},
		}
		api.request("PUT", api.admin, adminPrefix+path+"/versions/"+upload.version+"?"+q.Encode(),
			api.token, "application/zip", archive.Bytes(), http.StatusCreated)
	}
	check := func(games []game, description string, versions ...string) {
		t.Helper()
		page := parse[testGamesPage](t, api.get(publicPrefix+"/games", http.StatusOK))
		checkCatalogPage(t, page.Games, page.Next, games, 0)
		titles := make([]string, 0, len(games))
		for _, g := range games {
			titles = append(titles, g.TitleID)
		}
		slices.Sort(titles)
		body := api.get(publicPrefix+"/mods", http.StatusOK)
		if len(games) == 0 {
			checkCatalogMods(t, body, 0)
		} else {
			checkCatalogMods(t, body, 0, id)
		}
		for _, gameTitle := range []string{title, otherTitle} {
			body := api.get(publicPrefix+"/mods?title_id="+gameTitle, http.StatusOK)
			if slices.Contains(titles, gameTitle) {
				mods := checkCatalogMods(t, body, 0, id)
				if !slices.Equal(mods[0].TitleIDs, titles) || mods[0].Status != "" {
					t.Fatalf("unexpected filtered mod: %+v", mods[0])
				}
			} else {
				checkCatalogMods(t, body, 0)
			}
		}
		if len(versions) == 0 {
			api.get(publicPrefix+path, http.StatusNotFound)
			return
		}
		d := parse[modDetail](t, api.get(publicPrefix+path, http.StatusOK))
		var got []string
		for _, v := range d.Versions {
			if v.Status != "" {
				t.Fatalf("public detail exposed version moderation status: %+v", v)
			}
			got = append(got, v.ID)
		}
		slices.Sort(got)
		if !slices.Equal(got, versions) || !slices.Equal(d.TitleIDs, titles) || d.Description != description || d.Status != "" {
			t.Fatalf("unexpected detail: %+v; want versions %v and titles %v", d, versions, titles)
		}
	}

	check(nil, "")
	api.put(path+"/status", approvedJSON, http.StatusNoContent)
	check(nil, "")
	api.put(path+"/versions/v1/status", approvedJSON, http.StatusNoContent)
	check([]game{{title, "First game", 1}}, "Initial text.", "v1")
	api.put(path+"/versions/v2/status", approvedJSON, http.StatusNoContent)
	check([]game{{title, "First game", 1}}, "Initial text.", "v1", "v2")

	list, _ := api.request("GET", api.admin, adminPrefix+"/mods?title_id="+otherTitle+"&status=approved", api.token, "", nil, http.StatusOK)
	mods := checkCatalogMods(t, list, 0, id)
	if !slices.Equal(mods[0].TitleIDs, []string{title, otherTitle}) || mods[0].Status != "approved" {
		t.Fatalf("admin filter no longer sees the pending target: %+v", mods[0])
	}

	api.put(path+"/versions/other/status", approvedJSON, http.StatusNoContent)
	check([]game{{title, "First game", 1}, {otherTitle, "Other game", 1}}, "Initial text.", "other", "v1", "v2")
	api.put("/games/"+otherTitle, `{"name":"A renamed game"}`, http.StatusNoContent)
	both := []game{{otherTitle, "A renamed game", 1}, {title, "First game", 1}}
	check(both, "Initial text.", "other", "v1", "v2")
	api.put(path+"/versions/v1/status", rejectedJSON, http.StatusNoContent)
	check(both, "Initial text.", "other", "v2")
	api.put(path+"/versions/v2/status", rejectedJSON, http.StatusNoContent)
	check(both[:1], "Initial text.", "other")
	api.put(path+"/status", rejectedJSON, http.StatusNoContent)
	check(nil, "")

	admin, _ := api.request("GET", api.admin, adminPrefix+path, api.token, "", nil, http.StatusOK)
	d := parse[modDetail](t, admin)
	if d.Status != "rejected" || len(d.Versions) != 3 || !slices.Equal(d.TitleIDs, []string{title, otherTitle}) {
		t.Fatalf("admin detail lost rejected records: %+v", d)
	}
	for _, v := range d.Versions {
		want := "rejected"
		if v.ID == "other" {
			want = "approved"
		}
		if v.Status != want {
			t.Fatalf("admin version %q status = %q, want %q", v.ID, v.Status, want)
		}
	}
	api.put(path+"/status", approvedJSON, http.StatusNoContent)
	check(both[:1], "Initial text.", "other")
	api.put(path, strings.Replace(metadata, "Initial text.", "Revised text.", 1), http.StatusNoContent)
	check(nil, "")
	api.put(path+"/status", approvedJSON, http.StatusNoContent)
	check(both[:1], "Revised text.", "other")
}
