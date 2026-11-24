package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
)

const gameBananaSample = `{"games":[{"title_id":"0400C3F00006E000","name":"Mario Kart World","mods":[{
  "id":123,"name":"Circuit sample","summary":"A test listing","description":"Plain text.\nCredits are separate.",
  "author":"Circuit Lab","credits":[{"name":"Circuit Lab","role":"Author","url":"https://gamebanana.com/members/789"}],
  "type":"graphics","category":"UI","page_url":"https://gamebanana.com/mods/123",
  "preview_urls":["https://images.gamebanana.com/img/ss/mods/circuit.jpg"],
  "updated_at":"2026-10-18T14:11:02Z","files":[
    {"id":456,"name":"circuit.zip","size_bytes":1048576,"uploaded_at":"2026-10-17T10:02:00Z",
     "download_url":"https://gamebanana.com/dl/456",
     "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","build_ids":["1111111111111111"]},
    {"id":457,"name":"alternate.7z","size_bytes":1024,"uploaded_at":"2026-10-16T08:00:00Z",
     "download_url":"https://gamebanana.com/dl/457","sha256":null,"build_ids":[]}
  ]}]}]}`

func TestGameBananaImport(t *testing.T) {
	api := newTestAPI(t)
	if err := api.store.seedDemo(context.Background()); err != nil {
		t.Fatal(err)
	}
	api.put("/gamebanana/drop-demos", "", http.StatusConflict)
	api.request(http.MethodPut, api.public, adminPrefix+"/gamebanana", api.token,
		"application/json", []byte(gameBananaSample), http.StatusNotFound)
	api.request(http.MethodPut, api.admin, adminPrefix+"/gamebanana", "",
		"application/json", []byte(gameBananaSample), http.StatusUnauthorized)
	api.put("/gamebanana", gameBananaSample, http.StatusOK)
	api.put("/gamebanana", gameBananaSample, http.StatusOK)
	detail := parse[modDetail](t, api.get("/api/v1/mods/gb-123", 200))
	if detail.Source != "gamebanana" || detail.SourceURL != "https://gamebanana.com/mods/123" ||
		detail.External == nil || detail.External.Compatibility != "unverified" || len(detail.Versions) != 0 ||
		detail.PreviewURL == nil || *detail.PreviewURL != "https://images.gamebanana.com/img/ss/mods/circuit.jpg" ||
		!reflect.DeepEqual(detail.TitleIDs, []string{"0400C3F00006E000"}) {
		t.Fatalf("unexpected public external detail: %+v", detail)
	}
	m := detail.External
	if m.ID != 123 || m.Category != "UI" || len(m.Credits) != 1 || len(m.Files) != 2 ||
		m.Files[0].DownloadURL != "https://gamebanana.com/dl/456" || m.Files[0].SHA256 == nil ||
		!reflect.DeepEqual(m.Files[0].BuildIDs, []string{"1111111111111111"}) ||
		m.Files[1].SHA256 != nil || len(m.Files[1].BuildIDs) != 0 {
		t.Fatalf("unexpected external metadata: %+v", m)
	}
	api.get("/api/v1/mods/gb-123/versions/456/download", 404)
	api.get("/api/v1/mods/gb-123/preview", 404)
	matching := parse[testModsPage](t, api.get("/api/v1/mods?title_id=0400C3F00006E000&q=sample&type=graphics", 200))
	if len(matching.Mods) != 1 || matching.Mods[0].ID != "gb-123" {
		t.Fatalf("GameBanana listing missing from filtered mods: %+v", matching)
	}
	api.put("/gamebanana/drop-demos", "", http.StatusNoContent)
	games := parse[testGamesPage](t, api.get("/api/v1/games", 200))
	if len(games.Games) != 1 || games.Games[0].ModCount != 1 {
		t.Fatalf("game should count one real listing after dropping demos: %+v", games)
	}
	api.get("/api/v1/mods/demo-kart-cheats", 404)
	assets, err := os.ReadDir(api.store.assets)
	if err != nil || len(assets) != 10 {
		t.Fatalf("linked entries must not change the stored assets: %d, %v", len(assets), err)
	}
	api.put("/mods/gb-123/status", `{"status":"rejected"}`, http.StatusNoContent)
	api.put("/gamebanana", gameBananaSample, http.StatusOK)
	api.get("/api/v1/mods/gb-123", 404)
}

func TestGameBananaRejectsInvalidManifest(t *testing.T) {
	api := newTestAPI(t)
	api.put("/gamebanana", strings.Replace(gameBananaSample, "https://gamebanana.com/dl/456", "https://example.com/456", 1), 400)
	api.put("/gamebanana", strings.Replace(gameBananaSample, "https://images.gamebanana.com/", "https://example.com/", 1), 400)
	api.put("/gamebanana", strings.Replace(gameBananaSample, "1111111111111111", "1234", 1), 400)
	api.get("/api/v1/mods/gb-123", 404)
	api.put("/gamebanana", gameBananaSample, 200)
	api.put("/gamebanana", strings.Replace(gameBananaSample, "0400C3F00006E000", "040077201BE50000", 1), 409)
	games := parse[testGamesPage](t, api.get("/api/v1/games", 200))
	if len(games.Games) != 1 {
		t.Fatal("the rejected reassignment should not change the catalog")
	}
}

func TestGameBananaAmbiguousLayout(t *testing.T) {
	api := newTestAPI(t)
	sample := strings.Replace(gameBananaSample, "alternate.7z", "alternate.zip", 1)
	withLayout := func(value string) string {
		return strings.Replace(sample, `"sha256":null`, `"ambiguous_layout":`+value+`,"sha256":null`, 1)
	}
	checkFiles := func(id, wantLayout string) {
		t.Helper()
		detail := parse[struct {
			External struct {
				Files []struct {
					ID              int64           `json:"id"`
					SHA256          json.RawMessage `json:"sha256"`
					AmbiguousLayout json.RawMessage `json:"ambiguous_layout"`
				} `json:"files"`
			} `json:"external"`
		}](t, api.get("/api/v1/mods/"+id, 200))
		files := detail.External.Files
		if len(files) != 2 || files[0].ID != 456 || len(files[0].AmbiguousLayout) != 0 ||
			files[1].ID != 457 || string(files[1].SHA256) != "null" ||
			string(files[1].AmbiguousLayout) != wantLayout {
			t.Fatalf("files = %+v, want 456 unchanged and 457 with null SHA and layout %q", files, wantLayout)
		}
	}

	api.put("/gamebanana", withLayout(`"true"`), http.StatusBadRequest)
	api.put("/gamebanana", withLayout("null"), http.StatusBadRequest)
	api.put("/gamebanana", sample, http.StatusOK)
	checkFiles("gb-123", "")
	api.put("/gamebanana", withLayout("true"), http.StatusOK)
	checkFiles("gb-123", "true")
	api.put("/gamebanana", strings.Replace(sample, "alternate.zip", "renamed.zip", 1), http.StatusOK)
	checkFiles("gb-123", "true")
	api.put("/gamebanana", strings.ReplaceAll(sample, "123", "124"), http.StatusOK)
	checkFiles("gb-124", "")
	api.put("/gamebanana", withLayout("false"), http.StatusOK)
	checkFiles("gb-123", "false")
	api.put("/gamebanana", sample, http.StatusOK)
	checkFiles("gb-123", "false")
}
