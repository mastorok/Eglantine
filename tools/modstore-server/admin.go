package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

var titlePattern = regexp.MustCompile(`^[0-9A-F]{16}$`)
var buildIDPattern = regexp.MustCompile(`^[0-9A-F]{64}$`)
var slugPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)
var versionPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)
var buildTagPattern = regexp.MustCompile(`^build-[1-9][0-9]*$`)

const maxVersions = 128

func validType(s string) bool {
	switch s {
	case "graphics", "fps", "cheats", "translation", "gameplay":
		return true
	}
	return false
}

func validStatus(s string) bool {
	return s == "pending" || s == "approved" || s == "rejected"
}

func (s *store) adminHandler(token string) http.Handler {
	mux := http.NewServeMux()
	s.registerReads(mux, adminPrefix, true)
	write := func(route string, handler endpoint) {
		mux.Handle("PUT "+adminPrefix+route, endpoint(func(w http.ResponseWriter, r *http.Request) error {
			if err := handler(w, r); err != nil {
				return err
			}
			s.invalidateCatalog()
			return nil
		}))
	}
	write("/games/{title}", s.putGame)
	write("/mods/{id}", s.putMod)
	write("/mods/{id}/preview", s.putPreview)
	write("/mods/{id}/status", s.putStatus)
	write("/mods/{id}/versions/{version}", s.putVersion)
	write("/mods/{id}/versions/{version}/status", s.putStatus)
	write("/gamebanana", s.putGameBanana)
	write("/gamebanana/drop-demos", s.dropDemos)
	write("/github", s.putGitHub)
	want := sha256.Sum256([]byte("Bearer " + token))
	writer := make(chan struct{}, 1)
	return responseHeaders(endpoint(func(w http.ResponseWriter, r *http.Request) error {
		got := sha256.Sum256([]byte(r.Header.Get("Authorization")))
		if subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			return problem{401, "unauthorized", "bearer token required"}
		}
		if r.Method == http.MethodPut {
			select {
			case writer <- struct{}{}:
				defer func() { <-writer }()
			default:
				return problem{429, "writer_busy", "another admin write is in progress"}
			}
		}
		mux.ServeHTTP(w, r)
		return nil
	}))
}

func mediaType(r *http.Request) string {
	value, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	return value
}

func requestBody(w http.ResponseWriter, r *http.Request, maxBytes int64) (io.Reader, error) {
	if r.ContentLength > maxBytes {
		return nil, problem{413, "body_too_large", "request body exceeds the byte limit"}
	}
	return http.MaxBytesReader(w, r.Body, maxBytes), nil
}

func bodyError(err error) error {
	var limit *http.MaxBytesError
	if errors.As(err, &limit) {
		return problem{413, "body_too_large", "request body exceeds the byte limit"}
	}
	return invalid("could not read request body")
}

func jsonBody(w http.ResponseWriter, r *http.Request, value any) error {
	return jsonBodyLimit(w, r, value, 24<<10)
}

func jsonBodyLimit(w http.ResponseWriter, r *http.Request, value any, limit int64) error {
	if mediaType(r) != "application/json" {
		return problem{415, "unsupported_media_type", "expected application/json"}
	}
	body, err := requestBody(w, r, limit)
	if err != nil {
		return err
	}
	raw, err := io.ReadAll(body)
	if err != nil {
		return bodyError(err)
	}
	if !utf8.Valid(raw) || !bytes.HasPrefix(bytes.TrimSpace(raw), []byte("{")) {
		return invalid("expected a JSON object")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(value) != nil || decoder.Decode(new(any)) != io.EOF {
		return invalid("expected one JSON object with documented fields")
	}
	return nil
}

func (s *store) putGame(w http.ResponseWriter, r *http.Request) error {
	id := strings.ToUpper(r.PathValue("title"))
	var input struct {
		Name string `json:"name"`
	}
	if err := jsonBody(w, r, &input); err != nil {
		return err
	}
	input.Name = strings.TrimSpace(input.Name)
	if !titlePattern.MatchString(id) || !validText(input.Name, 120, false) {
		return invalid("expected a 16-digit hexadecimal title ID and a game name")
	}
	if _, err := s.db.ExecContext(r.Context(), `INSERT INTO games (title_id, name) VALUES (?,?)
	    ON CONFLICT(title_id) DO UPDATE SET name=excluded.name`, id, input.Name); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (s *store) putMod(w http.ResponseWriter, r *http.Request) error {
	id := r.PathValue("id")
	var input struct {
		Name        string `json:"name"`
		Summary     string `json:"summary"`
		Description string `json:"description"`
		Author      string `json:"author"`
		Type        string `json:"type"`
	}
	if err := jsonBody(w, r, &input); err != nil {
		return err
	}
	input.Name = strings.TrimSpace(input.Name)
	input.Summary = strings.TrimSpace(input.Summary)
	input.Description = strings.TrimSpace(input.Description)
	input.Author = strings.TrimSpace(input.Author)
	if !slugPattern.MatchString(id) || !portableName(id) || gameBananaIDPattern.MatchString(id) || !validType(input.Type) ||
		!validText(input.Name, 120, false) || !validText(input.Summary, 240, false) ||
		!validText(input.Description, 4000, true) || !validText(input.Author, 100, false) {
		return invalid("invalid mod ID, metadata or type")
	}
	if err := s.rejectGitHubWrite(r.Context(), id); err != nil {
		return err
	}
	if _, err := s.db.ExecContext(r.Context(), `INSERT INTO mods
	    (id,name,summary,description,author,type,demo,status,updated_at) VALUES (?,?,?,?,?,?,0,'pending',?)
	    ON CONFLICT(id) DO UPDATE SET name=excluded.name, summary=excluded.summary, description=excluded.description,
	    author=excluded.author, type=excluded.type, status='pending', updated_at=excluded.updated_at
	    WHERE mods.name<>excluded.name OR mods.summary<>excluded.summary OR mods.description<>excluded.description
	       OR mods.author<>excluded.author OR mods.type<>excluded.type`,
		id, input.Name, input.Summary, input.Description, input.Author, input.Type, time.Now().UnixMilli()); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (s *store) putPreview(w http.ResponseWriter, r *http.Request) error {
	id := r.PathValue("id")
	if gameBananaIDPattern.MatchString(id) {
		return invalid("GameBanana previews come from the import manifest")
	}
	if err := s.rejectGitHubWrite(r.Context(), id); err != nil {
		return err
	}
	var modID string
	if err := s.db.QueryRowContext(r.Context(), "SELECT id FROM mods WHERE id=?", id).Scan(&modID); err != nil {
		return err
	}
	media := mediaType(r)
	if media != "image/png" && media != "image/jpeg" {
		return problem{415, "unsupported_media_type", "expected image/png or image/jpeg"}
	}
	body, err := requestBody(w, r, maxPreviewBytes)
	if err != nil {
		return err
	}
	content, err := io.ReadAll(body)
	if err != nil {
		return bodyError(err)
	}
	actual, err := previewType(content)
	if err != nil {
		return err
	}
	if actual != media {
		return invalid("preview does not match its Content-Type")
	}
	image, err := s.putAsset(func(w io.Writer) error {
		_, err := w.Write(content)
		return err
	})
	if err != nil {
		return err
	}
	if _, err := s.db.ExecContext(r.Context(), `UPDATE mods SET preview_sha=?, preview_type=?, status='pending', updated_at=?
	    WHERE id=? AND preview_sha IS NOT ?`, image.digest, media, time.Now().UnixMilli(), id, image.digest); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (s *store) putStatus(w http.ResponseWriter, r *http.Request) error {
	id, ver := r.PathValue("id"), r.PathValue("version")
	var input struct {
		Status string `json:"status"`
	}
	if err := jsonBody(w, r, &input); err != nil {
		return err
	}
	if !validStatus(input.Status) {
		return invalid("status must be pending, approved or rejected")
	}
	var err error
	if ver != "" {
		err = s.changeVersionStatus(r.Context(), id, ver, input.Status)
	} else {
		err = expectChanged(s.db.ExecContext(r.Context(), "UPDATE mods SET status=?, updated_at=? WHERE id=?",
			input.Status, time.Now().UnixMilli(), id))
	}
	if err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (s *store) putVersion(w http.ResponseWriter, r *http.Request) error {
	id, ver := r.PathValue("id"), r.PathValue("version")
	if gameBananaIDPattern.MatchString(id) {
		return invalid("GameBanana files cannot be uploaded")
	}
	if err := s.rejectGitHubWrite(r.Context(), id); err != nil {
		return err
	}
	if !versionPattern.MatchString(ver) || !portableName(ver) {
		return invalid("invalid version ID")
	}
	if mediaType(r) != "application/zip" {
		return problem{415, "unsupported_media_type", "expected application/zip"}
	}
	q, err := params(r, "title_id", "game_version", "build_id", "min_lichii_build")
	if err != nil {
		return err
	}
	title := strings.ToUpper(q.Get("title_id"))
	gameVersion := q.Get("game_version")
	buildID := strings.ToUpper(q.Get("build_id"))
	minBuild := q.Get("min_lichii_build")
	minimum, numberErr := strconv.ParseUint(strings.TrimPrefix(minBuild, "build-"), 10, 64)
	if !titlePattern.MatchString(title) || !validText(gameVersion, 32, false) ||
		gameVersion != strings.TrimSpace(gameVersion) || gameVersion == demoGameVersion ||
		!buildIDPattern.MatchString(buildID) || buildID == demoBuildID ||
		!buildTagPattern.MatchString(minBuild) || numberErr != nil || minimum == 0 {
		return invalid("expected exact title_id, game_version, build_id and min_lichii_build")
	}
	ctx := r.Context()
	var demo, gameExists, versionExists bool
	var versionCount int
	err = s.db.QueryRowContext(ctx, `SELECT m.demo, EXISTS(SELECT 1 FROM games WHERE title_id=?),
	    EXISTS(SELECT 1 FROM versions WHERE mod_id=m.id AND id=?),
	    (SELECT count(*) FROM versions WHERE mod_id=m.id) FROM mods m WHERE m.id=?`, title, ver, id).
		Scan(&demo, &gameExists, &versionExists, &versionCount)
	if err != nil {
		return err
	}
	if demo {
		return problem{409, "demo_readonly", "demo artifacts are generated by seed-demo"}
	}
	if !gameExists {
		return invalid("register the game before uploading a version")
	}
	if versionExists || versionCount >= maxVersions {
		return problem{409, "version_conflict", "the version exists or the mod has the maximum number of versions"}
	}
	body, err := requestBody(w, r, maxZIPBytes)
	if err != nil {
		return err
	}
	in, err := os.CreateTemp(s.assets, ".input-*")
	if err != nil {
		return err
	}
	defer os.Remove(in.Name())
	defer in.Close()
	n, err := io.Copy(in, body)
	if err != nil {
		return bodyError(err)
	}
	var unpacked int64
	archive, err := s.putAsset(func(w io.Writer) error {
		var normalizeErr error
		unpacked, normalizeErr = normalizeZIP(in, n, w)
		return normalizeErr
	})
	if err != nil {
		return err
	}
	v := version{ID: ver, TitleID: title, GameVersion: gameVersion, BuildID: buildID, MinLichiiBuild: minBuild,
		SHA256: archive.digest, SizeBytes: archive.size, UnpackedSizeBytes: unpacked, Status: "pending",
		DownloadURL: adminPrefix + "/mods/" + id + "/versions/" + ver + "/download"}
	if err := insertVersion(ctx, s.db, id, v, time.Now().UnixMilli()); err != nil {
		return err
	}
	reply(w, 201, v)
	return nil
}
