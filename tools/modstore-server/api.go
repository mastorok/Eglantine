package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const maxPage = 100

type problem struct {
	status  int
	code    string
	message string
}

func (p problem) Error() string { return p.message }

func invalid(message string) error { return problem{400, "invalid_input", message} }

func reply(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

type endpoint func(http.ResponseWriter, *http.Request) error

func (handle endpoint) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := handle(w, r); err != nil {
		var p problem
		switch {
		case errors.As(err, &p):
		case errors.Is(err, sql.ErrNoRows):
			p = problem{404, "not_found", "not found"}
		default:
			log.Printf("request failed: %v", err)
			p = problem{500, "internal_error", "internal server error"}
		}
		reply(w, p.status, map[string]any{"error": map[string]string{"code": p.code, "message": p.message}})
	}
}

func responseHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Security-Policy", "sandbox; default-src 'none'")
		next.ServeHTTP(w, r)
	})
}

func (s *store) publicHandler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("GET /healthz", endpoint(func(w http.ResponseWriter, r *http.Request) error {
		if err := s.health(r.Context()); err != nil {
			return problem{503, "unavailable", "catalog unavailable"}
		}
		reply(w, 200, map[string]string{"status": "ok"})
		return nil
	}))
	mux.Handle("GET "+publicPrefix+"/games", s.responses.wrap(endpoint(s.listGames)))
	s.registerReads(mux, publicPrefix, false)
	return responseHeaders(mux)
}

func (s *store) registerReads(mux *http.ServeMux, prefix string, admin bool) {
	jsonRead := func(route string, handle endpoint, slots chan struct{}) {
		var handler http.Handler = handle
		if !admin {
			handler = s.responses.wrapWithSlots(handler, slots)
		}
		mux.Handle("GET "+prefix+route, handler)
	}
	assetRead := func(route string, handle endpoint) {
		mux.Handle("GET "+prefix+route, endpoint(func(w http.ResponseWriter, r *http.Request) error {
			if !admin {
				select {
				case s.assetTransfers <- struct{}{}:
					defer func() { <-s.assetTransfers }()
				default:
					return problem{503, "server_busy", "too many concurrent asset requests"}
				}
			}
			return handle(w, r)
		}))
	}
	jsonRead("/mods", func(w http.ResponseWriter, r *http.Request) error {
		return s.listMods(w, r, admin)
	}, catalogFillSlots)
	jsonRead("/mods/{id}", func(w http.ResponseWriter, r *http.Request) error {
		detail, err := s.detail(r.Context(), r.PathValue("id"), admin, catalogTargets(w, r, admin))
		if err != nil {
			return err
		}
		reply(w, 200, detail)
		return nil
	}, catalogDetailSlots)
	assetRead("/mods/{id}/preview", func(w http.ResponseWriter, r *http.Request) error {
		id := r.PathValue("id")
		query := `SELECT m.preview_sha, m.preview_type FROM mods m WHERE m.id=? AND m.preview_sha IS NOT NULL
		    AND NOT EXISTS (SELECT 1 FROM github WHERE mod_id=m.id)`
		if !admin {
			query += " AND " + publicMod("catalog_targets")
		}
		return s.serveAsset(w, r, s.db.QueryRowContext(r.Context(), query, id), "")
	})
	assetRead("/mods/{id}/versions/{version}/download", func(w http.ResponseWriter, r *http.Request) error {
		id, ver := r.PathValue("id"), r.PathValue("version")
		query := `SELECT v.sha256, 'application/zip' FROM versions v JOIN mods m ON m.id=v.mod_id
		    WHERE m.id=? AND v.id=? AND NOT EXISTS (SELECT 1 FROM github WHERE mod_id=m.id)`
		if !admin {
			query += " AND m.status='approved' AND v.status='approved'"
		}
		return s.serveAsset(w, r, s.db.QueryRowContext(r.Context(), query, id, ver), id+"-"+ver+".zip")
	})
}

const githubFeatureHeader = "X-Lichii-Catalog-Features"

func githubCatalog(r *http.Request) bool {
	values := r.Header.Values(githubFeatureHeader)
	return len(values) == 1 && values[0] == "github-v1"
}

func catalogTargets(w http.ResponseWriter, r *http.Request, admin bool) string {
	if admin {
		return "catalog_targets_with_github"
	}
	w.Header().Add("Vary", githubFeatureHeader)
	if githubCatalog(r) {
		return "catalog_targets_with_github"
	}
	return "catalog_targets"
}

func (s *store) serveAsset(w http.ResponseWriter, r *http.Request, row *sql.Row, name string) error {
	var digest, mediaType string
	if err := row.Scan(&digest, &mediaType); err != nil {
		return err
	}
	file, err := os.Open(filepath.Join(s.assets, digest))
	if err != nil {
		return err
	}
	defer file.Close()
	w.Header().Set("Content-Type", mediaType)
	if name != "" {
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
	}
	http.ServeContent(w, r, name, time.Time{}, file)
	return nil
}

type page[T any] struct {
	Items      []T
	NextOffset *int
}

func (p *page[T]) trim(limit, offset int) {
	if len(p.Items) > limit {
		p.Items = p.Items[:limit]
		next := offset + limit
		p.NextOffset = &next
	}
}

func params(r *http.Request, allowed ...string) (url.Values, error) {
	q, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		return nil, invalid("invalid query string")
	}
	for key, values := range q {
		valid := false
		for _, name := range allowed {
			valid = valid || name == key
		}
		if !valid || len(values) != 1 {
			return nil, invalid("unknown or repeated query parameter")
		}
	}
	return q, nil
}

func pageBounds(q url.Values) (int, int, error) {
	limit, offset := 50, 0
	if q.Has("limit") {
		n, err := strconv.Atoi(q.Get("limit"))
		if err != nil || n < 1 || n > maxPage {
			return 0, 0, invalid("limit must be 1..100")
		}
		limit = n
	}
	if q.Has("offset") {
		n, err := strconv.Atoi(q.Get("offset"))
		if err != nil || n < 0 || n > 1_000_000 {
			return 0, 0, invalid("offset must be 0..1000000")
		}
		offset = n
	}
	return limit, offset, nil
}

func (s *store) listGames(w http.ResponseWriter, r *http.Request) error {
	q, err := params(r, "limit", "offset")
	if err != nil {
		return err
	}
	limit, offset, err := pageBounds(q)
	if err != nil {
		return err
	}
	catalogTargets(w, r, false)
	view, err := s.catalog(r.Context())
	if err != nil {
		return err
	}
	games := view.data(githubCatalog(r)).games
	start := min(offset, len(games))
	result := page[game]{Items: games[start:min(start+limit+1, len(games))]}
	result.trim(limit, offset)
	reply(w, 200, map[string]any{"games": result.Items, "next_offset": result.NextOffset})
	return nil
}

func (s *store) listMods(w http.ResponseWriter, r *http.Request, admin bool) error {
	allowed := []string{"title_id", "q", "type", "sort", "limit", "offset"}
	if admin {
		allowed = append(allowed, "status")
	}
	q, err := params(r, allowed...)
	if err != nil {
		return err
	}
	limit, offset, err := pageBounds(q)
	if err != nil {
		return err
	}
	title, term, category := strings.ToUpper(q.Get("title_id")), strings.TrimSpace(q.Get("q")), q.Get("type")
	if q.Has("title_id") && !titlePattern.MatchString(title) {
		return invalid("title_id must contain 16 hexadecimal digits")
	}
	if term != "" && !validText(term, 128, false) {
		return invalid("q must be at most 128 characters without controls")
	}
	if q.Has("type") && !validType(category) {
		return invalid("invalid mod type")
	}
	sort := q.Get("sort")
	if sort != "" && sort != "updated" && sort != "name" {
		return invalid("sort must be updated or name")
	}
	targets := catalogTargets(w, r, admin)
	if !admin {
		view, err := s.catalog(r.Context())
		if err != nil {
			return err
		}
		result := view.data(githubCatalog(r)).mods(title, term, category, sort == "name", limit, offset)
		reply(w, 200, map[string]any{"mods": result.Items, "next_offset": result.NextOffset})
		return nil
	}
	where := []string{"1=1"}
	args := []any{}
	if title != "" {
		where = append(where, "EXISTS (SELECT 1 FROM "+targets+" t WHERE t.mod_id=m.id AND t.title_id=?)")
		args = append(args, title)
	}
	if term != "" {
		where = append(where, `(instr(lower(m.name), lower(?)) > 0 OR instr(lower(m.summary), lower(?)) > 0
		    OR instr(lower(m.author), lower(?)) > 0 OR instr(lower(m.description), lower(?)) > 0)`)
		args = append(args, term, term, term, term)
	}
	if category != "" {
		where = append(where, "m.type=?")
		args = append(args, category)
	}
	if status := q.Get("status"); q.Has("status") {
		if !validStatus(status) {
			return invalid("invalid moderation status")
		}
		where = append(where, "m.status=?")
		args = append(args, status)
	}
	order := "m.updated_at DESC, m.id"
	if sort == "name" {
		order = "m.name COLLATE NOCASE, m.id"
	}
	query := "SELECT " + modColumns(admin, targets) + modsFrom + " WHERE " + strings.Join(where, " AND ") +
		" ORDER BY " + order + " LIMIT ? OFFSET ?"
	rows, err := s.db.QueryContext(r.Context(), query, append(args, limit+1, offset)...)
	if err != nil {
		return err
	}
	defer rows.Close()
	result := page[mod]{Items: []mod{}}
	for rows.Next() {
		m, err := scanMod(rows, admin, nil)
		if err != nil {
			return err
		}
		result.Items = append(result.Items, m)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	result.trim(limit, offset)
	reply(w, 200, map[string]any{"mods": result.Items, "next_offset": result.NextOffset})
	return nil
}

func validText(s string, maxRunes int, multiline bool) bool {
	if !utf8.ValidString(s) || strings.TrimSpace(s) == "" || utf8.RuneCountInString(s) > maxRunes {
		return false
	}
	for _, r := range s {
		if unicode.IsControl(r) && !(multiline && (r == '\n' || r == '\t')) {
			return false
		}
	}
	return true
}
