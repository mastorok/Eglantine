package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf16"
)

var githubIDPattern = regexp.MustCompile(`^gh-[a-z0-9][a-z0-9-]{0,59}$`)
var githubRepositoryPattern = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9_.-]{1,100}$`)
var githubCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
var githubIPSBuildPattern = regexp.MustCompile(`^[0-9A-F]{16,64}$`)
var githubTimestampPattern = regexp.MustCompile(`^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$`)

const githubLicense = "NOASSERTION"

type githubOptional[T any] struct{ value *T }

func (o githubOptional[T]) IsZero() bool { return o.value == nil }

func (o githubOptional[T]) MarshalJSON() ([]byte, error) { return json.Marshal(o.value) }

func (o *githubOptional[T]) UnmarshalJSON(raw []byte) error {
	if bytes.Equal(raw, []byte("null")) {
		return errors.New("omit unused GitHub download fields instead of sending null")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	return decoder.Decode(&o.value)
}

type githubDownload struct {
	Path      string                 `json:"path"`
	SizeBytes int64                  `json:"size_bytes"`
	SHA256    string                 `json:"sha256"`
	Kind      string                 `json:"kind"`
	Member    githubOptional[string] `json:"member,omitzero"`
	Tag       githubOptional[string] `json:"tag,omitzero"`
}

type githubBuild struct {
	ID       string                         `json:"id"`
	Target   string                         `json:"target"`
	Download githubOptional[githubDownload] `json:"download,omitzero"`
}

type githubMetadata struct {
	Repository string        `json:"repository"`
	Commit     string        `json:"commit"`
	License    string        `json:"license"`
	Builds     []githubBuild `json:"builds"`
}

type githubTimestamp struct{ time.Time }

func (t *githubTimestamp) UnmarshalJSON(raw []byte) error {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	if !githubTimestampPattern.MatchString(value) {
		return errors.New("expected an RFC3339 timestamp")
	}
	parsed, err := time.Parse(time.RFC3339, value)
	t.Time = parsed
	return err
}

type githubModInput struct {
	githubMetadata
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Summary     string          `json:"summary"`
	Description string          `json:"description"`
	Author      string          `json:"author"`
	Type        string          `json:"type"`
	SourceURL   string          `json:"source_url"`
	UpdatedAt   githubTimestamp `json:"updated_at"`
}

type githubGameInput struct {
	TitleID string           `json:"title_id"`
	Name    string           `json:"name"`
	Mods    []githubModInput `json:"mods"`
}

type githubManifest struct {
	Games []githubGameInput `json:"games"`
}

func githubSourceURL(repository, commit string) string {
	return "https://github.com/" + repository + "/tree/" + commit
}

func validGitHubPath(value string) bool {
	if len(value) > 512 || !validText(value, 512, false) || strings.ContainsAny(value, `\:%?#`) {
		return false
	}
	for _, segment := range strings.Split(value, "/") {
		switch segment {
		case "", ".", "..":
			return false
		}
	}
	return true
}

func validGitHubDownload(build githubBuild, d githubDownload) bool {
	if !validGitHubPath(d.Path) || !fileSHA256.MatchString(d.SHA256) || d.SizeBytes <= 0 {
		return false
	}
	payload := d.Path
	extension := strings.ToLower(path.Ext(payload))
	limit := int64(2 << 20)
	if member := d.Member.value; member != nil {
		if extension != ".zip" || !validGitHubPath(*member) {
			return false
		}
		payload = *member
		extension = strings.ToLower(path.Ext(payload))
		limit = 64 << 20
	}
	if d.SizeBytes > limit {
		return false
	}
	if tag := d.Tag.value; tag != nil {
		if d.Kind != "cheat" || !validText(*tag, 160, false) || strings.ContainsAny(*tag, "[]{}") {
			return false
		}
		units := 0
		for _, r := range *tag {
			units += utf16.RuneLen(r)
			if units > 160 || r == unicode.ReplacementChar || unicode.Is(unicode.Cf, r) {
				return false
			}
		}
	}
	switch d.Kind {
	case "pchtxt":
		return extension == ".pchtxt" && len(build.ID) == 64
	case "ips":
		stem := strings.ToUpper(strings.SplitN(path.Base(payload), ".", 2)[0])
		return (extension == ".ips" || extension == ".ips32") &&
			githubIPSBuildPattern.MatchString(stem) && stem+strings.Repeat("0", 64-len(stem)) == build.ID
	case "cheat":
		return len(build.ID) == 16 && build.Target == "main" && (d.Member.value == nil || d.Tag.value != nil) &&
			strings.EqualFold(path.Base(payload), build.ID+".txt")
	default:
		return false
	}
}

func validateGitHubMod(m githubModInput) error {
	if !githubIDPattern.MatchString(m.ID) {
		return invalid("expected a GitHub mod ID starting with gh- and at most 63 characters")
	}
	if !validText(m.Name, 120, false) || !validText(m.Summary, 240, false) ||
		!validText(m.Description, 4000, true) || !validText(m.Author, 100, false) ||
		!validType(m.Type) || m.Type == "translation" || m.UpdatedAt.IsZero() {
		return invalid(fmt.Sprintf("invalid GitHub mod metadata for %s", m.ID))
	}
	_, repo, _ := strings.Cut(m.Repository, "/")
	if !githubRepositoryPattern.MatchString(m.Repository) || repo == "." || repo == ".." ||
		!githubCommitPattern.MatchString(m.Commit) || m.SourceURL != githubSourceURL(m.Repository, m.Commit) ||
		m.License != githubLicense {
		return invalid(fmt.Sprintf("expected a pinned GitHub repository URL and NOASSERTION license for %s", m.ID))
	}
	if m.Builds == nil || len(m.Builds) > 128 {
		return invalid(fmt.Sprintf("expected up to 128 GitHub builds for %s", m.ID))
	}
	seen := make(map[string]bool, len(m.Builds))
	for _, build := range m.Builds {
		if !archiveBuildID.MatchString(build.ID) || strings.Trim(build.ID, "0") == "" || seen[build.ID] ||
			(build.Target != "main" && build.Target != "unknown") {
			return invalid(fmt.Sprintf("invalid or duplicate GitHub build for %s", m.ID))
		}
		if d := build.Download.value; d != nil && !validGitHubDownload(build, *d) {
			return invalid(fmt.Sprintf("invalid GitHub download for %s build %s", m.ID, build.ID))
		}
		seen[build.ID] = true
	}
	return nil
}

func (s *store) putGitHub(w http.ResponseWriter, r *http.Request) error {
	var input githubManifest
	if err := jsonBodyLimit(w, r, &input, 8<<20); err != nil {
		return err
	}
	if input.Games == nil {
		return invalid("expected a GitHub games array")
	}
	titles := make(map[string]bool, len(input.Games))
	mods := make(map[string]bool)
	for _, g := range input.Games {
		if !titlePattern.MatchString(g.TitleID) || titles[g.TitleID] || !validText(g.Name, 120, false) || len(g.Mods) == 0 {
			return invalid("invalid or duplicate GitHub game, or missing mods")
		}
		titles[g.TitleID] = true
		for _, m := range g.Mods {
			if err := validateGitHubMod(m); err != nil {
				return err
			}
			if mods[m.ID] {
				return invalid(fmt.Sprintf("duplicate GitHub mod %s", m.ID))
			}
			mods[m.ID] = true
		}
	}
	ctx := r.Context()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, g := range input.Games {
		if _, err := tx.ExecContext(ctx, "INSERT INTO games (title_id,name) VALUES (?,?) ON CONFLICT DO NOTHING", g.TitleID, g.Name); err != nil {
			return err
		}
		for _, m := range g.Mods {
			result, err := tx.ExecContext(ctx, `INSERT INTO mods
			    (id,name,summary,description,author,type,demo,status,updated_at) VALUES (?,?,?,?,?,?,0,'approved',?)
			    ON CONFLICT(id) DO UPDATE SET name=excluded.name, summary=excluded.summary, description=excluded.description,
			    author=excluded.author, type=excluded.type, updated_at=excluded.updated_at
			    WHERE EXISTS (SELECT 1 FROM github WHERE mod_id=mods.id AND title_id=? AND repository=?)`,
				m.ID, m.Name, m.Summary, m.Description, m.Author, m.Type, m.UpdatedAt.UnixMilli(), g.TitleID, m.Repository)
			if err := expectChanged(result, err); errors.Is(err, sql.ErrNoRows) {
				return problem{409, "github_conflict", "GitHub mod ID is already assigned to another title or source"}
			} else if err != nil {
				return err
			}
			builds, err := json.Marshal(m.Builds)
			if err != nil {
				return err
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO github (mod_id,title_id,repository,commit_sha,license,builds)
			    VALUES (?,?,?,?,?,?) ON CONFLICT(mod_id) DO UPDATE SET repository=excluded.repository,
			    commit_sha=excluded.commit_sha, license=excluded.license, builds=excluded.builds`,
				m.ID, g.TitleID, m.Repository, m.Commit, m.License, string(builds)); err != nil {
				return err
			}
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	reply(w, 200, map[string]int{"games": len(titles), "mods": len(mods)})
	return nil
}

func (s *store) rejectGitHubWrite(ctx context.Context, id string) error {
	if !githubIDPattern.MatchString(id) {
		return nil
	}
	var exists bool
	if err := s.db.QueryRowContext(ctx, "SELECT EXISTS (SELECT 1 FROM github WHERE mod_id=?)", id).Scan(&exists); err != nil {
		return err
	}
	if exists {
		return problem{409, "github_readonly", "GitHub listings come from the import manifest; local assets are not supported"}
	}
	return nil
}
