package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"time"
)

var gameBananaIDPattern = regexp.MustCompile(`^gb-[0-9]+$`)
var gameBananaMemberURL = regexp.MustCompile(`^https://gamebanana\.com/members/[0-9]{1,19}$`)
var gameBananaPreviewURL = regexp.MustCompile(`^https://images\.gamebanana\.com/img/ss/mods/[A-Za-z0-9_.-]{1,240}\.(?i:jpg|jpeg|png|webp|gif)$`)
var archiveBuildID = regexp.MustCompile(`^([A-F0-9]{16}|[A-F0-9]{64})$`)
var fileSHA256 = regexp.MustCompile(`^[0-9a-f]{64}$`)

const gameBananaOrigin = "https://gamebanana.com"

type gameBananaCredit struct {
	Name string `json:"name"`
	Role string `json:"role"`
	URL  string `json:"url"`
}

type gameBananaFile struct {
	ID              int64           `json:"id"`
	Name            string          `json:"name"`
	SizeBytes       int64           `json:"size_bytes"`
	UploadedAt      time.Time       `json:"uploaded_at"`
	DownloadURL     string          `json:"download_url"`
	SHA256          *string         `json:"sha256"`
	BuildIDs        []string        `json:"build_ids"`
	AmbiguousLayout json.RawMessage `json:"ambiguous_layout,omitempty"`
}

type gameBananaMetadata struct {
	ID            int64              `json:"id"`
	Category      string             `json:"category"`
	Credits       []gameBananaCredit `json:"credits"`
	PreviewURLs   []string           `json:"preview_urls"`
	Files         []gameBananaFile   `json:"files"`
	Compatibility string             `json:"compatibility,omitempty"`
}

type gameBananaModInput struct {
	gameBananaMetadata
	Name        string    `json:"name"`
	Summary     string    `json:"summary"`
	Description string    `json:"description"`
	Author      string    `json:"author"`
	Type        string    `json:"type"`
	PageURL     string    `json:"page_url"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type gameBananaManifest struct {
	Games []struct {
		TitleID string               `json:"title_id"`
		Name    string               `json:"name"`
		Mods    []gameBananaModInput `json:"mods"`
	} `json:"games"`
}

func validateGameBananaMod(m gameBananaModInput) error {
	if m.ID <= 0 || m.PageURL != gameBananaOrigin+"/mods/"+strconv.FormatInt(m.ID, 10) ||
		m.UpdatedAt.IsZero() || !validType(m.Type) ||
		!validText(m.Name, 120, false) || !validText(m.Summary, 240, false) ||
		!validText(m.Description, 4000, true) || !validText(m.Author, 100, false) ||
		!validText(m.Category, 160, false) || m.Compatibility != "" ||
		len(m.Credits) < 1 || len(m.Credits) > 8 || len(m.PreviewURLs) > 8 ||
		len(m.Files) < 1 || len(m.Files) > 8 {
		return invalid(fmt.Sprintf("invalid GameBanana listing %d", m.ID))
	}
	for _, credit := range m.Credits {
		if !validText(credit.Name, 100, false) || !validText(credit.Role, 100, false) ||
			(credit.URL != "" && !gameBananaMemberURL.MatchString(credit.URL)) {
			return invalid(fmt.Sprintf("invalid GameBanana credit for listing %d", m.ID))
		}
	}
	for _, image := range m.PreviewURLs {
		if !gameBananaPreviewURL.MatchString(image) {
			return invalid(fmt.Sprintf("invalid GameBanana preview for listing %d", m.ID))
		}
	}
	seen := make(map[int64]bool, len(m.Files))
	for _, file := range m.Files {
		if file.ID <= 0 || seen[file.ID] || !validText(file.Name, 240, false) ||
			file.SizeBytes <= 0 || file.UploadedAt.IsZero() || len(file.BuildIDs) > 32 ||
			file.DownloadURL != gameBananaOrigin+"/dl/"+strconv.FormatInt(file.ID, 10) ||
			(file.SHA256 != nil && !fileSHA256.MatchString(*file.SHA256)) ||
			(file.SHA256 == nil && len(file.BuildIDs) != 0) {
			return invalid(fmt.Sprintf("invalid GameBanana file %d for listing %d", file.ID, m.ID))
		}
		switch string(file.AmbiguousLayout) {
		case "", "true", "false":
		default:
			return invalid(fmt.Sprintf("expected a boolean ambiguous_layout for GameBanana file %d", file.ID))
		}
		seen[file.ID] = true
		for _, build := range file.BuildIDs {
			if !archiveBuildID.MatchString(build) {
				return invalid(fmt.Sprintf("invalid build ID for GameBanana file %d", file.ID))
			}
		}
	}
	return nil
}

func (s *store) putGameBanana(w http.ResponseWriter, r *http.Request) error {
	var input gameBananaManifest
	if err := jsonBodyLimit(w, r, &input, 8<<20); err != nil {
		return err
	}
	if len(input.Games) < 1 || len(input.Games) > 48 {
		return invalid("expected between 1 and 48 GameBanana games")
	}
	titles := make(map[string]bool, len(input.Games))
	mods := make(map[int64]bool)
	for _, g := range input.Games {
		if !titlePattern.MatchString(g.TitleID) || titles[g.TitleID] || !validText(g.Name, 120, false) ||
			len(g.Mods) == 0 || len(g.Mods) > 20 {
			return invalid("invalid or duplicate GameBanana game, or expected 1..20 mods")
		}
		titles[g.TitleID] = true
		for _, m := range g.Mods {
			if mods[m.ID] {
				return invalid(fmt.Sprintf("duplicate GameBanana listing %d", m.ID))
			}
			mods[m.ID] = true
			if err := validateGameBananaMod(m); err != nil {
				return err
			}
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
			id := "gb-" + strconv.FormatInt(m.ID, 10)
			result, err := tx.ExecContext(ctx, `INSERT INTO mods
			    (id,name,summary,description,author,type,demo,status,updated_at) VALUES (?,?,?,?,?,?,0,'approved',?)
			    ON CONFLICT(id) DO UPDATE SET name=excluded.name, summary=excluded.summary, description=excluded.description,
			    author=excluded.author, type=excluded.type, updated_at=excluded.updated_at
			    WHERE EXISTS (SELECT 1 FROM gamebanana WHERE mod_id=mods.id AND title_id=?)`,
				id, m.Name, m.Summary, m.Description, m.Author, m.Type, m.UpdatedAt.UnixMilli(), g.TitleID)
			if err := expectChanged(result, err); errors.Is(err, sql.ErrNoRows) {
				return problem{409, "gamebanana_conflict", "GameBanana ID is already assigned to another title or source"}
			} else if err != nil {
				return err
			}
			var preview *string
			if len(m.PreviewURLs) != 0 {
				preview = &m.PreviewURLs[0]
			} else {
				m.PreviewURLs = []string{}
			}
			layouts := make(map[int64]json.RawMessage)
			var raw string
			err = tx.QueryRowContext(ctx, "SELECT metadata FROM gamebanana WHERE mod_id=?", id).Scan(&raw)
			if err == nil {
				var previous gameBananaMetadata
				if err := json.Unmarshal([]byte(raw), &previous); err != nil {
					return err
				}
				for _, file := range previous.Files {
					layouts[file.ID] = file.AmbiguousLayout
				}
			} else if !errors.Is(err, sql.ErrNoRows) {
				return err
			}
			for i := range m.Files {
				file := &m.Files[i]
				if file.AmbiguousLayout == nil {
					file.AmbiguousLayout = layouts[file.ID]
				}
				if file.BuildIDs == nil {
					file.BuildIDs = []string{}
				}
				file.UploadedAt = file.UploadedAt.UTC()
			}
			m.Compatibility = "unverified"
			metadata, err := json.Marshal(m.gameBananaMetadata)
			if err != nil {
				return err
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO gamebanana (mod_id,title_id,page_url,preview_url,metadata)
			    VALUES (?,?,?,?,?) ON CONFLICT(mod_id) DO UPDATE SET page_url=excluded.page_url,
			    preview_url=excluded.preview_url,metadata=excluded.metadata`, id, g.TitleID, m.PageURL, preview, string(metadata)); err != nil {
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

func (s *store) dropDemos(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var ready bool
	if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM gamebanana gb JOIN mods m ON m.id=gb.mod_id
	    WHERE m.status='approved')`).Scan(&ready); err != nil {
		return err
	}
	if !ready {
		return problem{409, "no_gamebanana_listings", "publish at least one GameBanana listing before dropping demos"}
	}
	if _, err := tx.ExecContext(ctx, "DELETE FROM versions WHERE mod_id IN (SELECT id FROM mods WHERE demo=1)"); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, "DELETE FROM mods WHERE demo=1"); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}
