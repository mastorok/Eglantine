package main

import (
	"context"
	"database/sql"
	"slices"
	"strings"
)

type catalogView struct {
	legacy, github *catalogData
}

type catalogRow struct {
	mod
	name, summary, author, description string
}

type catalogData struct {
	games          []game
	updated, named []*catalogRow
	updatedByTitle map[string][]*catalogRow
	namedByTitle   map[string][]*catalogRow
}

func (v *catalogView) data(github bool) *catalogData {
	if github {
		return v.github
	}
	return v.legacy
}

func (s *store) invalidateCatalog() {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	s.view.Store(nil)
	s.responses.invalidate()
}

func (s *store) catalog(ctx context.Context) (*catalogView, error) {
	if view := s.view.Load(); view != nil {
		return view, nil
	}
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	if view := s.view.Load(); view != nil {
		return view, nil
	}
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	legacy, err := loadCatalog(ctx, tx, "catalog_targets")
	if err != nil {
		return nil, err
	}
	github, err := loadCatalog(ctx, tx, "catalog_targets_with_github")
	if err != nil {
		return nil, err
	}
	view := &catalogView{legacy: legacy, github: github}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	s.view.Store(view)
	return view, nil
}

func loadCatalog(ctx context.Context, tx *sql.Tx, targets string) (*catalogData, error) {
	rows, err := tx.QueryContext(ctx, "SELECT "+modColumns(false, targets)+", m.description"+modsFrom+
		" WHERE "+publicMod(targets)+" ORDER BY m.updated_at DESC, m.id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	c := &catalogData{games: []game{}, updatedByTitle: make(map[string][]*catalogRow),
		namedByTitle: make(map[string][]*catalogRow)}
	for rows.Next() {
		var description string
		m, err := scanMod(rows, false, &description)
		if err != nil {
			return nil, err
		}
		c.updated = append(c.updated, &catalogRow{m, asciiLower(m.Name), asciiLower(m.Summary),
			asciiLower(m.Author), asciiLower(description)})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	c.named = slices.Clone(c.updated)
	slices.SortFunc(c.named, func(a, b *catalogRow) int {
		if byName := strings.Compare(a.name, b.name); byName != 0 {
			return byName
		}
		return strings.Compare(a.ID, b.ID)
	})
	for _, m := range c.updated {
		for _, title := range m.TitleIDs {
			c.updatedByTitle[title] = append(c.updatedByTitle[title], m)
		}
	}
	for _, m := range c.named {
		for _, title := range m.TitleIDs {
			c.namedByTitle[title] = append(c.namedByTitle[title], m)
		}
	}
	games, err := tx.QueryContext(ctx, "SELECT title_id, name FROM games ORDER BY name COLLATE NOCASE, title_id")
	if err != nil {
		return nil, err
	}
	defer games.Close()
	for games.Next() {
		var g game
		if err := games.Scan(&g.TitleID, &g.Name); err != nil {
			return nil, err
		}
		if g.ModCount = len(c.updatedByTitle[g.TitleID]); g.ModCount != 0 {
			c.games = append(c.games, g)
		}
	}
	return c, games.Err()
}

func asciiLower(text string) string {
	for i := 0; i < len(text); i++ {
		if text[i] >= 'A' && text[i] <= 'Z' {
			folded := []byte(text)
			for j := i; j < len(folded); j++ {
				if folded[j] >= 'A' && folded[j] <= 'Z' {
					folded[j] += 'a' - 'A'
				}
			}
			return string(folded)
		}
	}
	return text
}

func (c *catalogData) mods(title, term, category string, byName bool, limit, offset int) page[mod] {
	rows, byTitle := c.updated, c.updatedByTitle
	if byName {
		rows, byTitle = c.named, c.namedByTitle
	}
	if title != "" {
		rows = byTitle[title]
	}
	result := page[mod]{Items: make([]mod, 0, min(limit+1, len(rows)))}
	term = asciiLower(term)
	skip := offset
	for _, row := range rows {
		if category != "" && category != row.Type {
			continue
		}
		if term != "" && !strings.Contains(row.name, term) && !strings.Contains(row.summary, term) &&
			!strings.Contains(row.author, term) && !strings.Contains(row.description, term) {
			continue
		}
		if skip > 0 {
			skip--
			continue
		}
		result.Items = append(result.Items, row.mod)
		if len(result.Items) > limit {
			break
		}
	}
	result.trim(limit, offset)
	return result
}
