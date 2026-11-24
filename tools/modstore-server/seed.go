package main

import (
	"archive/zip"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"strings"
)

const demoGameVersion = "modstore-demo"
const demoBuildID = "0000000000000000000000000000000000000000000000000000000000000000"
const demoPublishedAt = 1767225600000

type fixture struct {
	id, name, category string
	title, game        string
	tint               color.RGBA
}

var demos = []fixture{
	{"demo-bananza-graphics", "Color Study (demo)", "graphics", "040074A01BF12000", "Donkey Kong Bananza", color.RGBA{235, 143, 49, 255}},
	{"demo-kirby-fps", "Frame Budget (demo)", "fps", "040077201BE50000", "Kirby Air Riders", color.RGBA{212, 89, 159, 255}},
	{"demo-kart-cheats", "Sample Catalog Card (demo)", "cheats", "0400C3F00006E000", "Mario Kart World", color.RGBA{109, 145, 242, 255}},
	{"demo-kart-translation", "Language Card (demo)", "translation", "0400C3F00006E000", "Mario Kart World", color.RGBA{58, 190, 188, 255}},
	{"demo-kirby-gameplay", "Challenge Card (demo)", "gameplay", "040077201BE50000", "Kirby Air Riders", color.RGBA{181, 113, 230, 255}},
}

func (s *store) seedDemo(ctx context.Context) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, f := range demos {
		if _, err := tx.ExecContext(ctx, "INSERT INTO games (title_id,name) VALUES (?,?) ON CONFLICT DO NOTHING",
			f.title, f.game); err != nil {
			return err
		}
		preview, err := s.putAsset(func(w io.Writer) error { return writeDemoPreview(w, f.tint) })
		if err != nil {
			return err
		}
		const summary = "An inert sample for testing the Lichii Mod Store."
		const description = "This demo has a README and a uniquely named text marker. It contains no game assets, cheats or functional changes. The listed compatibility target is synthetic."
		result, err := tx.ExecContext(ctx, `INSERT INTO mods (id,name,summary,description,author,type,demo,preview_sha,preview_type,status,updated_at)
		    VALUES (?,?,?,?,?,?,1,?,'image/png','approved',?) ON CONFLICT DO NOTHING`,
			f.id, f.name, summary, description, "Lichii", f.category, preview.digest, demoPublishedAt)
		if err != nil {
			return err
		}
		added, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if added == 0 {
			var isDemo bool
			if err := tx.QueryRowContext(ctx, "SELECT demo FROM mods WHERE id=?", f.id).Scan(&isDemo); err != nil {
				return err
			}
			if !isDemo {
				return fmt.Errorf("demo ID is already in use: %s", f.id)
			}
			continue
		}
		var unpacked int64
		archive, err := s.putAsset(func(w io.Writer) error {
			var err error
			unpacked, err = writeDemoZIP(w, f.id)
			return err
		})
		if err != nil {
			return err
		}
		v := version{ID: "1.0.0-demo", TitleID: f.title, GameVersion: demoGameVersion, BuildID: demoBuildID,
			MinLichiiBuild: "build-1", SHA256: archive.digest, SizeBytes: archive.size,
			UnpackedSizeBytes: unpacked, Status: "approved"}
		if err := insertVersion(ctx, tx, f.id, v, demoPublishedAt); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.invalidateCatalog()
	return nil
}

func writeDemoZIP(w io.Writer, id string) (int64, error) {
	z := zip.NewWriter(w)
	var size int64
	for _, file := range []struct{ name, content string }{
		{"README.txt", "Lichii Mod Store demo. This is test data and does not modify game behavior.\n"},
		{"romfs/lichii-modstore-" + id + "-marker.txt", "Inert Lichii Mod Store test marker.\n"},
	} {
		h := cleanZIPHeader(file.name, false)
		h.Method = zip.Store
		dst, err := z.CreateHeader(h)
		if err != nil {
			return 0, err
		}
		n, err := io.Copy(dst, strings.NewReader(file.content))
		if err != nil {
			return 0, err
		}
		size += n
	}
	return size, z.Close()
}

func writeDemoPreview(w io.Writer, tint color.RGBA) error {
	const width, height = 640, 360
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			c := color.RGBA{20 + uint8(x/32), 24 + uint8(y/22), 42 + uint8(x/20), 255}
			switch {
			case (x-410)*(x-410)+(y-135)*(y-135) < 64*64:
				c = tint
			case y > 228+(x-300)/7 && y < 305+(x-300)/7:
				c = color.RGBA{tint.R / 3, tint.G / 3, tint.B / 3, 255}
			case ((x-130)*(x-130)+(y-112)*(y-112))/64%34 < 1:
				c = color.RGBA{45, 54, 81, 255}
			}
			img.SetRGBA(x, y, c)
		}
	}
	return png.Encode(w, img)
}
