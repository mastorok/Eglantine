package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	maxZIPBytes      = 64 << 20
	maxUnpackedBytes = 256 << 20
	maxZIPEntries    = 4096
	maxPreviewBytes  = 2 << 20
	maxPreviewSide   = 2048
)

type asset struct {
	digest string
	size   int64
}

func (s *store) putAsset(write func(io.Writer) error) (asset, error) {
	file, err := os.CreateTemp(s.assets, ".upload-*")
	if err != nil {
		return asset{}, err
	}
	defer os.Remove(file.Name())
	defer file.Close()
	hash := sha256.New()
	if err := write(io.MultiWriter(file, hash)); err != nil {
		return asset{}, err
	}
	info, err := file.Stat()
	if err != nil {
		return asset{}, err
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	if err := file.Sync(); err != nil {
		return asset{}, err
	}
	if err := file.Close(); err != nil {
		return asset{}, err
	}
	if err := os.Rename(file.Name(), filepath.Join(s.assets, digest)); err != nil {
		return asset{}, err
	}
	dir, err := os.Open(s.assets)
	if err != nil {
		return asset{}, err
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return asset{}, err
	}
	return asset{digest, info.Size()}, nil
}

func previewType(content []byte) (string, error) {
	info, format, err := image.DecodeConfig(bytes.NewReader(content))
	if err != nil || (format != "png" && format != "jpeg") {
		return "", invalid("preview must be a PNG or JPEG image")
	}
	if info.Width < 1 || info.Height < 1 || info.Width > maxPreviewSide || info.Height > maxPreviewSide {
		return "", invalid("preview dimensions must be between 1 and 2048 pixels")
	}
	if _, _, err := image.Decode(bytes.NewReader(content)); err != nil {
		return "", invalid("invalid preview image")
	}
	return "image/" + format, nil
}

var pathComponent = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$`)

func portableName(name string) bool {
	if !pathComponent.MatchString(name) || strings.HasSuffix(name, ".") || strings.HasSuffix(name, " ") {
		return false
	}
	base, _, _ := strings.Cut(name, ".")
	base = strings.ToUpper(strings.TrimRight(base, " "))
	switch base {
	case "CON", "PRN", "AUX", "NUL":
		return false
	}
	return !(len(base) == 4 && (base[:3] == "COM" || base[:3] == "LPT") && base[3] >= '0' && base[3] <= '9')
}

var errInvalidZIP = problem{400, "invalid_archive", "expected a safe ZIP with a file beneath a recognized mod root"}
var errLargeZIP = problem{413, "archive_too_large", "ZIP exceeds the archive size or entry limits"}

func zipEntryCount(src io.ReaderAt, size int64) (int, error) {
	const footerSize, entrySize = 22, 46
	var b [footerSize]byte
	if size < footerSize {
		return 0, errInvalidZIP
	}
	end := size - footerSize
	if _, err := src.ReadAt(b[:], end); err != nil || !bytes.Equal(b[:4], []byte("PK\x05\x06")) ||
		binary.LittleEndian.Uint16(b[20:22]) != 0 {
		return 0, errInvalidZIP
	}
	count := int(binary.LittleEndian.Uint16(b[10:12]))
	length := int64(binary.LittleEndian.Uint32(b[12:16]))
	start := int64(binary.LittleEndian.Uint32(b[16:20]))
	if count > maxZIPEntries || length > 4<<20 {
		return 0, errLargeZIP
	}
	if binary.LittleEndian.Uint32(b[4:8]) != 0 || int(binary.LittleEndian.Uint16(b[8:10])) != count ||
		start+length != end || count == 0 {
		return 0, errInvalidZIP
	}
	remaining := count
	var entry [entrySize]byte
	for pos := start; pos < end; {
		if remaining == 0 || end-pos < entrySize {
			return 0, errInvalidZIP
		}
		if _, err := src.ReadAt(entry[:], pos); err != nil || !bytes.Equal(entry[:4], []byte("PK\x01\x02")) {
			return 0, errInvalidZIP
		}
		remaining--
		pos += entrySize + int64(binary.LittleEndian.Uint16(entry[28:30])) +
			int64(binary.LittleEndian.Uint16(entry[30:32])) + int64(binary.LittleEndian.Uint16(entry[32:34]))
		if pos > end {
			return 0, errInvalidZIP
		}
	}
	if remaining != 0 {
		return 0, errInvalidZIP
	}
	return count, nil
}

type cappedWriter struct {
	dst       io.Writer
	remaining int64
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, errLargeZIP
	}
	n, err := w.dst.Write(p)
	w.remaining -= int64(n)
	return n, err
}

type archiveNode struct {
	name     string
	dir      bool
	explicit bool
}

func normalizeZIP(src io.ReaderAt, size int64, dst io.Writer) (int64, error) {
	count, err := zipEntryCount(src, size)
	if err != nil {
		return 0, err
	}
	z, err := zip.NewReader(src, size)
	if err != nil || len(z.File) != count {
		return 0, errInvalidZIP
	}
	out := zip.NewWriter(&cappedWriter{dst, maxZIPBytes})
	seen := make(map[string]archiveNode)
	var unpacked int64
	hasModFile := false
	for _, f := range z.File {
		mode := f.Mode()
		isDir := mode.IsDir()
		name := strings.TrimSuffix(f.Name, "/")
		parts := strings.Split(name, "/")
		if len(name) > 220 || len(parts) > 16 || isDir != strings.HasSuffix(f.Name, "/") ||
			(mode.Type() != 0 && mode.Type() != os.ModeDir) || f.Flags&1 != 0 ||
			(f.Method != zip.Store && f.Method != zip.Deflate) {
			return 0, errInvalidZIP
		}
		switch parts[0] {
		case "exefs", "romfs", "romfs_ext", "romfslite", "cheats":
			if !isDir {
				if len(parts) < 2 {
					return 0, errInvalidZIP
				}
				hasModFile = true
			}
		default:
			base, ext, _ := strings.Cut(strings.ToLower(parts[0]), ".")
			if len(parts) != 1 || isDir || (base != "readme" && base != "license" && base != "licence") ||
				(ext != "" && ext != "txt" && ext != "md" && ext != "rst" && ext != "markdown") {
				return 0, errInvalidZIP
			}
		}
		for i, part := range parts {
			if !portableName(part) {
				return 0, errInvalidZIP
			}
			prefix := strings.Join(parts[:i+1], "/")
			key := strings.ToLower(prefix)
			last := i == len(parts)-1
			dir := isDir || !last
			old, exists := seen[key]
			if exists && (old.name != prefix || old.dir != dir || (last && old.explicit)) {
				return 0, errInvalidZIP
			}
			seen[key] = archiveNode{prefix, dir, last || old.explicit}
		}
		if f.UncompressedSize64 > uint64(maxUnpackedBytes-unpacked) {
			return 0, errLargeZIP
		}
		if isDir {
			if f.UncompressedSize64 != 0 {
				return 0, errInvalidZIP
			}
			if _, err := out.CreateHeader(cleanZIPHeader(f.Name, true)); err != nil {
				return 0, err
			}
			continue
		}
		entry, err := f.Open()
		if err != nil {
			return 0, errInvalidZIP
		}
		writer, err := out.CreateHeader(cleanZIPHeader(f.Name, false))
		if err != nil {
			entry.Close()
			return 0, err
		}
		n, err := io.Copy(writer, io.LimitReader(entry, int64(f.UncompressedSize64)+1))
		entry.Close()
		switch {
		case errors.Is(err, errLargeZIP):
			return 0, errLargeZIP
		case err != nil || uint64(n) != f.UncompressedSize64:
			return 0, errInvalidZIP
		}
		unpacked += n
	}
	if !hasModFile {
		return 0, errInvalidZIP
	}
	if err := out.Close(); err != nil {
		return 0, err
	}
	return unpacked, nil
}

func cleanZIPHeader(name string, dir bool) *zip.FileHeader {
	h := &zip.FileHeader{Name: name, Method: zip.Deflate}
	h.SetMode(0644)
	if dir {
		h.Method = zip.Store
		h.SetMode(os.ModeDir | 0755)
	}
	return h
}
