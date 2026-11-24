package main

import (
	"bytes"
	"compress/gzip"
	"container/list"
	"crypto/sha256"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	catalogCacheBytes   = 8 << 20
	catalogCacheEntries = 256
	catalogMaxBody      = 1 << 20
	catalogMaxKey       = 4 << 10
	catalogMaxAge       = 300 * time.Second
	catalogCacheControl = "public, max-age=300"
)

type catalogResponseKey struct {
	path, query string
	github      bool
}

type catalogRepresentation struct {
	body []byte
	etag string
}

type catalogResponse struct {
	header         http.Header
	identity, gzip catalogRepresentation
}

type catalogCacheEntry struct {
	key      catalogResponseKey
	response *catalogResponse
	size     int
	expires  time.Time
}

type catalogResponses struct {
	mu         sync.Mutex
	generation chan struct{}
	pending    map[catalogResponseKey]chan struct{}
	entries    map[catalogResponseKey]*list.Element
	order      list.List
	size       int
}

var catalogFillSlots = make(chan struct{}, 8)
var catalogDetailSlots = make(chan struct{}, 4)

var catalogCompressors = func() chan *gzip.Writer {
	pool := make(chan *gzip.Writer, 2)
	for range cap(pool) {
		pool <- nil
	}
	return pool
}()

func (c *catalogResponses) invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.generation != nil {
		close(c.generation)
		c.generation = nil
	}
	c.pending = nil
	c.entries = nil
	c.order.Init()
	c.size = 0
}

func (c *catalogResponses) wrap(next http.Handler) http.Handler {
	return c.wrapWithSlots(next, catalogFillSlots)
}

func (c *catalogResponses) wrapWithSlots(next http.Handler, slots chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			next.ServeHTTP(w, r)
			return
		}
		key := catalogResponseKey{r.URL.EscapedPath(), r.URL.RawQuery, githubCatalog(r)}
		for r.Context().Err() == nil {
			c.mu.Lock()
			if element := c.entries[key]; element != nil {
				entry := element.Value.(catalogCacheEntry)
				if time.Now().Before(entry.expires) {
					c.order.MoveToBack(element)
					c.mu.Unlock()
					entry.response.serve(w, r)
					return
				}
				c.remove(element)
			}
			if c.generation == nil {
				c.generation = make(chan struct{})
				c.pending = make(map[catalogResponseKey]chan struct{})
				c.entries = make(map[catalogResponseKey]*list.Element)
			}
			generation := c.generation
			if done := c.pending[key]; done != nil {
				c.mu.Unlock()
				select {
				case <-done:
				case <-generation:
				case <-r.Context().Done():
				}
				continue
			}
			done := make(chan struct{})
			c.pending[key] = done
			c.mu.Unlock()

			response, current := c.fill(w, r, next, key, generation, done, slots)
			if response == nil || r.Context().Err() != nil {
				return
			}
			if current {
				response.serve(w, r)
				return
			}
		}
	})
}

func (c *catalogResponses) fill(w http.ResponseWriter, r *http.Request, next http.Handler, key catalogResponseKey, generation, done, slots chan struct{}) (response *catalogResponse, current bool) {
	defer func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		defer close(done)
		if c.generation == generation {
			current = true
			delete(c.pending, key)
			if response != nil && r.Context().Err() == nil {
				c.insert(key, response)
			}
		}
	}()
	return captureCatalogResponse(w, r, next, slots), false
}

func (c *catalogResponses) insert(key catalogResponseKey, response *catalogResponse) {
	size := len(key.path) + len(key.query) + cap(response.identity.body) + cap(response.gzip.body)
	if len(key.path)+len(key.query) > catalogMaxKey {
		return
	}
	for name, values := range response.header {
		size += len(name) + 64
		for _, value := range values {
			size += len(value) + 16
		}
	}
	if size > catalogCacheBytes {
		return
	}
	for c.size+size > catalogCacheBytes || len(c.entries) >= catalogCacheEntries {
		c.remove(c.order.Front())
	}
	key.path = strings.Clone(key.path)
	key.query = strings.Clone(key.query)
	c.entries[key] = c.order.PushBack(catalogCacheEntry{key, response, size, time.Now().Add(catalogMaxAge)})
	c.size += size
}

func (c *catalogResponses) remove(element *list.Element) {
	entry := c.order.Remove(element).(catalogCacheEntry)
	delete(c.entries, entry.key)
	c.size -= entry.size
}

type catalogCapture struct {
	dst     http.ResponseWriter
	header  http.Header
	written http.Header
	status  int
	body    []byte
	head    bool
	sent    bool
}

func (w *catalogCapture) Header() http.Header { return w.header }

func (w *catalogCapture) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.written = w.header.Clone()
	if status != http.StatusOK {
		w.send()
	}
}

func (w *catalogCapture) Write(p []byte) (int, error) {
	if w.status == 0 {
		if w.header.Get("Content-Type") == "" {
			w.header.Set("Content-Type", http.DetectContentType(p))
		}
		w.WriteHeader(http.StatusOK)
	}
	if !w.sent && len(p) <= catalogMaxBody-len(w.body) {
		w.body = append(w.body, p...)
		return len(p), nil
	}
	if !w.sent {
		w.send()
	}
	if w.head {
		return len(p), nil
	}
	return w.dst.Write(p)
}

func (w *catalogCapture) send() {
	w.sent = true
	copyCatalogHeaders(w.dst.Header(), w.written)
	w.dst.Header().Set("Cache-Control", "no-store")
	w.dst.WriteHeader(w.status)
	if !w.head && len(w.body) != 0 {
		_, _ = w.dst.Write(w.body)
	}
	w.body = nil
}

func captureCatalogResponse(w http.ResponseWriter, r *http.Request, next http.Handler, slots chan struct{}) *catalogResponse {
	select {
	case slots <- struct{}{}:
		defer func() { <-slots }()
	case <-r.Context().Done():
		return nil
	}
	capture := &catalogCapture{dst: w, header: make(http.Header), head: r.Method == http.MethodHead}
	if capture.head {
		r = r.WithContext(r.Context())
		r.Method = http.MethodGet
	}
	next.ServeHTTP(capture, r)
	if capture.sent || r.Context().Err() != nil {
		return nil
	}
	if capture.status == 0 {
		capture.WriteHeader(http.StatusOK)
	}
	contentType, _, _ := mime.ParseMediaType(capture.written.Get("Content-Type"))
	if contentType != "application/json" || capture.written.Get("Content-Encoding") != "" ||
		capture.written.Get("Cache-Control") != "" || len(capture.written.Values("Set-Cookie")) != 0 {
		capture.send()
		return nil
	}
	for _, value := range capture.written.Values("Vary") {
		for name := range strings.SplitSeq(value, ",") {
			name = strings.TrimSpace(name)
			if name != "" && !strings.EqualFold(name, githubFeatureHeader) && !strings.EqualFold(name, "Accept-Encoding") {
				capture.send()
				return nil
			}
		}
	}
	capture.written.Del("Vary")
	var compressed bytes.Buffer
	var zw *gzip.Writer
	select {
	case zw = <-catalogCompressors:
	case <-r.Context().Done():
		return nil
	}
	if zw == nil {
		zw, _ = gzip.NewWriterLevel(&compressed, gzip.BestSpeed)
	} else {
		zw.Reset(&compressed)
	}
	_, _ = zw.Write(capture.body)
	_ = zw.Close()
	zw.Reset(io.Discard)
	catalogCompressors <- zw
	return &catalogResponse{
		header:   capture.written,
		identity: catalogRepresentation{capture.body, fmt.Sprintf("\"%x\"", sha256.Sum256(capture.body))},
		gzip:     catalogRepresentation{compressed.Bytes(), fmt.Sprintf("\"%x\"", sha256.Sum256(compressed.Bytes()))},
	}
}

func copyCatalogHeaders(dst, src http.Header) {
	for name, values := range src {
		if name == "Vary" {
			dst[name] = append(dst[name], values...)
		} else {
			dst[name] = append([]string(nil), values...)
		}
	}
}

func addCatalogVary(header http.Header, name string) {
	for _, value := range header.Values("Vary") {
		for vary := range strings.SplitSeq(value, ",") {
			if strings.EqualFold(strings.TrimSpace(vary), name) {
				return
			}
		}
	}
	header.Add("Vary", name)
}

func (response *catalogResponse) serve(w http.ResponseWriter, r *http.Request) {
	useGzip, acceptable := catalogGzip(r.Header.Values("Accept-Encoding"))
	h := w.Header()
	addCatalogVary(h, githubFeatureHeader)
	addCatalogVary(h, "Accept-Encoding")
	h.Set("Vary", strings.Join(h.Values("Vary"), ", "))
	if !acceptable {
		h.Set("Cache-Control", "no-store")
		if r.Method == http.MethodHead {
			h.Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotAcceptable)
			return
		}
		reply(w, http.StatusNotAcceptable, map[string]any{"error": map[string]string{
			"code": "not_acceptable", "message": "catalog requires gzip or identity encoding",
		}})
		return
	}
	representation := response.identity
	if useGzip {
		representation = response.gzip
	}
	if tags := r.Header.Values("If-Match"); len(tags) != 0 && !catalogTagMatches(tags, representation.etag, false) {
		h.Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusPreconditionFailed)
		return
	}
	copyCatalogHeaders(h, response.header)
	h.Set("ETag", representation.etag)
	h.Set("Cache-Control", catalogCacheControl)
	h.Set("Cloudflare-CDN-Cache-Control", "no-store")
	if catalogTagMatches(r.Header.Values("If-None-Match"), representation.etag, true) {
		h.Del("Content-Type")
		h.Del("Content-Encoding")
		h.Del("Content-Length")
		w.WriteHeader(http.StatusNotModified)
		return
	}
	if useGzip {
		h.Set("Content-Encoding", "gzip")
	}
	h.Set("Content-Length", strconv.Itoa(len(representation.body)))
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		_, _ = w.Write(representation.body)
	}
}

func catalogGzip(values []string) (useGzip, acceptable bool) {
	gzipQ, starQ, identityQ := -1.0, -1.0, -1.0
	for _, value := range values {
		for item := range strings.SplitSeq(value, ",") {
			name, param, hasParam := strings.Cut(item, ";")
			q := 1.0
			if hasParam {
				key, value, _ := strings.Cut(strings.TrimSpace(param), "=")
				value = strings.TrimSpace(value)
				var err error
				q, err = strconv.ParseFloat(value, 64)
				if !strings.EqualFold(strings.TrimSpace(key), "q") || err != nil || !(q >= 0 && q <= 1) {
					q = 0
				}
			}
			switch strings.ToLower(strings.TrimSpace(name)) {
			case "gzip", "x-gzip":
				gzipQ = max(gzipQ, q)
			case "identity":
				identityQ = q
			case "*":
				starQ = q
			}
		}
	}
	if gzipQ < 0 {
		gzipQ = starQ
	}
	if identityQ < 0 && starQ == 0 {
		identityQ = 0
	}
	return gzipQ > 0 && gzipQ >= identityQ, gzipQ > 0 || identityQ != 0
}

func catalogTagMatches(values []string, etag string, weak bool) bool {
	for _, value := range values {
		for value = strings.TrimLeft(value, ", \t"); value != ""; value = strings.TrimLeft(value, ", \t") {
			if rest, star := strings.CutPrefix(value, "*"); star {
				rest = strings.TrimLeft(rest, " \t")
				if rest == "" || rest[0] == ',' {
					return true
				}
				break
			}
			tag, isWeak := strings.CutPrefix(value, "W/")
			if !strings.HasPrefix(tag, "\"") {
				break
			}
			end := strings.IndexByte(tag[1:], '"')
			if end < 0 {
				break
			}
			end += 2
			value = strings.TrimLeft(tag[end:], " \t")
			if value != "" && value[0] != ',' {
				break
			}
			if (!isWeak || weak) && tag[:end] == etag {
				return true
			}
		}
	}
	return false
}
