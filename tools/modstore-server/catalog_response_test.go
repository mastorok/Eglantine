package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func catalogRequest(h http.Handler, method, path string, fields ...string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, nil)
	for i := 0; i < len(fields); i += 2 {
		r.Header.Add(fields[i], fields[i+1])
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestCatalogResponseHTTP(t *testing.T) {
	var cache catalogResponses
	calls := 0
	h := responseHeaders(cache.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Add("Vary", githubFeatureHeader)
		reply(w, http.StatusOK, map[string]string{"path": r.URL.EscapedPath(), "query": r.URL.RawQuery})
	})))
	const path = "/api/v1/mods?limit=2"
	plain := catalogRequest(h, "GET", path)
	zipped := catalogRequest(h, "GET", path, "Accept-Encoding", "br, GZip;q=0.5")
	tag := zipped.Header().Get("ETag")
	if plain.Code != 200 || zipped.Code != 200 || zipped.Header().Get("Content-Encoding") != "gzip" ||
		tag == "" || plain.Header().Get("ETag") == "" || tag == plain.Header().Get("ETag") {
		t.Fatalf("unexpected representations: identity %d %v, gzip %d %v", plain.Code, plain.Header(), zipped.Code, zipped.Header())
	}
	decoder, err := gzip.NewReader(zipped.Body)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := io.ReadAll(decoder)
	_ = decoder.Close()
	if err != nil || !bytes.Equal(decoded, plain.Body.Bytes()) {
		t.Fatalf("decoded gzip got %q, err %v; want %q", decoded, err, plain.Body.Bytes())
	}
	head := catalogRequest(h, "HEAD", path, "Accept-Encoding", "gzip")
	if head.Code != 200 || head.Body.Len() != 0 {
		t.Fatalf("HEAD got %d with %d bytes", head.Code, head.Body.Len())
	}
	for _, name := range []string{"Content-Type", "Content-Length", "Content-Encoding", "ETag"} {
		if head.Header().Get(name) != zipped.Header().Get(name) {
			t.Fatalf("HEAD %s = %q, want %q", name, head.Header().Get(name), zipped.Header().Get(name))
		}
	}
	notModified := catalogRequest(h, "GET", path, "Accept-Encoding", "gzip", "If-None-Match", `"has,comma"`, "If-None-Match", "W/"+tag)
	if notModified.Code != 304 || notModified.Body.Len() != 0 || notModified.Header().Get("ETag") != tag ||
		notModified.Header().Get("Content-Length") != "" {
		t.Fatalf("conditional response: %d %v (%d bytes)", notModified.Code, notModified.Header(), notModified.Body.Len())
	}
	for _, result := range []*httptest.ResponseRecorder{plain, zipped, head, notModified} {
		if header := result.Header(); header.Get("Cache-Control") != "public, max-age=300" ||
			header.Get("Cloudflare-CDN-Cache-Control") != "no-store" ||
			header.Get("Vary") != githubFeatureHeader+", Accept-Encoding" ||
			header.Get("X-Content-Type-Options") != "nosniff" {
			t.Fatalf("response %d headers: %v", result.Code, header)
		}
	}
	for _, tc := range []struct {
		encoding string
		want     string
		status   int
	}{
		{"", "", 200},
		{"x-gzip, identity;q=0", "gzip", 200},
		{"*;q=0.7", "gzip", 200},
		{"gzip;q=0, *;q=1", "", 200},
		{"gzip;q=0.4, identity;q=0.8", "", 200},
		{"identity;q=0, gzip;q=0", "", 406},
	} {
		got := catalogRequest(h, "GET", path, "Accept-Encoding", tc.encoding)
		if got.Code != tc.status || got.Header().Get("Content-Encoding") != tc.want {
			t.Errorf("%q got status %d, encoding %q; want %d, %q", tc.encoding, got.Code, got.Header().Get("Content-Encoding"), tc.status, tc.want)
		}
		if got.Code != 200 && got.Header().Get("Cache-Control") != "no-store" {
			t.Errorf("error got cache headers %v", got.Header())
		}
	}
	if got := catalogRequest(h, "GET", path, "If-None-Match", tag); got.Code != 200 {
		t.Errorf("gzip tag on identity response matched: %d", got.Code)
	}
	if got := catalogRequest(h, "HEAD", path, "If-None-Match", "*"); got.Code != 304 || got.Body.Len() != 0 {
		t.Errorf("conditional HEAD got %d, %d bytes", got.Code, got.Body.Len())
	}
	if calls != 1 {
		t.Fatalf("same catalog generated %d times", calls)
	}
}

func TestCatalogResponseKeysAndErrors(t *testing.T) {
	var cache catalogResponses
	calls := 0
	h := cache.wrap(endpoint(func(w http.ResponseWriter, r *http.Request) error {
		calls++
		if r.URL.Path == "/missing" {
			w.Header().Set("X-Sample", "error header")
			return invalid("bad query")
		}
		reply(w, http.StatusOK, fmt.Sprintf("%s?%s: %t", r.URL.EscapedPath(), r.URL.RawQuery, githubCatalog(r)))
		return nil
	}))
	for _, tc := range []struct {
		path     string
		features []string
		want     string
	}{
		{"/a%2Fb?a=1&b=2", nil, "/a%2Fb?a=1&b=2: false"},
		{"/a/b?a=1&b=2", nil, "/a/b?a=1&b=2: false"},
		{"/a%2Fb?b=2&a=1", nil, "/a%2Fb?b=2&a=1: false"},
		{"/a%2Fb?a=1&b=2", []string{"github-v1"}, "/a%2Fb?a=1&b=2: true"},
		{"/a%2Fb?a=1&b=2", []string{"GitHub-v1"}, "/a%2Fb?a=1&b=2: false"},
		{"/a%2Fb?a=1&b=2", []string{"github-v1", "github-v1"}, "/a%2Fb?a=1&b=2: false"},
	} {
		var fields []string
		for _, feature := range tc.features {
			fields = append(fields, githubFeatureHeader, feature)
		}
		w := catalogRequest(h, "GET", tc.path, fields...)
		if w.Code != 200 || !strings.Contains(w.Body.String(), strings.ReplaceAll(tc.want, "&", `\u0026`)) {
			t.Fatalf("%s %q: got %d %s, want %q", tc.path, tc.features, w.Code, w.Body.String(), tc.want)
		}
	}
	if calls != 4 {
		t.Fatalf("got %d fills for four effective keys", calls)
	}
	for range 2 {
		w := catalogRequest(h, "GET", "/missing", "Accept-Encoding", "gzip", "If-None-Match", "*")
		if w.Code != 400 || w.Header().Get("X-Sample") != "error header" ||
			w.Header().Get("Cache-Control") != "no-store" || w.Header().Get("Content-Encoding") != "" ||
			w.Header().Get("ETag") != "" || !strings.Contains(w.Body.String(), "bad query") {
			t.Fatalf("error was changed: %d %v %s", w.Code, w.Header(), w.Body.String())
		}
	}
	if calls != 6 {
		t.Fatalf("error was cached: %d calls, want 6", calls)
	}
}

func TestCatalogResponseBounds(t *testing.T) {
	var cache catalogResponses
	var calls int
	big := strings.Repeat("a", catalogMaxBody/2+1)
	overflow := strings.Repeat("b", catalogMaxBody+1)
	h := cache.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		switch r.URL.Path {
		case "/large":
			reply(w, http.StatusOK, big)
		case "/overflow":
			reply(w, http.StatusOK, overflow)
		default:
			reply(w, http.StatusOK, r.URL.RawQuery)
		}
	}))
	for i := range catalogCacheEntries + 1 {
		catalogRequest(h, "GET", "/small?item="+strconv.Itoa(i))
	}
	if len(cache.entries) != catalogCacheEntries {
		t.Fatalf("entry cap: %d", len(cache.entries))
	}
	for i := range 2*catalogCacheBytes/len(big) + 1 {
		catalogRequest(h, "GET", "/large?item="+strconv.Itoa(i))
	}
	if cache.size > catalogCacheBytes || len(cache.entries) > catalogCacheBytes/len(big) {
		t.Fatalf("byte cap: %d bytes, %d entries", cache.size, len(cache.entries))
	}
	before := calls
	for range 2 {
		w := catalogRequest(h, "GET", "/overflow")
		if w.Code != 200 || w.Body.Len() != len(overflow)+3 || w.Header().Get("Cache-Control") != "no-store" {
			t.Fatalf("large response did not pass through: %d %v (%d bytes)", w.Code, w.Header(), w.Body.Len())
		}
		catalogRequest(h, "GET", "/small?query="+strings.Repeat("x", catalogMaxKey))
	}
	if calls != before+4 {
		t.Fatalf("an oversized body or key was cached: %d calls, want %d", calls, before+4)
	}
}

func TestCatalogResponseConcurrentFill(t *testing.T) {
	var cache catalogResponses
	var calls atomic.Int32
	started, release := make(chan struct{}), make(chan struct{})
	h := cache.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		reply(w, http.StatusOK, "same")
	}))
	var readers sync.WaitGroup
	for i := range 12 {
		readers.Go(func() {
			encoding := "identity"
			if i%2 == 0 {
				encoding = "gzip"
			}
			w := catalogRequest(h, "GET", "/catalog", "Accept-Encoding", encoding)
			if w.Code != 200 || w.Header().Get("ETag") == "" {
				t.Errorf("fill got %d %v", w.Code, w.Header())
			}
		})
	}
	<-started
	close(release)
	readers.Wait()
	if calls.Load() != 1 {
		t.Fatalf("handler called %d times", calls.Load())
	}
}

func TestCatalogResponseGenerationAndCancellation(t *testing.T) {
	t.Run("generation", func(t *testing.T) {
		var cache catalogResponses
		var calls atomic.Int32
		started, release := make(chan struct{}), make(chan struct{})
		h := cache.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if calls.Add(1) == 1 {
				close(started)
				<-release
				reply(w, http.StatusOK, "old")
				return
			}
			reply(w, http.StatusOK, "new")
		}))
		original := make(chan *httptest.ResponseRecorder, 1)
		go func() { original <- catalogRequest(h, "GET", "/catalog") }()
		<-started
		cache.invalidate()
		current := catalogRequest(h, "GET", "/catalog")
		close(release)
		for _, got := range []*httptest.ResponseRecorder{current, <-original, catalogRequest(h, "GET", "/catalog")} {
			if got.Code != 200 || got.Body.String() != "\"new\"\n" || got.Header().Get("ETag") != current.Header().Get("ETag") {
				t.Fatalf("stale generation: %d %s", got.Code, got.Body.String())
			}
		}
		if calls.Load() != 2 {
			t.Fatalf("expected one fill per generation, got %d", calls.Load())
		}
	})
	t.Run("canceled leader", func(t *testing.T) {
		var cache catalogResponses
		var calls atomic.Int32
		started := make(chan struct{})
		h := cache.wrap(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if calls.Add(1) == 1 {
				close(started)
				<-r.Context().Done()
				reply(w, http.StatusOK, "canceled")
				return
			}
			reply(w, http.StatusOK, "live")
		}))
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		leader := make(chan struct{})
		go func() {
			defer close(leader)
			h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/catalog", nil).WithContext(ctx))
		}()
		<-started
		waiter := make(chan *httptest.ResponseRecorder, 1)
		go func() { waiter <- catalogRequest(h, "GET", "/catalog") }()
		cancel()
		<-leader
		for _, got := range []*httptest.ResponseRecorder{<-waiter, catalogRequest(h, "GET", "/catalog")} {
			if got.Code != 200 || got.Body.String() != "\"live\"\n" {
				t.Fatalf("canceled response poisoned healthy request: %d %s", got.Code, got.Body.String())
			}
		}
		if calls.Load() != 2 {
			t.Fatalf("expected canceled fill then healthy fill, got %d", calls.Load())
		}
	})
}
