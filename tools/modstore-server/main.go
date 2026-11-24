package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
)

const publicAddress = "127.0.0.1:8767"
const adminAddress = "127.0.0.1:8768"

var tokenPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func configuredPublicAddress() string {
	if address := os.Getenv("MODSTORE_PUBLIC_ADDR"); address != "" {
		return address
	}
	return publicAddress
}

func tokenFromEnv() (string, error) {
	token := os.Getenv("MODSTORE_ADMIN_TOKEN")
	if !tokenPattern.MatchString(token) {
		return "", errors.New("set MODSTORE_ADMIN_TOKEN to 64 lowercase hexadecimal digits")
	}
	return token, nil
}

func serve(s *store, publicAddr, token string) error {
	if _, err := s.catalog(context.Background()); err != nil {
		return fmt.Errorf("load catalog: %w", err)
	}
	public, err := net.Listen("tcp4", publicAddr)
	if err != nil {
		return err
	}
	defer public.Close()
	admin, err := net.Listen("tcp4", adminAddress)
	if err != nil {
		return err
	}
	defer admin.Close()
	publicHandler := s.publicHandler()
	publicServer := http.Server{Handler: publicHandler, ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, WriteTimeout: 90 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	adminServer := http.Server{Handler: s.adminHandler(token), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 90 * time.Second, WriteTimeout: 120 * time.Second, IdleTimeout: 15 * time.Second, MaxHeaderBytes: 16 << 10}
	defer publicServer.Close()
	defer adminServer.Close()
	stopping, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errs := make(chan error, 2)
	go func() { errs <- publicServer.Serve(public) }()
	go func() { errs <- adminServer.Serve(admin) }()
	log.Printf("Public listener: %s; admin listener: %s", public.Addr(), admin.Addr())
	select {
	case err := <-errs:
		return err
	case <-stopping.Done():
		drain, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		publicShutdown := make(chan error, 1)
		go func() { publicShutdown <- publicServer.Shutdown(drain) }()
		adminErr := adminServer.Shutdown(drain)
		return errors.Join(adminErr, <-publicShutdown)
	}
}

func adminRequest(args []string) error {
	if len(args) < 2 || len(args) > 3 || (args[0] != "GET" && args[0] != "PUT") {
		return errors.New("usage: modstore admin GET|PUT /admin/v1/... [content-type]")
	}
	token, err := tokenFromEnv()
	if err != nil {
		return err
	}
	target, err := url.ParseRequestURI(args[1])
	if err != nil || target.IsAbs() || !strings.HasPrefix(target.Path, adminPrefix+"/") {
		return errors.New("expected an /admin/v1/ path")
	}
	var body io.Reader
	if args[0] == "PUT" {
		body = os.Stdin
	}
	r, err := http.NewRequest(args[0], "http://"+adminAddress+target.RequestURI(), body)
	if err != nil {
		return err
	}
	r.Header.Set("Authorization", "Bearer "+token)
	if len(args) == 3 {
		r.Header.Set("Content-Type", args[2])
	}
	client := http.Client{Timeout: 125 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := client.Do(r)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	dst := os.Stdout
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		dst = os.Stderr
	}
	if _, err := io.Copy(dst, res.Body); err != nil {
		return err
	}
	if dst == os.Stderr {
		return fmt.Errorf("admin request returned HTTP %d", res.StatusCode)
	}
	return nil
}

func lockCatalog(dir string) (func(), error) {
	path, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(path, 0700); err != nil {
		return nil, err
	}
	uri := url.URL{Scheme: "file", Path: filepath.ToSlash(filepath.Join(path, "service-lock.sqlite3")),
		RawQuery: "_txlock=immediate"}
	db, err := sql.Open("sqlite", uri.String())
	if err != nil {
		return nil, err
	}
	tx, err := db.Begin()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("lock catalog directory (stop the running service before seeding): %w", err)
	}
	return func() {
		_ = tx.Rollback()
		_ = db.Close()
	}, nil
}

func run(args []string) error {
	command := "serve"
	if len(args) != 0 {
		command = args[0]
	}
	switch command {
	case "admin":
		return adminRequest(args[1:])
	case "health":
		host, port, err := net.SplitHostPort(configuredPublicAddress())
		if err != nil {
			return err
		}
		if host == "" || net.ParseIP(host).IsUnspecified() {
			host = "127.0.0.1"
		}
		transport := &http.Transport{DialContext: func(ctx context.Context, _, address string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp4", address)
		}}
		defer transport.CloseIdleConnections()
		client := http.Client{Timeout: 3 * time.Second, Transport: transport,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		r, err := http.NewRequestWithContext(context.Background(), http.MethodGet,
			"http://"+net.JoinHostPort(host, port)+"/healthz", nil)
		if err != nil {
			return err
		}
		res, err := client.Do(r)
		if err != nil {
			return err
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			return fmt.Errorf("health check returned HTTP %d", res.StatusCode)
		}
		return nil
	case "serve", "seed-demo":
		dir := os.Getenv("MODSTORE_DATA_DIR")
		if dir == "" {
			dir = "data"
		}
		release, err := lockCatalog(dir)
		if err != nil {
			return err
		}
		defer release()
		s, err := openStore(dir)
		if err != nil {
			return err
		}
		defer s.db.Close()
		if command == "seed-demo" {
			return s.seedDemo(context.Background())
		}
		token, err := tokenFromEnv()
		if err != nil {
			return err
		}
		return serve(s, configuredPublicAddress(), token)
	default:
		return errors.New("usage: modstore [serve|seed-demo|health|admin]")
	}
}

func main() {
	log.SetFlags(0)
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}
