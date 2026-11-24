# Lichii Mod Store server

This standalone service serves a moderated catalog from SQLite and a directory of
local files. The server does not fetch or proxy third-party files or previews.
There is no public upload or account system.
See [API.md](API.md) for the Qt contract and private admin routes.

## Start on the host

From `tools/modstore-server/`, with Docker and Docker Compose 2.30 or later:

```sh
sudo install -d -m 0700 -o 65532 -g 65532 data
(
  umask 077
  set -C
  { printf 'MODSTORE_ADMIN_TOKEN='; openssl rand -hex 32; } > .env.pi
)
sudo docker compose build
# Optional: insert the original, inert demo catalog before starting the service.
sudo docker compose run --rm --no-deps modstore seed-demo
sudo docker compose up -d
curl --fail http://127.0.0.1:8767/healthz
```

Keep `.env.pi` on the host. The command refuses to overwrite an existing file.
Never put the token in shell arguments, source control, logs, or a public
client. Docker administrators can inspect container environment variables.
Grant Docker access only to trusted operators, and use
`docker compose config -q` if you need to check the Compose file without
printing its evaluated configuration.

Only host loopback port **8767** is published. An ingress proxy on the same
host may use `http://127.0.0.1:8767` as its origin. The public listener has
no admin routes. The admin listener binds to **container** loopback port
**8768**, which is not published. Use `docker compose exec` locally or over SSH
to run the bundled client. Do not publish 8768, share the container network
namespace, or run the Compose service in host network mode.

Compose uses a non-root container, a read-only root filesystem, and a 192 MiB
memory limit. Only the data bind mount and a small temporary filesystem are
writable. Set `MODSTORE_HOST_DATA_DIR` before invoking Compose to use a different
data directory with the same ownership. Set `MODSTORE_ENV_FILE` to choose
another host-only token file. If the host uses rootless Docker or user
namespace remapping, give the runtime-mapped UID/GID write access to that
directory instead of using literal host UID/GID 65532.

To run the same image without Compose after creating `data/` and `.env.pi`:

```sh
sudo docker build -t lichii-modstore:local .
sudo docker run -d --name lichii-modstore --restart unless-stopped \
  --memory 192m --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m \
  --mount "type=bind,source=$PWD/data,target=/data" --env-file .env.pi \
  -p 127.0.0.1:8767:8767 lichii-modstore:local
sudo docker exec lichii-modstore /modstore health
```

The image builds natively on ARM64 and x86-64. It uses Go's HTTP, ZIP and
image packages and a pure-Go SQLite driver.

## Moderate a release

Use `/modstore admin GET|PUT /admin/v1/... [content-type]` inside the
container. The client adds the token from the container environment.
For a PUT, pipe the JSON, ZIP, or image bytes into its standard input:

```sh
sudo docker compose exec -T modstore /modstore admin GET '/admin/v1/mods?status=pending'
sudo docker compose exec -T modstore /modstore admin GET '/admin/v1/mods/my-mod'
sudo docker compose exec -T modstore /modstore admin GET \
  '/admin/v1/mods/my-mod/versions/1.0.0/download' > review.zip
```

Register the game before you upload a version. Review the metadata, inspect
the files, and verify you can distribute the material and the exact target
is correct before you approve it. A preview upload stores the original PNG
or JPEG. A ZIP upload validates the paths and data, then writes a fresh
archive with ordinary ZIP entries. Use the returned SHA-256 and sizes for
the served archive, not for the original upload. Structural checks cannot
determine whether a patch is safe to run or whether you have distribution
rights.

New mods and new versions are pending. A metadata or preview change hides
an already published mod until it is approved again. A new version does not
hide existing approved versions. Approve the desired version and its parent
mod to publish:

```sh
printf '%s\n' '{"status":"approved"}' | sudo docker compose exec -T modstore \
  /modstore admin PUT '/admin/v1/mods/my-mod/versions/1.0.0/status' application/json
printf '%s\n' '{"status":"approved"}' | sudo docker compose exec -T modstore \
  /modstore admin PUT '/admin/v1/mods/my-mod/status' application/json
```

Set the same object to `rejected` to hide it. Rejection does not delete the
bytes. Stop the service before running `seed-demo` on an existing catalog,
then start it again to rebuild the public catalog snapshot. `serve` and
`seed-demo` refuse to share a data directory while either is running.
Running the seed again doesn't overwrite records or undo moderation.

GameBanana entries are links to third-party listings, not locally hosted mods.
See [the import guide](GAMEBANANA.md) to refresh them and remove the demo listings.
See [the GitHub patch catalog guide](GITHUB.md) for verified direct downloads
of pinned patches and cheats.

## Back up or restore

Stop the service before copying data so the database and assets come from
the same point in time. Back up the entire data directory, including
`catalog.sqlite3` and any WAL files, not just the main database file.
Back up the token separately in a secret store, or create a new one at restore.

```sh
sudo docker compose stop modstore
sudo tar -C data -czf - . > "$HOME/modstore-data-$(date +%F).tar.gz"
sudo docker compose up -d
```

If the backup fails, do not treat the output file as a valid archive. To
restore, stop the service, move the old data directory aside, and create an
empty directory with the original ownership. Extract only a trusted backup into
that directory with `sudo tar --numeric-owner -xzf FILE -C data`.
Start the service and check health and a known download. Never run two
containers against the same writable catalog. Monitor disk usage: old
assets and files from interrupted writes are not automatically collected.

## Run the tests

From this directory, run `go test ./...` with Go 1.26 or later. To check
the ARM64 build on x86-64, run
`CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o /tmp/modstore-arm64 .`.
