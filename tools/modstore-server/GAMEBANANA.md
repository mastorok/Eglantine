# Import GameBanana listings

Run the importer with Python 3.9 or later on a workstation, outside the
mod-store container. It writes one JSON manifest to stdout and puts diagnostics
on stderr. It uses public GameBanana API v11 JSON and links back to the author,
submission, preview images and official download. It does not scrape pages,
sign in, mirror files, or create server-hosted versions. The desktop client can
download supported ZIPs directly from GameBanana, even without a catalog SHA-256.

## Stage a manifest

From the repository root, use a private scratch directory outside the checkout:

```sh
umask 077
scratch="$HOME/.capy/work/gamebanana"
mkdir -p "$scratch"
python3 -B tools/modstore-server/import_gamebanana.py --scratch "$scratch/cache" \
  > "$scratch/manifest.next" &&
  mv "$scratch/manifest.next" "$scratch/manifest.json"
```

The defaults select up to 20 mods per mapped title. Each title reads the first
12 records from the New & Updated feed and the first 12 from Most Liked. The
importer alternates between the feeds, removes duplicates and checks each
candidate against its own game-scoped profile. Use `--title-id TITLE_ID` for a
smaller import. Repeat it to include several titles. Use `--per-feed` and
`--limit` to reduce the sample further.

The importer needs public, unrated, visible submissions with named credits and
at least one active file that GameBanana reports as clean. It keeps up to eight
credits, previews and files per mod and omits titles with no eligible mods.
The manifest includes a plain-text description excerpt. The canonical
submission page has the full description and author terms.

The title-to-hub map lives in `import_gamebanana.py`. Add only confirmed
title and platform matches. The Breath of the Wild entry uses Switch hub
`6386`, not Wii U hub `5866`. Shared hubs do not establish compatibility for
another edition. The shared Kirby and the Forgotten Land hub imports only to
the Switch 1 title. It takes files uploaded before August 27, 2025 UTC, or
files whose listing or file metadata explicitly identifies Switch 1 or
`01004D300C5AE000`. Conflicting Switch 2 labels disqualify the listing or file.
Pokémon Legends: Z-A is mapped only to its canonical Switch title ID. No
Switch 2 Xenoblade title is mapped.

Run one importer at a time. API and download requests share one monotonic pace
of at least a second per request, including redirects. The importer uses
cached API responses for 24 hours. If access is blocked or rate-limited, the
importer stops without a new manifest. Keep the same scratch cache and wait
for the recorded `Retry-After` deadline before rerunning online. If no usable
deadline was sent, the cooldown is one hour. Do not switch endpoints or routes
to get around a block. The importer skips a profile if its response is empty
or identifies itself as the ordinary public submission page. It does not read
listing data from HTML. Unexpected HTML, login, and challenge pages stop the
run.

Pass `--offline` with the same scratch root and feed limit to stage only the
listings already cached after an interrupted run. This makes no network
requests, skips cache misses, and logs how many feeds and profiles were
unavailable. Review the partial result before publishing it. If nothing is
eligible, the importer exits without a manifest.

## Opt in to file evidence

Without `--inspect-files`, the importer does not request downloads and emits
`sha256: null` and `build_ids: []` for every file. Add `--inspect-files` only when
you need file evidence. The default budget is eight new downloads of at most
8 MiB each. Set `--max-files` to 1–32 and `--max-file-mib` to 8–32, if needed.
Downloads start at the official `/dl/<id>` link and follow only vetted
GameBanana file-host redirects. Files without an upstream MD5 are skipped.

The importer checks the size and upstream MD5 before recording SHA-256. It
looks for build IDs in exact `.ips` or `.ips32` filenames and
`cheats/<16 hex digits>.txt` paths. In ZIP archives it reads a bounded central
directory, skips symlinks and traversal paths, and looks for patches under
`exefs` and `exefs_patches`. It reads only the `@nsobid-` directive from
direct `.pchtxt` files and regular, unencrypted `.pchtxt` entries in those
patch directories. These reads are capped at 64 KiB per patch and 256 KiB
per archive. The importer never extracts or executes patches. RAR and 7z
can get a hash but contribute no member evidence.

The temporary archive is deleted. Successful verification is cached by file
ID and upstream MD5 so later runs with `--inspect-files` can reuse it without
downloading again. An unavailable or changed file skips repeat attempts for
24 hours so the next run can inspect other files. Rate limits and transient
errors stop the run without creating a per-file result. Respect the upstream
cooldown before retrying. Keep the flag on to carry verified evidence into
the next manifest.

Missing hashes or build IDs mean that evidence was not collected. Without a
catalog hash, the client computes a local SHA-256 and still validates the archive
layout. A build ID in a filename is a clue, not a compatibility check. Imported
files remain unverified unless exact compatibility is established.

## Publish a reviewed manifest

Review the staged JSON before uploading it. If you confirm that an archive
contains independent components the client cannot select, add
`"ambiguous_layout": true` to that file object in `games[].mods[].files[]`.
The file stays link-only even if it has a SHA-256. Do not infer the marker from
a name, a missing hash, or a failed download. The importer does not set it.

Leave the field out of routine re-imports. The server preserves a marker for
the same file ID in the same mod. To clear a block after re-review, publish
`"ambiguous_layout": false` on that file. The marker does not affect other
files in the listing.

Deploy a server binary that supports `ambiguous_layout` before uploading a
manifest with the field. Publish known blocks before releasing clients that
allow installs without a catalog hash. Only the server operator runs this
command, from the mod-store Compose directory on the deployment host:

```sh
sudo docker compose exec -T modstore /modstore admin \
  PUT /admin/v1/gamebanana application/json < /path/to/reviewed/manifest.json
```

The private admin endpoint upserts supplied listings and preserves local
moderation. It does not remove listings absent from the new sample. Verify
that marked files expose `ambiguous_layout: true` under `external.files` in
the public mod detail. After verifying the first real publish in the public
catalog, remove demo listings with the separate private command:

```sh
sudo docker compose exec -T modstore /modstore admin \
  PUT /admin/v1/gamebanana/drop-demos < /dev/null
```

## Run the tests

From the repository root:

```sh
python3 -B -m unittest discover -s tools/modstore-server -p 'test_import_gamebanana.py'
```

The tests use synthetic API responses and archives. They do not contact
GameBanana or the mod-store server.
