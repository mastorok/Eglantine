# GitHub patch catalog

The GitHub catalog indexes existing patch and cheat repositories. The server
does not copy files or serve downloads. Each listing names the source and
author, links to a full Git commit, and records the license status. The desktop
downloads eligible patches directly from that commit on GitHub, verifies the
catalog SHA-256, and installs only the selected patch or cheat section.

## Reviewed sources

The upstream trees and READMEs below do not declare redistribution terms for
their patches. GitHub's license endpoint also returned 404 for all five
repositories. The importer records `NOASSERTION`, which is not a license grant.
None of the five reviewed READMEs explicitly forbids direct downloads by
other tools. Review new pins and any path-specific terms before adding them.
Keep sources that forbid third-party tools link-only. Also omit downloads for
patches whose format, target, or selected component cannot be established.
The Octopath Traveler 0 README prohibits reuploads and asks readers to use
the original links. The server does not reupload those files.

| Source | Reviewed commit | License status |
|---|---|---|
| [Kenji-NX/switch-pchtxt-mods](https://github.com/Kenji-NX/switch-pchtxt-mods/tree/13f0e6559136ead857e7c66d53f1ef1775d339f4) | `13f0e6559136ead857e7c66d53f1ef1775d339f4` | Not declared |
| [Fl4sh9174/Switch-Emulator-Ultrawide-FPS-Mods](https://github.com/Fl4sh9174/Switch-Emulator-Ultrawide-FPS-Mods/tree/5eed21701e8091389e2e746f073f694ea2ad4ec6) | `5eed21701e8091389e2e746f073f694ea2ad4ec6` | Not declared |
| [StevensND/ue4-emuswitch-60fps](https://github.com/StevensND/ue4-emuswitch-60fps/tree/53c69bf93c2f79a6a0e6e7a46f26ea0f0010f06c) | `53c69bf93c2f79a6a0e6e7a46f26ea0f0010f06c` | Not declared |
| [ChanseyIsTheBest/NX-60FPS-RES-GFX-Cheats](https://github.com/ChanseyIsTheBest/NX-60FPS-RES-GFX-Cheats/tree/a2755da3eea706734d459349370757fbe2375cfc) | `a2755da3eea706734d459349370757fbe2375cfc` | Not declared |
| [StevensND/switch-port-mods](https://github.com/StevensND/switch-port-mods/tree/774e7f7c85561ddd865c1cab555f2a335054d522) | `774e7f7c85561ddd865c1cab555f2a335054d522` | Not declared |

Use the title IDs and game names declared by the source. A conflicting
README, archive path, or game mapping is not enough to establish a target.
The Chansey repository's `GAMES.md` links to both the cheat and its game
credit file. A documented 16-digit cheat build ID compares with the first
eight bytes of the installed main executable's build ID. An untyped IPS or
IPSwitch build ID does not prove a mismatch with main.

The desktop needs GitHub catalog support to show these entries. Older desktop
builds continue to see the same GameBanana and managed listings.

## Build and publish a manifest

Clone the five repositories from GitHub into one scratch directory, using
each repository's basename. The importer reads the reviewed commit from
the local Git object store. It does not watch the working tree, contact
GitHub, or execute scripts from the clones.

```sh
scratch="$HOME/.capy/work/github-catalog"
review_dir=$(mktemp -d "$scratch/run.XXXXXXXX") &&
	python3 -I tools/modstore-server/import_github.py "$scratch/clones" \
		--batch-dir "$review_dir/batches" \
		> "$review_dir/manifest.json" 2> "$review_dir/import.log" &&
	printf 'Review run: %s\n' "$review_dir"
```

If the import fails, check `import.log` in the run directory and do not
publish any of its batches. After a successful import, review `manifest.json`
and the omission counts in `import.log` from the printed run directory.
If Git does not have a reviewed commit, fetch that exact commit rather than
substituting a branch tip. Reusing the clones makes the import repeatable
without API requests.

Deploy a server that supports the GitHub endpoint. Set `run` to the absolute
path of the successful, reviewed run directory on the deployment host. From
the server's Compose directory, publish through the private admin listener:

```sh
(
	run=/absolute/path/to/reviewed/run
	resume_batch=
	export LC_ALL=C
	set -- "$run"/batches/github-*.json
	if [ ! -f "${1:-}" ]; then
		printf 'No publish batches in %s\n' "$run/batches" >&2
		exit 1
	fi
	for batch in "$@"; do
		if [ -n "$resume_batch" ]; then
			[ "${batch##*/}" = "$resume_batch" ] || continue
			resume_batch=
		fi
		sudo docker compose exec -T modstore /modstore admin \
			PUT /admin/v1/github application/json < "$batch" || {
			status=$?
			printf 'Failed to publish %s\n' "$batch" >&2
			exit "$status"
		}
	done
	if [ -n "$resume_batch" ]; then
		printf 'Batch %s not found in %s\n' "$resume_batch" "$run/batches" >&2
		exit 1
	fi
)
```

If an upload fails, the loop prints the failed batch path and returns
nonzero. Earlier batches stay published. To retry the same reviewed files
after fixing the failure, rerun the loop with the same `run` and set
`resume_batch` to the failed filename, such as `github-002.json`.
The loop skips the earlier batches. If the metadata needs a change, build
and review a new run instead.

The import keeps moderation decisions and does not remove omitted entries.
If an author removes a source or asks for removal, reject the listing through
the admin API. Verify the updated listing with the `github-v1` header and
check that requests without the header still return the legacy catalog.
