# Keeping this fork in step with Lichii

This fork's `claude/lichii` branch is `zortos293/lichii` (branch `nx2`) plus a few commits of our own. It
descends from the real Lichii history, so Lichii's later commits merge in without losing either side.

## Take Lichii's new commits

```sh
git remote add lichii https://github.com/zortos293/lichii      # once
git fetch --filter=blob:none lichii nx2                         # history only; file contents on demand
git log --oneline HEAD..lichii/nx2                              # what is new
git merge lichii/nx2                                            # never rebase, never force-push
```

A conflict only appears where both sides changed the same lines. Resolve it keeping both changes,
build, then push. A merge was tried from a Lichii commit two days older carrying our workflow commit:
it merged without a conflict, although Lichii had edited the same workflow file.

## Keep our own changes easy to merge

- One small commit per change, with a message saying what it does and whether it ran in a game.
- Prefer a new file over an edit to a Lichii file, and a new switch that is off by default over a changed default.
- Do not reformat or move Lichii's code.
- Our workflow change (Windows only, no release, no update checker) is one commit on top of Lichii's tree.
  `ENABLE_UPDATE_CHECKER` stays OFF so the program never replaces itself with the upstream build.

## Build

A push to `claude/lichii` runs `release-binaries` (windows-x86_64 and windows-x86_64-pgo). The zips are
artifacts of the run, and the `publish-latest` job also puts them in one GitHub release tagged
`lichii-latest`, replacing the previous build's files each time (assets `lichii-windows-x86_64.zip` and
`lichii-windows-x86_64-pgo.zip`, no `build-N` tag per run). The launcher on the owner's PC reads that release
through the GitHub API, which needs no token because the fork is public.

## Things that bit us

- Lichii's workflow saves the compiler caches only on `nx2`; ours also saves them on `claude/lichii`, so a
  build here takes about 20-25 minutes warm and about 45-60 minutes cold (right after a big merge).
- The release job tags the branch tip (`--target "${GITHUB_REF_NAME}"`). Targeting the build's own commit
  fails with HTTP 403 if the branch moved on while the build ran.
- After a merge, the first push builds cold. Do not push again before it ends if a release matters.
