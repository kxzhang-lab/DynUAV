# GitHub Upload Notes

This project is intended to be uploaded from `TrackTrack-main` as a standalone repository.

## Before Upload

Check the files Git will include:

```bash
git status --short
git add --dry-run .
```

The `.gitignore` excludes generated pickles, model weights, tracking outputs, and local scratch scripts. Do not commit datasets or trained weights directly to GitHub; use GitHub Releases, Hugging Face, Google Drive, or another file host and link them from `README.md`.

The precomputed baseline artifact `FastReID/dyn.pickle` contains detections with ReID embeddings. Keep it out of Git history and publish it as a downloadable artifact, then add the Google Drive link to `README.md`.

## Create and Push

If the remote repository already exists:

```bash
git init
git branch -M main
git add .
git commit -m "Release DynUAV TrackTrack baseline"
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

If you use the GitHub CLI:

```bash
gh repo create <OWNER>/<REPO> --public --source=. --remote=origin --push
```

Replace `<OWNER>/<REPO>` with the target GitHub organization/user and repository name.
