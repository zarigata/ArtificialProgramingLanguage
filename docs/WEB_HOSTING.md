# Hosting VeZ docs on GitHub Pages

This repository now includes a static landing page at `docs/index.html`. You can host it with GitHub Pages without additional build steps.

## Quick setup (project site)
1. Push the repo to GitHub (public or private with Pages enabled).
2. In **Settings → Pages**, set:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main` (or your default) / `/docs` folder
3. Save. GitHub Pages will publish at `https://<org>.github.io/<repo>/`.

## Custom domain (recommended)
- Buy and point `vez-lang.org` (or your chosen domain) to the Pages CNAME.
- In **Settings → Pages → Custom domain**, enter the domain (e.g., `vez-lang.org`).
- Add DNS records:
  - `A` records to GitHub Pages IPs or `CNAME vez-lang.org` → `<org>.github.io`.
- Commit a `CNAME` file at repo root (optional but recommended) containing the domain.

## Local preview
```bash
# From repo root
python -m http.server 8000 --directory docs
# Visit http://localhost:8000
```

## SEO checklist for this page
- Title/description/meta keywords already set in `docs/index.html`.
- OpenGraph/Twitter tags included.
- Structured data: SoftwareApplication schema embedded.
- Use meaningful anchors (`#why`, `#llm`, `#docs`, `#examples`, `#contact`).
- Keep filenames and URLs lower-case and stable.

## Extending content
- Add new sections to `docs/index.html` (keep headings semantic: `h2` for sections, `h3` for sub-sections).
- Link to existing markdown: `docs/AI_INTEGRATION.md`, `docs/ARCHITECTURE.md`, `docs/SPECIFICATION.md`, `examples/complete_example.zari`.
- If you add assets, place them under `docs/assets/` and reference with relative paths.

## Automation (optional)
- **Enable GitHub Pages workflow (added)**: `.github/workflows/pages.yml` auto-deploys `docs/` to Pages on every push to `main` or manual dispatch.
- Add a `sitemap.xml` in `docs/` if you grow beyond a single page.
- Add `robots.txt` allowing `User-agent: *` and `Allow: /`.

## GitHub Actions deploy (already configured)
The repo includes `.github/workflows/pages.yml` using the official Pages actions.

What it does:
- Runs on pushes to `main` (and `workflow_dispatch`).
- Uploads `docs/` as the static site artifact.
- Deploys via `actions/deploy-pages`.

How to enable in GitHub UI (one-time):
1) Push the workflow to GitHub.
2) In **Settings → Pages**, set **Source** to **GitHub Actions**.
3) Trigger: push to `main` or run the workflow manually from **Actions → Deploy docs to GitHub Pages**.

Notes:
- Keep `docs/index.html` as the entry point.
- Custom domain: add CNAME in Settings → Pages and (optionally) commit a `CNAME` file at repo root.
