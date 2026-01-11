# VeZ + AI Tooling Guide

This guide explains how to integrate VeZ with AI-assisted coding tools, provide context for LLMs, and ensure deterministic, reproducible outputs.

## Goals
- **Fast onboarding for AI agents**: clear project structure and tasks.
- **Deterministic builds**: pin targets and profiles.
- **Minimal context packs**: small, focused prompts to reduce hallucinations.

## Recommended AI Context Pack
Provide these snippets to your AI tool when asking for help:

1) **Project entry points**
- Workspace: `Cargo.toml`
- Compiler lib: `compiler/src/lib.rs`
- Compiler bin: `compiler/src/main.rs`
- Tools: `tools/vpm`, `tools/lsp`, `tools/testing`

2) **Key docs**
- `README.md` (overview)
- `docs/INSTALL.md` (install)
- `FINAL_FEATURE_CHECKLIST.md` (what’s implemented)
- `BUILD_AND_TEST.md` (commands)
- `VEZ_FUNCTIONALITY_REPORT.md` (status)
- `PLUGIN_SYSTEM.md` (plugins)

3) **Build commands**
```bash
cargo build --workspace --release
cargo test --workspace
cargo check --workspace
```

4) **Binaries**
- `vezc` (compiler)
- `vpm` (package manager)
- `vez-lsp` (language server)

## AI-Friendly Conventions
- **Deterministic logging**: prefer structured logs; avoid timestamps in examples.
- **Explicit targets**: use `--target` when cross-compiling.
- **Small diffs**: when editing, touch the smallest surface.
- **Module map**: reference `PROJECT_STRUCTURE.md` for file locations.

## Example Prompts for AI
- “Explain how to add a new optimization pass in `compiler/src/optimizer`.”
- “Generate a plugin manifest and loader stub following `docs/PLUGIN_SYSTEM.md`.”
- “Add a CLI flag to `vezc` in `compiler/src/main.rs` and wire it to the driver.”

## Integration with Editors / LSP
- Use the `vez-lsp` binary built from releases or `target/release/vez-lsp`.
- Point your editor to the binary path; no extra config needed beyond standard LSP wiring.

## Deterministic CI Guidance
- Build with `cargo build --workspace --release --locked`.
- Cache cargo registry and git deps.
- Emit artifacts per target triple (as in `.github/workflows/release.yml`).

## Safety Notes
- Do not auto-run external network calls in build scripts.
- Keep API keys out of the repo; use env vars / secrets in CI.

## Troubleshooting AI Outputs
- If an AI suggests non-existent types (e.g., struct-style variants), cross-check `compiler/src/parser/ast.rs`.
- If borrow checker guidance is unclear, prefer two-phase read-then-write patterns.
