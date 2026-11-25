# Box, Box — 2026 Australian Grand Prix F1 Analytics

## Data Setup (TracingInsights Sparse Checkout)
`data/2026/` is kept as a separate local Git repository (a folder of) (ignored in `.gitignore`) so its nested `.git` history stays isolated from `Box-Box`. 

To reproduce the local snapshot in `data/2026/Australian Grand Prix`, run:

```bash
git clone --filter=blob:none --no-checkout https://github.com/TracingInsights/2026.git && git -C 2026 sparse-checkout set "Australian Grand Prix" && git -C 2026 checkout