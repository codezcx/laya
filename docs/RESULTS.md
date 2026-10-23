# MI + MAT experiment

Data preparation complete;12 unit/distributed tests passed, all MI manifest rows match the prior run, full49,193-trial coverage verified, and every MAT train/validation window and coordinate shape checked. Two-rank final-partial-batch training/validation smoke passed. Production launched on GPUs2–3 and reached step30/1538; no full validation yet. Tests/smoke and production status are recorded in artifacts/mi_mat/pipeline.json. Production metrics: artifacts/mi_mat/contrast/metrics.jsonl. No quality results yet. See [PROTOCOL.md](PROTOCOL.md).

Historical context (different auxiliary dataset): MI-only final macro BA81.90%; MI+SEED shared head55.87% MI /55.56% SEED; independent heads53.00% MI /60.00% SEED. These are text-match validation scores. Detailed historical results and implementation are archived in Git7f579e0; ignored artifacts remain intact.
