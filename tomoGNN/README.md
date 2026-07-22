# TomoGNN — Netlist Tomography 

TomoGNN is a self-supervised graph model trained with PPA-aware contrastive loss to produce instance embeddings that encode 
stable spatial-timing relationships. HDBSCAN-based clustering then generates soft guides, which are adaptively refined across the P&R flow. 
 
## Features
- Joint learning over the shared netlist graph G and per‑run co‑location graphs H.
- Quality‑aware fusion across runs; enhanced contrastive loss with calibration.
- End‑to‑end pipeline: data loading → graph building → training → clustering → DEF.

## Project layout 
```
tomoGNN/
├── main_pipeline.py           # Entry point: end-to-end pipeline
├── train.py                   # Training loop (TomoGNN) + embedding export
├── config.py                  # DualViewConfig + TrainConfig (alias)
├── models/
│   └── models_dualview.py     # TomoGNN model (class TomoGNN)
├── data/
│   ├── data_loader.py         # CSV ingestion, clock period helpers
│   ├── graph_builder_dualview.py  # Multi-run graph construction (G,H)
│   ├── graph_builder.py       # Netlist graph features/utilities
│   ├── colocation_graph.py    # Co-location graph builders + caching
│   ├── edge_weights.py        # Criticality/length/hierarchy weights
│   ├── hierarchy.py           # Hierarchy path utilities and weights
│   └── multi_run_module.py    # Learnable/naive multi-run weighting
├── utils/
│   ├── sampler.py             # Positive/negative pair sampling
│   ├── cluster_hdbscan.py     # HDBSCAN clustering
│   └── cluster_leiden.py      # Optional Leiden clustering
├── generate_def.py            # DEF writer + cluster CSV export
├── requirements.txt           # Minimal runtime requirements
├── run.sh                     # Example launcher (uses test_data/)
└── setup_env.sh               # Environment bootstrap (optional)
```

## Requirements
- Linux, CUDA-enabled GPU recommended (≥12 GB).
- Python ≥ 3.7.
- PyTorch, PyTorch Geometric, numpy, pandas, scikit-learn, hdbscan, etc. See `requirements.txt` and `setup_env.sh`.

## Setup
Option A (scripted):
1. `cd tomoGNN`
2. `./setup_env.sh`
3. `source venv/bin/activate`

## Data inputs
Provide two CSVs per design:
- Nodes: `*_merged_nodes.csv`, with per-run coordinates `pt_x_1..pt_x_R`, `pt_y_1..pt_y_R`, slack fields, etc.
- Edges: `*_merged_edges.csv`, with net connectivity and per-run timing/length stats.
The clock period is auto‑derived from nodes via `data_loader.get_clock_period`.

## Run
Quick start via the example launcher (adjust paths as needed):
```bash
cd tomoGNN
bash run.sh
```
Or run the pipeline module directly:
```bash
python -m tomoGNN.main_pipeline \
  --design <name> \
  --nodes <path/to/nodes.csv> \
  --edges <path/to/edges.csv> \
  --output results/<name>_<timestamp>
```
Most hyperparameters live in `config.py`. CLI flags override config defaults.

## Outputs
Under the chosen `--output` directory:
- `embeddings_tomognn.npy` — L2‑normalized node embeddings
- `cluster_map_dualview.csv` — node → cluster mapping
- `<design>_clusters_tomognn.def` — DEF overlay for physical evaluation
- `checkpoints/` with `best_model_tomognn.pt` when enabled
- `run.log` and `training_log.txt` — logs

## Co‑location graph caching
Sequential graph build caches each run’s H to:
```
tomoGNN/data/.cache/colocation_graphs/run_<runIdx>_<cacheKey>.pkl
```
Cache key includes coordinate hash and parameters (radius/sigma/q/etc.). On cache hits, H is loaded directly.

## Troubleshooting
- Python version: ensure ≥ 3.7 (older versions error at future annotations).
- CUDA OOM: reduce `batch_size_nodes`, `fanouts`, or `colocation_q` in `config.py`.
- Missing PyG/torch wheels: install versions compatible with your CUDA, or use `setup_env.sh`.
- Leiden not available: install `python-igraph` and `leidenalg`, or use HDBSCAN (`--use-leiden` off).

## License
See repository root license for usage terms of this codebase.

