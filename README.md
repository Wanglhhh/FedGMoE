# FedGMoE

FedGMoE is a federated graph learning method for node classification under distributed graph data. This repository contains the core implementation and a lightweight runner based on OpenFGL.

## Environment

The code is intended for Python 3.10 or later. The main dependencies are:

- PyTorch 2.x
- PyTorch Geometric 2.x
- NumPy
- scikit-network
- tqdm
- matplotlib

Install PyTorch and PyTorch Geometric according to the local CUDA version, then install the remaining packages:

```bash
pip install numpy scikit-network tqdm matplotlib
```

CPU execution is supported, while a CUDA-enabled environment is recommended for larger datasets.

## Data

Datasets are stored under `dataset/` by default. Common PyG datasets can be prepared automatically on the first run. A different data directory can be specified with `--root`.

## Run

Run one FedGMoE experiment on Cora:

```bash
python run_fedgmoe.py --dataset Cora
```

Select a GPU and change the number of clients if needed:

```bash
python run_fedgmoe.py --dataset Cora --gpuid 0 --num_clients 10
```

Each command executes one run with one scalar seed. Use `--use_cuda false` for CPU execution. The retained datasets are Cora, CiteSeer, PubMed, Actor, Minesweeper, and Roman-empire.

## Output

Training logs are written to `logs/` by default. The output directory can be changed with `--log_root`.

## Project Structure

```text
FedGMoE/
|-- openfgl/
|   |-- data/
|   |-- flcore/fedgmoe/
|   |-- task/
|   `-- utils/
|-- run_fedgmoe.py
`-- README.md
```
