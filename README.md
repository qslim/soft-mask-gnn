# Soft-mask GNNs: Interpretable Hierarchical Graph Representation Learning

The code is built upon the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

## Prerequisites

python 3.7.3

pytorch 1.2

pytorch-geometric 1.3.0

Additional modules: numpy, easydict

## Running the tests

1. Set hyper-parameters in [configs](configs).

2. Run tests.

   - To run 10-fold cross-validation on MUTAG, run the classification_main.py script:

     ```
     python classification_main.py --config=configs/MUTAG.json
     ```

     To run other classification benchmarks, replace MUTAG.json with corresponding configuration files.

   - To run the QM9 test, run the regression_main.py script:

     ```
     python regression_main.py --config=configs/QM9.json
     ```

