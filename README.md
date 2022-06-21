# Soft-mask: Adaptive Substructure Extractions for Graph Neural Networks

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper "Soft-mask: Adaptive Substructure Extractions for Graph Neural Networks".

## Prerequisites

The code is built upon [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

The following packages need to be installed:

- `python 3.7.3`
- `pytorch 1.2`
- `pytorch-geometric 1.3.0`
- `Additional modules: numpy, easydict`

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

## Reference
```
@inproceedings{yang2021softmask,
	author = {Yang, Mingqi and Shen, Yanming and Qi, Heng and Yin, Baocai},
	title = {Soft-Mask: Adaptive Substructure Extractions for Graph Neural Networks},
	year = {2021},
	url = {https://doi.org/10.1145/3442381.3449929},
	booktitle = {Proceedings of the Web Conference 2021},
	pages = {2058–2068},
	numpages = {11},
	series = {WWW'21}
}
```

## License

[MIT License](LICENSE)