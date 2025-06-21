## Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing

![替代文本](Framework.jpg)

This repository provides the PyTorch implementation for our TKDE'24 paper [[BGCH+](https://arxiv.org/pdf/2408.09239)].

## 💾 Datasets

### Download

The datasets used in our paper can be downloaded from the following link. 

* **Dataset Link:** [Google Drive](https://drive.google.com/file/d/1Aj782w1YR1JooW5uXZyozv5ddlNwvRbe/view?usp=sharing)

After downloading, place the dataset folder under the root.





## 🔧 Requirements

The codes are built with the following main dependencies:


* `pytorch=1.8.1`
* `pandas`
* `scipy`
* `numpy`
* `tensorboardX`
* `scikit-learn`
* `tqdm`
* `pybind11`
* `cppimport`



## 🚀 Usage

To training/evaluating our model, please execute the command of the specific dataset, as shown in sota_settings.py

## 📄 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{chen2024towards,
  title={Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing},
  author={Chen, Yankai and Fang, Yixiang and Zhang, Yifei and Ma, Chenhao and Hong, Yang and King, Irwin},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```