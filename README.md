# Continuous-Time Attention: PDE-Guided Mechanisms for Long-Sequence Transformers

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg)](https://2025.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.20666-b31b1b.svg)](https://arxiv.org/abs/2505.20666)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue.svg)](https://github.com/XueqingZhou/Continuous-Time-Attention)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://aclanthology.org/2025.emnlp-main.1097.pdf)

Official implementation of **Continuous-Time Attention**, a PDE-guided formulation of self-attention that treats token interactions as trajectories of a continuous-time dynamical system governed by partial differential equations.

## Overview

Transformers achieve state-of-the-art performance on a wide range of sequence modeling tasks, but their quadratic attention complexity and discrete-layer parameterization make long-sequence modeling both expensive and difficult to analyze. 

**Continuous-Time Attention** addresses these challenges by:

- Modeling token interactions as solutions of partial differential equations (PDEs) in continuous time
- Enabling efficient and stable modeling of long-range dependencies in Transformers
- Providing better control over information propagation
- Improving stability for long-range dependencies
- Offering favorable computational properties for long sequences

This framework allows us to derive new attention mechanisms with a principled lens on how information flows across depth and positions.

## Links

- **Paper**: [arXiv:2505.20666](https://arxiv.org/abs/2505.20666)
- **PDF**: [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1097.pdf)
- **Code**: [GitHub Repository](https://github.com/XueqingZhou/Continuous-Time-Attention) (Coming Soon)
- **Project Page**: [https://xueqingzhou.github.io/Continuous-Time-Attention/](https://xueqingzhou.github.io/Continuous-Time-Attention/)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang-zhou-2025-continuous,
  title = {Continuous-Time Attention: {PDE}-Guided Mechanisms for Long-Sequence Transformers},
  author = {Zhang, Yukun and
    Zhou, Xueqing},
  editor = {Christodoulopoulos, Christos and
    Chakraborty, Tanmoy and
    Rose, Carolyn and
    Peng, Violet},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  month = nov,
  year = {2025},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2025.emnlp-main.1097/},
  doi = {10.18653/v1/2025.emnlp-main.1097},
  pages = {21654--21674},
  ISBN = {979-8-89176-332-6}
}
```

## Authors

- **Yukun Zhang**<sup>*</sup> - The Chinese University of Hong Kong
- **Xueqing Zhou**<sup>*</sup> - Fudan University

<sup>*</sup>Equal contribution

## Abstract

Transformers achieve state-of-the-art performance on a wide range of sequence modeling tasks, but their quadratic attention complexity and discrete-layer parameterization make long-sequence modeling both expensive and difficult to analyze. We propose **Continuous-Time Attention**, a PDE-guided formulation of self-attention that treats token interactions as trajectories of a continuous-time dynamical system governed by partial differential equations. This view allows us to derive new attention mechanisms with better control over information propagation, improved stability for long-range dependencies, and favorable computational properties for long sequences. We instantiate our framework in several long-sequence benchmarks, where Continuous-Time Attention attains competitive or superior performance to strong Transformer baselines while offering a principled lens on how information flows across depth and positions.

## License

This project is licensed under the terms specified in the repository.

## Acknowledgments

This template was borrowed from [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) which was adopted from the [Nerfies](https://nerfies.github.io) project page under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
