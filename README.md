# <img src="logo.png" alt="Logo" width="100" height="45"/>: Target Sequence-Conditioned Generation of Peptide Binders via Masked Language Modeling 
In this work, we introduce **PepMLM**, a purely target sequence-conditioned de novo generator of linear peptide binders. By employing a novel masking strategy that uniquely positions cognate peptide sequences at the terminus of target protein sequences, PepMLM tasks the state-of-the-art ESM-2 pLM to fully reconstruct the binder region, achieving low perplexities matching or improving upon previously-validated peptide-protein sequence pairs. After successful *in silico* benchmarking with AlphaFold-Multimer, we experimentally verify PepMLM's efficacy via fusion of model-derived peptides to E3 ubiquitin ligase domains, demonstrating endogenous degradation of target substrates in cellular models. In total, PepMLM enables the generative design of candidate binders to any target protein, without the requirement of target structure, empowering downstream programmable proteome editing applications.


Check out our [manuscript](https://arxiv.org/abs/2310.03842) on the *arXiv*!

- HuggingFace: [Link](https://huggingface.co/TianlaiChen/PepMLM-650M)
- Demo: HuggingFace Space Demo [Link](https://huggingface.co/spaces/TianlaiChen/PepMLM).
- Colab Notebook: [Link](https://colab.research.google.com/drive/1u0i-LBog_lvQ5YRKs7QLKh_RtI-tV8qM?usp=sharing)
- Nature Biotechnology: [Link](https://www.nature.com/articles/s41587-025-02761-2)

# Load Model
```
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("TianlaiChen/PepMLM-650M")
model = AutoModelForMaskedLM.from_pretrained("TianlaiChen/PepMLM-650M")
```
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
Contact: pranam@seas.upenn.edu

# Citation
```
@article{chen2025pepmlm,
  title={Target sequence-conditioned design of peptide binders using masked language modeling},
  author={Chen, Tianlai and Quinn, Zachary and Dumas, Madeleine and Peng, Christina and Hong, Lauren and Lopez-Gonzalez, Moises and Mestre, Alexander and Watson, Rio and Vincoff, Sophia and Zhao, Lin and Wu, Jianli and Stavrand, Audrey and Schaepers-Cheu, Mayumi and Wang, Tian Zi and Srijay, Divya and Monticello, Connor and Vure, Pranay and Pulugurta, Rishab and Pertsemlidis, Sarah and Kholina, Kseniia and Goel, Shrey and DeLisa, Matthew P. and Chi, Jen-Tsan Ashley and Truant, Ray and Aguilar, Hector C. and Chatterjee, Pranam},
  journal={Nature Biotechnology},
  year={2025}
}
```
