# LoRA-Mini: Adaptation Matrices Decomposition and Selective Training

This repository contains the official implementation of the paper: [LoRA-Mini : Adaptation Matrices Decomposition and Selective Training](https://arxiv.org/abs/2411.15804) **(Accepted at AAAI CoLoRAI Workshop)**

## Introduction

The recent advancements in Large Language Models (LLMs) have highlighted the need for efficient fine-tuning methods. While Low-Rank Adaptation (LoRA) has been a significant step towards Parameter Efficient Fine-Tuning, it still presents storage challenges.

This paper introduces
**LoRA-Mini**, an optimized adaptation of LoRA that enhances parameter efficiency by decomposing low-rank matrices and applying selective training. Our approach splits the low-rank matrices into four parts, with only the two inner matrices being trainable. This method achieves up to a **20x** reduction in trainable parameters compared to standard LoRA while maintaining comparable performance.

![image](https://github.com/user-attachments/assets/4ccbc95a-021b-4df9-8ba0-9211be82d70a)


## Installation

To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/RajdeepAher/lora-mini.git
cd lora-mini
pip install -r requirements.txt
```

## Datasets

## Results


### Reproducing Results

## Citation

If you find our work useful, please cite our paper:

```
@misc{singh2024loraminiadaptationmatrices,
      title={LoRA-Mini : Adaptation Matrices Decomposition and Selective Training}, 
      author={Ayush Singh and Rajdeep Aher and Shivank Garg},
      year={2024},
      eprint={2411.15804},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.15804}, 
}
```

## Contact
For any questions or suggestions, please feel free to open an issue or contact the authors:

- Ayush Singh: ayush_s@mt.iitr.ac.in 
- Rajdeep Aher: aher_rp@ma.iitr.ac.in 
- Shivank Garg: shivank_g@mfs.iitr.ac.in 



