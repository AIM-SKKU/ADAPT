<div align="center">
 <h1>Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment</h1>
<div>
    <a href='https://youjia-zhang.github.io/' target='_blank'>Youjia Zhang</a><sup>1</sup>&emsp;
    <a href='https://youngryan1993.github.io/homepage/' target='_blank'>Youngeun Kim</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/view/ygchoi' target='_blank'>Young-Geun Choi</a><sup>1</sup>&emsp;
    <a href='https://redleaf-kim.github.io/' target='_blank'>Hongyeob Kim</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?view_op=list_works&hl=en&user=WQfYo0MAAAAJ' target='_blank'>Huiling Liu </a><sup>1</sup>&emsp;
    <a href='https://www.csehong.com/' target='_blank'>Sungeun Hong</a><sup>1</sup>
</div>

 <br>
<div>
    <sup>1</sup>Sungkyunkwan University, &nbsp;&nbsp; <sup>2</sup>Yale University
</div>

<div align="center">
  <h4>NeurIPS 2025</h4>
  <a href=" "><strong>[Paper](https://arxiv.org/pdf/2508.15568)</strong></a>
 &nbsp;&nbsp;
  <a href=" "><strong>[Project Page](https://aim-skku.github.io/ADAPT/)</strong></a>
</div>
</div>


<div align="left">

## Abstract
Test-time adaptation (TTA) enhances the zero-shot robustness under distribution shifts by leveraging unlabeled test data during inference. Despite notable advances, several challenges still limit its broader applicability.
First, most methods rely on backpropagation or iterative optimization, which limits scalability and hinders real-time deployment. 
Second, they lack explicit modeling of class-conditional feature distributions. This modeling is crucial for producing reliable decision boundaries and calibrated predictions, but it remains underexplored due to the lack of both source data and supervision at test time.
In this paper, we propose an **A**dvanced **D**istribution-**A**ware and back**P**ropagation-free **T**est-time adaptation (**ADAPT**) method. We reframe TTA as a Gaussian probabilistic inference task by modeling class-conditional likelihoods using gradually updated class means and a shared covariance matrix. This enables closed-form, training-free inference. To correct potential likelihood bias, we introduce lightweight regularization guided by CLIP priors and a historical knowledge bank. ADAPT requires no source data, no gradient updates, and no full access to target data, supporting both online and transductive settings.
Extensive experiments across diverse benchmarks demonstrate that our method achieves state-of-the-art performance under a wide range of distribution shifts with superior scalability and robustness. 


<br>
<br>
<img src="images/Overview.png"  width="100%" height="100%">

<div align="left">
Fig. 1: Overview of Online ADAPT. We perform TTA by modeling class-conditional feature distributions under a Gaussian assumption with shared covariance across classes. Class means are initialized from CLIP prototypes and refined using high-confidence samples in fixed-size per-class knowledge banks. To avoid error accumulation, the current test sample is excluded from updates. Predictions are made via a closed-form, backpropagation-free solution. In the transductive setting, the knowledge bank is built using the top-L most confident samples per class from the full test set.


---
### Requirements
```python
Python >= 3.10  
PyTorch == 2.5.1
```
## Datasets
We evaluate our method under three tasks:
### Task1: Natural Distribution Shifts
ImageNet, ImageNet-V2，ImageNet-A, ImageNet-R, ImageNet-Sketch  
 
### Task12: Corruption Robustness
ImageNet-C

### Task13: Cross-Dataset Generalization
Flower102, OxfordPets, SUN397, DTD, Food101, StanfordCars, Aircraft, UCF101, EuroSAT, Caltech101 


Please refer to [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT) for more details on data.

<div align="left">

## Using ADAPT
Online Scenario:

```
bash ADAPT_online_TTA.sh
```
Transductive Scenario

```
bash ADAPT_Transductive_TTA.sh
```


### Citation

If you find this work useful, please consider citing it.

```
@inproceedings{zhangbackpropagation,
  title={Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment},
  author={Zhang, Youjia and Kim, Youngeun and Choi, Young-Geun and Kim, Hongyeob and Liu, Huiling and Hong, Sungeun},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

# Acknowledgements
We thank the authors of [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp), [TPT](https://github.com/azshue/TPT) and [AWT](https://github.com/MCG-NJU/AWT) for their open-source implementation and instructions on data preparation.

