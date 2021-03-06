# DeepMTS: Deep Multi-Task Survival model for joint survival prediction and tumor segmentation
In this study, we propose a 3D end-to-end Deep Multi-Task Survival model (DeepMTS) for joint survival prediction and tumor segmentation. Our novelty is the introduction of a hard-sharing segmentation backbone to guide the extraction of local features related to the primary tumors, which reduces the interference from non-relevant background information. In addition, we also introduce a cascaded survival network to capture the prognostic information existing out of primary tumors and further leverage the global tumor information (e.g., tumor size, shape, and locations) derived from the segmentation backbone. Our experiments demonstrate that our DeepMTS can consistently outperform traditional radiomics-based survival models and existing deep survival models.  
**For more details, please refer to our paper. [[IEEE](https://ieeexplore.ieee.org/document/9794806)] [[arXiv](https://arxiv.org/abs/2109.07711)]**

## Overview
![workflow](https://github.com/MungoMeng/Survival-DeepMTS/blob/master/Figure/Overview.png)

## Publication
If this repository helps your work, please kindly cite our papers:
* **Mingyuan Meng, Bingxin Gu, Lei Bi, Shaoli Song, David Dagan Feng, Jinman Kim, "DeepMTS: Deep Multi-task Learning for Survival Prediction in Patients with Advanced Nasopharyngeal Carcinoma using Pretreatment PET/CT," IEEE Journal of Biomedical and Health Informatics, 2022, doi: 10.1109/JBHI.2022.3181791. [[IEEE](https://ieeexplore.ieee.org/document/9794806)] [[arXiv](https://arxiv.org/abs/2109.07711)]**
* **Mingyuan Meng, Yige Peng, Lei Bi, Jinman Kim, "Multi-task Deep Learning for Joint Tumor Segmentation and Outcome Prediction in Head and Neck Cancer," Head and Neck Tumor Segmentation and Outcome Prediction (HECKTOR 2021), pp. 160-167, 2021, doi: 10.1007/978-3-030-98253-9_15. [[Springer](https://link.springer.com/chapter/10.1007/978-3-030-98253-9_15)]**
