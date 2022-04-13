# DeepMTS: a Deep Multi-Task Survival model (DeepMTS) for joint survival prediction and tumor segmentation
In this study, we propose a 3D end-to-end Deep Multi-Task Survival model (DeepMTS) for joint survival prediction and tumor segmentation. Our novelty is the introduction of a hard-sharing segmentation backbone to guide the extraction of local features related to the primary tumors, which reduces the interference from non-relevant background information. In addition, we also introduce a cascaded survival network to capture the prognostic information existing out of primary tumors and further leverage the global tumor information (e.g., tumor size, shape, and locations) derived from the segmentation backbone. Our experiments with two clinical datasets demonstrate that our DeepMTS can consistently outperform traditional radiomics-based survival prediction models and existing deep survival models.
**For more details, please refer to our paper. [[arXiv](https://arxiv.org/abs/2109.07711)]**

## Workflow
![workflow](https://github.com/MungoMeng/DeepMTS/blob/master/Figure/Workflow.png)

## Publication
If this repository helps your work, please kindly cite our papers as follows:

* **Mingyuan Meng, Bingxin Gu, Lei Bi, Shaoli Song, David Dagan Feng, Jinman Kim, "DeepMTS: Deep Multi-task Learning for Survival Prediction in Patients with Advanced Nasopharyngeal Carcinoma using Pretreatment PET/CT," arXiv:2109.07711 (under review). [[arXiv](https://arxiv.org/abs/2109.07711)]**
* **Mingyuan Meng, Yige Peng, Lei Bi, Jinman Kim, "Multi-task Deep Learning for Joint Tumor Segmentation and Outcome Prediction in Head and Neck Cancer," In: Andrearczyk, V., Oreiller, V., Hatt, M., Depeursinge, A. (eds) Head and Neck Tumor Segmentation and Outcome Prediction. HECKTOR 2021. Lecture Notes in Computer Science, vol 13209. Springer, Cham. [[Springer](https://link.springer.com/chapter/10.1007/978-3-030-98253-9_15)]**
