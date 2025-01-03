# XGBoost Heart Disease Prediction

Recreating the XGBoost library from scratch and replicating results from the article *An Optimized XGBoost-Based Diagnostic System for Effective Prediction of Heart Disease* using heart disease data.

This project is part of the "Advanced Machine Learning" course at ENSAE, taught by Austin Stromme.

## Table of Contents
- [Context](#context)
- [Installation](#installation)
- [Data](#data)
- [Results](#results)
- [References](#references)
- [License](#license)

## Context
Heart disease is one of the leading causes of mortality worldwide. Early and accurate detection can significantly improve patient outcomes and reduce fatalities. This project aims to implement the XGBoost algorithm from scratch and replicate the results presented in the article *An Optimized XGBoost-Based Diagnostic System for Effective Prediction of Heart Disease*. The dataset used is publicly available and contains clinical information collected from patients.

The project also draws theoretical foundations and inspiration from the article *XGBoost: A Scalable Tree Boosting System*. By developing XGBoost from the ground up, this project seeks to provide insights into its mechanics while validating its effectiveness in predicting heart disease.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MoutetMaxime/xgboost-heart-disease.git
   cd xgboost-heart-disease
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data
The heart disease dataset can be accessed at the following link:
[Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Results
- Metrics: Precision, Recall, AUC
- Visualizations: Confusion matrix, ROC curves (available in the `results/` folder).

## References
```bibtex
@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Tianqi Chen and Carlos Guestrin},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016},
  organization={ACM}
}

@article{optimized_xgboost_heart,
  title={An Optimized XGBoost-Based Diagnostic System for Effective Prediction of Heart Disease},
  author={Authors Unknown},
  journal={Journal of Biomedical Informatics},
  year={2020},
  volume={103},
  pages={103377},
  doi={10.1016/j.jbi.2020.103377}
}
```

