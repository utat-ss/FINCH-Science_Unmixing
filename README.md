# FINCH-Science_Unmixing
Code to explore hyperspectral unmixing algorithms and conduct extensive analysis to determine scientific requirements for the FINCH mission.

## Description
This repository, and specifically this release, has been used to analyze ground truth and atmospherically simulated hyperspectral data for the paper to be published by University of Toronto Aerospace Team, Space Systems Division, Science Team. 

## Hyperspectral Unmixing Models
To perform both unmixing and determine requirements, the Unmixing sub-team of the Science team has been exploring various different models and algoritmhs, based on classical machine learning, statistical learning, and deep learning. The unmixing models and algorithms are as follows:
- Classical Machine Learning-based:
    - K-Nearest-Neighbors (KNN)
    - Random Forest (RF)
    - Fully Constrained Direct Optimizatin via Least Squares (FCLS)
- Statistical Learning-based:
    - Frequentist Multiple Linear Regression (MLR)
    - Bayesian Linear Regression with Gaussian Priors, with linear basis function (BLR-G)
    - Bayesian Linear Regression with Dirichlet Priors, with linear basis function (BLR-D-Lin)
    - Bayesian Linear Regression with Dirichlet Priors, with non-linear basis functions:
        - With exponential basis (BLR-D-Exp)
        - With logarithmic basis (BLR-D-Log)
        - With quadratic basis (BLR-D-Quad)
        - With cubic basis (BLR-D-Cub)
- Deep Learning-based:
    - Multi-Layered Perceptron (MLP)
    - Convolutional Neural Network (CNN)
    - Fourier Neural Operator (FNO)

For the ISPRS publication, the following models are not explored. For more information about them, please check the main branch.
- FCLS, BNN, P-MLP

## Test Metric
There are widely accepted and applied testing metrics to evaluate hyperspectral unmixing models under certain data constraints or conditions:
- $R^{2}$ (coefficient of determination) of a linear regression on ground truth vs model prediction plots of each endmember class
- Class determination accuracy, classified by the confusion matrix of the model's predictions and ground truth data, binned into classes of endmember combinations
Given the more widespread use of $R^{2}$ value in the hyperspectral unmixing algorithm development literature, it was chosen as the sole metric for the development of FINCH's unmixing algorithms.

## Installation & Usage
### Installation
1. Clone the repository:
    ```bash
    git clone [https://github.com/utat-space/FINCH-Science_Unmixing.git](https://github.com/utat-space/FINCH-Science_Unmixing.git)
    cd FINCH-Science_Unmixing
2. Create a virtual environment based on Python 3.11
3. Install requirements:
    - Specific pytorch version first:
    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/<your_cuda_version_or_cpu>
    ```
    - Then, rest of the requirements:
    ```bash
    pip install -r requirements.txt
### Usage
There are various different model runs that can be found in the <runs> folder. Please check the notebooks and logs there. Given they are notebooks, you can simply re-run the notebooks to get the results.

## Citation
If you use this code or dataset in your research or works, please cite our upcoming ISPRS 2026 paper:

## License
The code is licensed under MIT License.

## Contact & Support
### Questions?
For questions regarding the codebase, the paper, or reproduction of results, please reach out to **Ege Artan**, the Science Lead of FINCH at the time of development for this code and paper:
* **LinkedIn:** [Ege Artan](https://www.linkedin.com/in/ege-artan/)
* **GitHub:** [@enatrage](https://github.com/enatrage)
You can also reach out to **Zoe Augspach**, the Unmixing Lead at the Science team:
* **LinkedIn:** [Zoe Augspach](https://www.linkedin.com/in/zoe-a-165055319/)
* **Github:** [@Zoe-Au](https://github.com/Zoe-Au)

### Found a Bug?
If you encounter any issues with the code or have feature requests, please check if the issue has already been reported. If not, feel free to open a new issue on our repository:
* [Report a Bug or Issue](https://github.com/utat-space/FINCH-Science_SyntheticData/issues)

## Project Structure
```text
.
├── data/
│   ├── simpler_data_rwc.csv
│   └── atmospheric/
├── runs/
│   └── dl/
├── src/
│   ├── dl
│   ├── mlr
│   └── rf
├── .gitignore
├── LICENSE
└── README.md
