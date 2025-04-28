# DNP3_IDS_Project
Machine Learning-based Intrusion Detection System (IDS) for the DNP3 protocol using SVM and PCA. Achieves 99% accuracy on the DNP3 Intrusion Detection Dataset. Academic project for TU Dublin.

# DNP3 Intrusion Detection System using SVM and PCA

## Description

This repository contains the code and resources for developing a Machine Learning-based Intrusion Detection System (IDS) specifically tailored for the DNP3 communication protocol, commonly used in Industrial Control Systems (ICS) and SCADA environments. The project explores different ML approaches and finds that a Support Vector Machine (SVM) combined with Principal Component Analysis (PCA) achieves high accuracy on a specialized DNP3 dataset.

This project was undertaken as part of academic work at the School of Enterprise Computing and Digital Transformation, TU Dublin.

## Motivation & Problem

The DNP3 protocol, while widely adopted in critical infrastructure, lacks inherent security mechanisms, making ICS networks vulnerable to cyberattacks. Effective and reliable detection of malicious activity within DNP3 traffic is crucial for maintaining operational safety and security. This project aims to address this need by developing and evaluating an ML-based IDS.

## Dataset

This project utilizes the publicly available **"DNP3 Intrusion Detection Dataset"**.

*   **Citation:** Radoglou-Grammatikis, P., Kelli, V., Lagkas, T., Argyriou, V. and Sarigiannidis, P. (2022). *DNP3 Intrusion Detection Dataset*. Federated Research Data Repository. Available at: https://doi.org/10.21227/s7h0-b081.
*   **Contents:** The dataset contains labelled network flow statistics derived from simulated DNP3 communications in an ICS testbed. It includes samples of normal traffic and 9 specific DNP3 attack types (e.g., Replay, Cold Restart, MITM-DoS, Disable Unsolicited).
*   **Source:** The dataset originates from ITHACA - University of Western Macedonia.
*   **Access:** You will need to download the dataset separately from the source linked above (or other official distribution points) and place the relevant CSV files (`Custom_DNP3_Parser_Training_Balanced.csv`, `Custom_DNP3_Parser_Testing_Balanced.csv`) in an accessible location (e.g., an `input` directory or as specified in the notebooks).

## Methodology

### 1. Preprocessing
A rigorous preprocessing pipeline was applied to the dataset:
*   **Cleaning:** Removal of irrelevant columns (indices, IDs, timestamps, IPs).
*   **Encoding:** Numerical encoding of categorical features (`firstPacketDIR`, target labels).
*   **Imputation:** Handling missing values using Median Imputation.
*   **Scaling:** Feature normalization using Standard Scaling.
*   **Dimensionality Reduction:** Principal Component Analysis (PCA) applied, retaining 15 components capturing >95% of the variance.
*   **Splitting:** Data divided into training, validation, and test sets using stratified sampling.

### 2. Model Development & Evaluation
*   **Initial Exploration:** Artificial Neural Networks (ANNs) were initially tested but showed poor generalization on this dataset.
*   **Pivot to Traditional ML:** Compared Random Forest, Gradient Boosting, XGBoost, and Support Vector Machine (SVM).
*   **SVM Optimization:** SVM showed the best preliminary results. Hyperparameters were tuned using `GridSearchCV` (5-fold CV). Optimal parameters found: Kernel='rbf', C=50, Gamma=0.1.
*   **Evaluation:** The final optimized SVM model was evaluated on the unseen test set using standard metrics (Accuracy, Precision, Recall, F1-Score) and statistical significance tests (McNemar's).

## Results

The final optimized SVM model achieved excellent performance on the held-out test set:
*   **Accuracy:** 99.00%
*   **Macro F1-Score:** 0.99
*   **Weighted F1-Score:** 0.99

The model demonstrated high and balanced performance across nearly all DNP3 attack classes present in the test data. Statistical tests confirmed its significant superiority over the other traditional ML models evaluated in this study.

Detailed results, classification reports, and visualizations (like confusion matrices) can be found in the associated Jupyter notebooks (specifically `final_dnp3-project-notebook.ipynb`) and potentially in the project thesis/poster if available.

## Usage / Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MoHaroon777/DNP3_IDS_Project.git
    cd ./DNP3_IDS_Project
    ```
2.  **Set up Environment:** A Python environment with standard data science libraries is required. Key libraries include:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`
    *   `seaborn`
    *   `jupyter` (to run notebooks)
    You can typically install these using pip or conda:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    # or using conda
    # conda install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Download Dataset:** Obtain the "DNP3 Intrusion Detection Dataset" (see Dataset section above) and place the required CSV files in the expected location (you might need to create an `input/` directory or adjust paths in the notebooks).
4.  **Run Notebooks:** Launch Jupyter Notebook or JupyterLab and run the notebooks, particularly `final_dnp3-project-notebook.ipynb`, to see the complete workflow from data loading to final model evaluation.

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project or dataset in your research, please cite the original dataset paper:

Radoglou-Grammatikis, P., Kelli, V., Lagkas, T., Argyriou, V. and Sarigiannidis, P. (2022). DNP3 Intrusion Detection Dataset. Federated Research Data Repository. https://doi.org/10.21227/s7h0-b081

You may also cite this repository if appropriate for the code implementation.
