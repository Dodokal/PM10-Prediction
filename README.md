# PM10 Prediction Models Repository

This repository provides a reproducible framework for predicting PM10 air pollution levels using three machine learning approaches:
- **Random Forest (RF)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (NB)**

All scripts are implemented in R and include data preprocessing, model training with cross-validation, hyperparameter tuning, evaluation (including ROC curves), and spatial prediction mapping. The repository is designed to facilitate replication and validation of the modeling approaches described in the corresponding publication.

---

## Requirements

Ensure you have **R** (version 4.x or later) installed. The following R packages are required (they can be installed via `install.packages()` if not already present):

- terra
- raster
- plyr
- dplyr
- RStoolbox
- RColorBrewer
- ggplot2
- sp
- caret
- doParallel
- e1071
- pdftools
- gridExtra
- GGally
- doSNOW
- klaR
- pROC

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pm10-prediction-models.git
    ```
2. **Open the project in RStudio** (or your preferred R IDE).

3. **Set up your working environment**:
   - Ensure that the `data` folder (with `S2.csv`, `Levels_key.csv`, and the `rasters` subfolder) is in the root of the repository.
   - Create an `output` folder in the repository root if it does not already exist.

4. **Install necessary R packages**:
    ```r
    install.packages(c("terra", "raster", "plyr", "dplyr", "RStoolbox", "RColorBrewer",
                       "ggplot2", "sp", "caret", "doParallel", "e1071", "pdftools",
                       "gridExtra", "GGally", "doSNOW", "klaR", "pROC"))
    ```

## Usage

Each model (RF, KNN, NB) has its own R script located in the `scripts` directory:

- **Random Forest (RF)**
  - Run `RF_model.R` to preprocess data, train the Random Forest model, evaluate performance, and generate prediction maps.
- **K-Nearest Neighbors (KNN)**
  - Run `KNN_model.R` to execute data preparation, train the KNN model, assess accuracy, and create spatial predictions.
- **Naive Bayes (NB)**
  - Run `NB_model.R` for the complete pipeline: data cleaning, visualization, hyperparameter tuning, model evaluation (including ROC curves), and mapping predictions.

Each script is self-contained and uses relative file paths, ensuring that the pipeline can be replicated across different environments.

## Replication

For full reproducibility:
- Use the fixed seeds provided in the scripts (e.g., `set.seed(123)` for data splitting and `set.seed(849)` for model training).
- Refer to the output PDFs and GeoTIFF maps generated in the `output` folder to compare against your results.
- Follow the step-by-step instructions provided in each script’s comments to understand the data preprocessing, model tuning, and spatial prediction processes.
