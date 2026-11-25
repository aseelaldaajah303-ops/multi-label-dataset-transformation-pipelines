# Multi-Label Dataset Transformation Pipelines (`Advanced_Project.ipynb`)

A complete Python pipeline for processing **Multi-Label Classification** datasets and transforming them into **Single-Label Classification** problems using **4 core transformation strategies**, then evaluating and comparing them across **20 machine learning algorithms** on **Google Colab**.

---

## Project Overview

In multi-label datasets, a single instance can belong to multiple labels simultaneously. This project benchmarks four transformation strategies to evaluate how each approach affects model performance:

1. **MFL (Most Frequent Label):** Selects the most frequent label among the active labels for each instance (`yeast_single_label_mfl.csv`).
2. **LFL (Least Frequent Label):** Selects the least frequent label among the active labels for each instance (`yeast_single_label_lfl.csv`).
3. **LP (Label Powerset):** Combines each unique set of active labels into a new distinct single class (`yeast_single_label_lp.csv`).
4. **IGNORE (Filtered Multi-Label):** Keeps only single-label instances and discards multi-label instances (`yeast_single_label_ignore.csv`).

---

## Running on Google Colab

### 1. Open the Notebook and Set Up the Environment
1. Upload `Advanced_Project.ipynb` to **Google Colab** or open it directly from GitHub.
2. From the top menu, go to **Runtime > Change runtime type** and select **Python 3** (runs efficiently on either **CPU** or **T4 GPU**).
3. Ensure the required libraries are installed by running:

```python
!pip install -q pandas numpy scikit-learn
```

### 2. Upload / Load Dataset Files on Colab
Before running the transformation or training cells, you must load the dataset files into the Colab environment using one of the following methods:

#### Option 1: Direct Upload from Your Local Machine
Best for quick runs (uploaded files are cleared when the Colab session disconnects or restarts):

```python
from google.colab import files
import pandas as pd

uploaded = files.upload()  # Opens a file picker to upload CSV files from your computer
```

#### Option 2: Mount Google Drive (Recommended for Persistent Storage)
Keeps your datasets and generated output files saved persistently across sessions:

```python
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

# Update the path according to where your files are stored in Google Drive
df = pd.read_csv('/content/drive/MyDrive/multi_label_project/yeast_data.csv')
df.head()
```

#### Option 3: Clone the Repository Directly into Colab
```python
!git clone https://github.com/aseelaldaajah303-ops/multi-label-dataset-transformation-pipelines.git
%cd multi-label-dataset-transformation-pipelines
!ls -la
```

---

## Running the Current Dataset (`yeast`)

1. Make sure the `yeast` dataset files are present in your Colab working directory (either the raw dataset or the four pre-transformed files: `yeast_single_label_mfl.csv`, `yeast_single_label_lfl.csv`, `yeast_single_label_lp.csv`, `yeast_single_label_ignore.csv`).
2. Execute the notebook cells sequentially from top to bottom:
   - **Data loading and feature (`X`) / label (`Y`) preparation cell.**
   - **Four transformation cells** (`MFL`, `LFL`, `LP`, `IGNORE`) to generate or load each transformed dataset.
   - **20-algorithm training and evaluation cell** to compute **Accuracy**, **Precision**, and **Recall** for each transformation, followed by the final comparison summary.

---

## Switching to a New Dataset

If you want to apply this pipeline to a **new dataset** (e.g., medical, text, extracted image features, or any multi-label tabular dataset), you must convert it into the **same 4 transformations** before running the training and comparison cells:

### 1. Prepare the New Dataset Structure (`X` and `Y`)
Ensure your new `CSV` file contains:
- **Feature columns (`X`):** Numerical values (categorical or text columns must be encoded beforehand).
- **Label columns (`Y`):** Binary columns (`0` and `1`) indicating the absence or presence of each label per instance.

```python
import pandas as pd

# 1. Load the new dataset after uploading it to Colab
df_new = pd.read_csv("new_dataset.csv")

# 2. Specify the label column names (update this list to match your dataset)
label_cols = ["label_1", "label_2", "label_3", "label_4"]

# 3. Separate features from labels and handle missing values
X = df_new.drop(columns=label_cols).fillna(0)
Y = df_new[label_cols].fillna(0).astype(int)
```

### 2. Convert the New Dataset into the Four Transformations (MFL, LFL, LP, IGNORE)
Run the following script to generate and save the four transformed CSV files for your new dataset in the exact format expected by the rest of the notebook:

```python
import pandas as pd

def apply_transformations(X: pd.DataFrame, Y: pd.DataFrame):
    # Compute global frequency of each label across the dataset
    label_counts = Y.sum(axis=0)
    most_freq_order = list(label_counts.sort_values(ascending=False).index)
    least_freq_order = list(label_counts.sort_values(ascending=True).index)

    # 1) MFL: Select the most frequent active label for each row
    mfl_target, mfl_valid_idx = [], []
    for idx, row in Y.iterrows():
        active = [col for col in most_freq_order if row[col] == 1]
        if active:
            mfl_target.append(active[0])
            mfl_valid_idx.append(idx)

    df_mfl = X.loc[mfl_valid_idx].copy()
    df_mfl["target"] = mfl_target

    # 2) LFL: Select the least frequent active label for each row
    lfl_target, lfl_valid_idx = [], []
    for idx, row in Y.iterrows():
        active = [col for col in least_freq_order if row[col] == 1]
        if active:
            lfl_target.append(active[0])
            lfl_valid_idx.append(idx)

    df_lfl = X.loc[lfl_valid_idx].copy()
    df_lfl["target"] = lfl_target

    # 3) LP: Combine active labels per row into a single joint class string
    lp_target, lp_valid_idx = [], []
    for idx, row in Y.iterrows():
        active = [col for col in Y.columns if row[col] == 1]
        if active:
            lp_target.append("_".join(active))
            lp_valid_idx.append(idx)

    df_lp = X.loc[lp_valid_idx].copy()
    df_lp["target"] = lp_target

    # 4) IGNORE: Keep only instances with exactly one active label
    single_label_mask = (Y.sum(axis=1) == 1)
    df_ignore = X.loc[single_label_mask].copy()
    df_ignore["target"] = Y.loc[single_label_mask].idxmax(axis=1)

    return {
        "MFL": df_mfl,
        "LFL": df_lfl,
        "LP": df_lp,
        "IGNORE": df_ignore,
    }

transformed_datasets = apply_transformations(X, Y)

# Save the four transformed datasets for direct use in the training cells
transformed_datasets["MFL"].to_csv("new_single_label_mfl.csv", index=False)
transformed_datasets["LFL"].to_csv("new_single_label_lfl.csv", index=False)
transformed_datasets["LP"].to_csv("new_single_label_lp.csv", index=False)
transformed_datasets["IGNORE"].to_csv("new_single_label_ignore.csv", index=False)
```

### 3. Update File Paths in the Training Cells
After generating the four transformed files for the new dataset, update the file paths dictionary in `Advanced_Project.ipynb` to point to the new files:

```python
datasets_paths = {
    "MFL": "new_single_label_mfl.csv",
    "LFL": "new_single_label_lfl.csv",
    "LP": "new_single_label_lp.csv",
    "IGNORE": "new_single_label_ignore.csv",
}
```

---

## Important Technical Notes for Future Developers

- **Target Column Name (`target`):** Ensure the generated target column matches the column name expected during `train_test_split` and model training.
- **Stratified Splitting in `LP`:** In the **Label Powerset** transformation, rare label combinations (appearing only once) may occur and cause an error when using `stratify=y` in `train_test_split`. If this happens, filter out classes with fewer than 2 samples or disable `stratify`.
- **Sample Size Reduction in `IGNORE`:** If most instances in your new dataset have multiple active labels, the remaining dataset size under `IGNORE` will decrease significantly.
- **Evaluated Algorithms (20 Models):** The pipeline evaluates each transformation across 20 classifiers—including Linear Models, KNN, Decision Trees, Ensemble Methods (Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost), SVM, and Neural Networks (MLP)—and outputs a comprehensive comparison table (`Accuracy`, `Precision`, `Recall`).

---

## Benchmark Results Summary (`yeast` Dataset)

- **Best Overall Transformation (Average):** `MFL (Most Frequent Label)` with a mean accuracy of **30.07%**.
- **Best Overall Combination:** `MFL` + `15. Extra Trees Ensemble` achieving **37.19% Accuracy** (`Precision: 0.2458`, `Recall: 0.3719`).
