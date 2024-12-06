# UFlow Model Training and Pipeline Improvements

## Enhancements to Training and Inference

### 1. Data Augmentation for Training
- **Issue:** Previously, both training and inference utilized the same `datamodule.py` file to load data. This caused unintended data augmentation during inference, which is not ideal.
- **Solution:**
  - Separated data loading logic into two distinct files:
    - **`train_datamodule.py`**: Includes data augmentation for training.
    - **`test_datamodule.py`**: Excludes data augmentation for inference.
  - **Benefit:** Training benefits from data augmentation, while inference remains unaffected, ensuring improved reliability of results.

---

### 2. Confidence Score-Based OK/NG Classification
- **New Feature:** During inference, a **confidence score threshold** is implemented to classify results as either **OK** or **NG**.
- **Benefit:** Provides a clear and consistent decision-making process based on model predictions.

---

## Key Insights from UFlow Model Training

### 1. Requirement for Defective Test Data
- **Challenge:** The UFlow model requires at least one defective test sample to calculate the loss function and update model weights.
- **Problem:** In anomaly detection, defective data is often scarce or unavailable.

# Utilization of Artificially Generated Defective Data

## Solution
In the absence of defective data, artificially generated defective data was created using normal samples. This artificially generated data was used as test data.

## Results
The experimental results demonstrated no significant performance differences between the following:

- A model trained using real defective test data.  
- A model trained using artificially generated defective data.

Additionally, the defective data previously used for testing exhibited minimal differences from normal data and lacked distinct defect features. In contrast, the artificially generated test data was designed to possess clear defect characteristics, making it significantly distinguishable from normal data. Performance evaluation based on these clearly defined defective data resulted in a high **AUC (Area Under the Curve)** score. Furthermore, in actual testing, models trained with artificially generated defective data outperformed those trained with real defective data.

## Conclusion
Artificially generated defective data can effectively substitute real defective data.

## Benefits
- Enables efficient model training and evaluation even in environments with limited defective data.  
- Leveraging clearly defined defective data helps further enhance model performance.

#Experiment result 

---

## Summary of Updates
1. **Data Augmentation:**
   - Separate logic for training (`train_datamodule.py`) and inference (`test_datamodule.py`).
   - Ensures augmentation benefits training without affecting inference.
2. **Confidence Score Classification:**
   - Added OK/NG classification based on confidence scores for reliable inference.
3. **Defective Data Handling:**
   - Artificially generated defective data proved effective as a substitute for real defective data, enabling robust model performance.
