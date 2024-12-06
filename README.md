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

### 2. Using Artificially Generated Defective Data
- **Solution:** Assuming no defective data is available:
  - Artificially generated defective data was created from normal samples.
  - This artificially created data was used as test data.
- **Result:** Experimental results showed no significant performance difference between:
  - The model trained using real defective test data.
  - The model trained with artificially generated defective data.

### 3. Conclusion
- Artificially generated defective data can effectively substitute real defective data.
- **Benefit:** Enables efficient model training and evaluation, even in environments where defective data is limited.

---

## Summary of Updates
1. **Data Augmentation:**
   - Separate logic for training (`train_datamodule.py`) and inference (`test_datamodule.py`).
   - Ensures augmentation benefits training without affecting inference.
2. **Confidence Score Classification:**
   - Added OK/NG classification based on confidence scores for reliable inference.
3. **Defective Data Handling:**
   - Artificially generated defective data proved effective as a substitute for real defective data, enabling robust model performance.
