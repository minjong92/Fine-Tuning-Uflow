# UFlow Model Training and Pipeline Improvements

## Enhancements to Training and Inference

### 1. Data Augmentation for Training
- **Issue:**  
  Previously, both training and inference utilized the same `datamodule.py` file to load data. This caused unintended data augmentation during inference, which is not ideal.

- **Solution:**  
  - Separated data loading logic into two distinct files:
    - **`train_datamodule.py`**: Includes data augmentation for training.
    - **`test_datamodule.py`**: Excludes data augmentation for inference.
  - **Benefit:**  
    Training benefits from data augmentation, while inference remains unaffected, ensuring improved reliability of results.

---

### 2. Confidence Score-Based OK/NG Classification
- **New Feature:**  
  A **confidence score threshold** is implemented during inference to classify results as either **OK** or **NG**.

- **Benefit:**  
  Provides a clear and consistent decision-making process based on model predictions.

---

## Key Insights from UFlow Model Training

### 1. Requirement for Defective Test Data
- **Challenge:**  
  The UFlow model requires at least one defective test sample to calculate the loss function and update model weights.

- **Problem:**  
  Defective data is often scarce or unavailable in anomaly detection tasks.

---

# Utilization of Artificially Generated Defective Data

## Solution
In the absence of defective data, artificially generated defective data was created using normal samples. This artificially generated data was used as test data.

## Results
- Experimental results demonstrated no significant performance differences between:
  - A model trained using real defective test data.  
  - A model trained using artificially generated defective data.

- **Key Findings:**
  - Real defective data used for testing lacked distinct defect features, making it hard to differentiate from normal data.
  - Artificially generated defective data had clear and distinguishable defect characteristics.
  - Performance evaluation with artificially generated data showed a high **AUC (Area Under the Curve)** score.
  - Models trained with artificially generated defective data outperformed those trained with real defective data in actual testing.

## Conclusion
Artificially generated defective data can effectively substitute real defective data.

## Benefits
- Enables efficient model training and evaluation even in environments with limited defective data.  
- Enhances model performance by leveraging clear and distinguishable defective data.

---

# Experiment Results

### 1. Comparison of Artificially Generated and Real Defective Images
<table>
  <tr>
    <td align="center"><b>Artificially Generated Defective Image</b></td>
    <td align="center"><b>Real Defective Image</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8a7686ad-1239-4588-b024-5c0d21bb8cc7" alt="Artificially Generated Image" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/a7f535b1-59e1-423e-a346-d21617752290" alt="Real Defective Image" width="400"></td>
  </tr>
</table>

---

### 2. Comparison of Artificially Generated and Real Defective Mask Images
<table>
  <tr>
    <td align="center"><b>Artificially Generated Defective Mask</b></td>
    <td align="center"><b>Real Defective Mask</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9918492a-e407-4840-9019-49228a43f0b5" alt="Artificially Generated Mask" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/59cb52a5-eaaf-45de-819e-c516b2ad00f1" alt="Real Defective Mask" width="400"></td>
  </tr>
</table>

---

### 3. Defect Detection Heatmaps
<table>
  <tr>
    <td align="center"><b>Heatmap from Real Defective Images</b></td>
    <td align="center"><b>Heatmap from Artificially Generated Defective Images</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/95d0b6b4-bc29-427a-86ed-f8a8d9649571" alt="Real Defective Heatmap" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/ec42d5f0-c3ab-47a7-94a5-c89edb4e5b37" alt="Artificially Generated Heatmap" width="400"></td>
  </tr>
</table>

---

### 4. Normal Product Detection Heatmaps
<table>
  <tr>
    <td align="center"><b>Heatmap from Real Defective Images</b></td>
    <td align="center"><b>Heatmap from Artificially Generated Defective Images</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2148b5b2-3e59-4225-acb8-574a5dfdddcc" alt="Real Defective Normal Detection Heatmap" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/628684c8-708e-4be5-9d05-0791188faa24" alt="Artificially Generated Normal Detection Heatmap" width="400"></td>
  </tr>
</table>

---

## Summary of Updates

### 1. **Data Augmentation**
   - Separated logic for training (`train_datamodule.py`) and inference (`test_datamodule.py`).
   - Ensures augmentation benefits training without affecting inference.

### 2. **Confidence Score Classification**
   - Introduced OK/NG classification based on confidence scores for consistent and reliable inference.

### 3. **Defective Data Handling**
   - Artificially generated defective data successfully replaces real defective data, enabling robust model performance and efficient training in data-limited environments.
