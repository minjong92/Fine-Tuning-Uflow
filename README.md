To enhance model training performance, data augmentation has been added to the pipeline.
Previously, both training and inference utilized the same datamodule.py file to load data. However, this caused unintended data augmentation during inference, which is not ideal.

To address this issue:

Separated the data loading logic into two distinct files:
train_datamodule.py: Includes data augmentation for training.
test_datamodule.py: Excludes data augmentation for inference.
This ensures that training benefits from augmentation while inference remains unaffected, improving the overall reliability of the results.

Added Confidence Score-Based OK/NG Classification

During inference, a confidence score threshold is now implemented to determine the result as either OK or NG.
This ensures a clear and consistent decision-making process based on model predictions.
These updates improve both training robustness and inference reliability, providing a well-defined separation of concerns and better performance metrics.

To train the UFlow model, at least one defective test sample is required. This is because the loss function is calculated based on the test data to update the model's weights. However, the primary purpose of anomaly detection is to identify outliers in situations where defective data is scarce or unavailable.

Thus, assuming the absence of defective data, artificially generated defective data was created from normal samples and used as test data. Experimental results showed no significant performance difference between the model trained using real defective test data and the model trained with artificially generated defective data.

These findings suggest that artificially generated defective data can serve as an effective substitute for real defective data, enabling efficient model training and evaluation even in environments where defective data is limited.
