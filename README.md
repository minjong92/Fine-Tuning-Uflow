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
