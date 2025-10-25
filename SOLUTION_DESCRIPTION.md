# Problem

Two classification models were built for medical data (`0 = healthy`, `1 = diseased`) to , each designed for a different diagnostic purpose:

- **Model A (Jupyter Notebook):**  
  Designed for **screening** scenarios, where the goal is to catch as many potential disease cases as possible. It’s suitable for early detection, even if it occasionally flags healthy individuals incorrectly.

- **Model B (Python file):**  
  Designed for **confirming health status**, where the goal is to be confident that someone classified as healthy truly is healthy. It’s useful in follow-up assessments or low-risk populations, where false alarms should be minimized.

> This comparison helps select the appropriate model depending on the medical context:  
> - **Model A** for broad disease detection  
> - **Model B** for reliable health confirmation

### Type:
- **SVM (CVM kernel)**  
  Chosen for its ability to handle non-linear relationships and small, high-dimensional datasets. Particularly effective when features like shape descriptors (e.g., inertia tensors) are involved.

### Generator:
- **ADASYN**  
  Used to synthetically generate minority class samples in complex regions of the feature space. This helps improve class balance and recall for diseased cases without oversimplifying decision boundaries.
### Scaler:
- **SimpleScaler**
  Applied before oversampling to ensure that all features contribute equally to distance-based calculations. This is especially important for SVM and ADASYN, which rely on feature geometry. Scaling improves convergence and prevents bias from dominant features.

  
### Hyperparameter Tuning:
- **GridSearchCV**  
  Applied to systematically explore combinations of model parameters and select the best configuration based on cross-validation performance.

### Model Evaluation:
- **Result Table**  
  Includes precision, recall, F1-score, and support for both classes to assess diagnostic reliability.
- **ROC Curve**  
  Visualizes the trade-off between sensitivity and specificity across thresholds.
- **PR Curve**  
  Highlights performance in imbalanced settings, focusing on precision-recall dynamics.

---

## Why these choices fit the data

The dataset contains geometric and pixel-based features extracted from medical images, including:
- **Cardiothoracic Ratio (CTR)** and organ dimensions  
- **Inertia tensors** (`xx`, `yy`, `xy`, `normalized_diff`) — describing shape, elongation, and rotation of heart and lung regions  
- **Area and perimeter metrics** — useful for capturing structural abnormalities

These features are non-linear and spatially complex, making SVM with a non-linear kernel a strong candidate. ADASYN complements this by enriching the minority class (diseased) in regions where decision boundaries are harder to define. Together, they support robust classification in a medically relevant, imbalanced dataset.


# Evaluation Summary

## Example values:

### Model A (Jupyter Notebook)
- **Cross-validation mean score**: 0.661  
- **Standard deviation**: 0.199  
- **Recall (healthy)**: 0.667  
- **Recall (diseased)**: 0.821  
- **Accuracy**: 0.784  
- **Macro F1-score**: 0.726

### Model B (Python file)
- **Cross-validation mean score**: 0.720  
- **Standard deviation**: 0.183  
- **Recall (healthy)**: 0.556  
- **Recall (diseased)**: 0.893  
- **Accuracy**: 0.811  
- **Macro F1-score**: 0.733

# Conclusion

Model A is better suited for general screening, where missing a disease is risky.  
Model B is better for confirming health, where false positives should be minimized.

Both models perform reasonably well given the small size of the dataset, showing acceptable generalization and diagnostic value despite limited training data.
