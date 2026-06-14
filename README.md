# Diabetes Prediction using ANN Classification

## Overview

This project implements a comprehensive machine learning solution using Artificial Neural Networks (ANN) for binary classification to predict the likelihood of diabetes in patients. The model uses various health indicators and patient data to make accurate predictions, helping in early detection and intervention.

## Project Description

Diabetes is one of the most prevalent chronic diseases worldwide. Early prediction and diagnosis are crucial for effective disease management and prevention. This project leverages deep learning techniques with artificial neural networks to build a robust predictive model that can classify patients as diabetic or non-diabetic based on their health metrics.

## Key Features

- **Artificial Neural Network**: Multi-layer perceptron (MLP) architecture for binary classification
- **Data Preprocessing**: Comprehensive data cleaning, normalization, and feature scaling
- **Model Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Interactive Notebooks**: Jupyter notebooks for exploration, training, and testing
- **Easy Integration**: Simple interface for making predictions on new data
- **Visualization**: Comprehensive plots and visualizations for model analysis

## Dataset Information

The dataset contains patient health records with the following features:

- **Age**: Patient's age in years
- **Gender**: Male/Female
- **Blood Glucose Level**: Fasting glucose levels
- **Blood Pressure**: Systolic/Diastolic pressure
- **BMI (Body Mass Index)**: Weight relative to height
- **Insulin Level**: Serum insulin levels
- **Diabetes Pedigree Function**: Family history indicator
- **Pregnancies**: Number of pregnancies (for females)
- **Outcome**: Target variable (0: Non-Diabetic, 1: Diabetic)

## Technologies & Libraries

### Core Libraries
- **Python 3.x** - Programming language
- **TensorFlow / Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities and preprocessing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## Project Structure

```
Diabetes-Prediction-using-ANN-Classification/
├── README.md                          # Project documentation
├── notebooks/                         # Jupyter notebooks
│   ├── exploratory_analysis.ipynb    # Data exploration and analysis
│   ├── data_preprocessing.ipynb      # Data cleaning and preprocessing
│   └── model_training.ipynb          # Model training and evaluation
├── data/                              # Dataset files
│   ├── diabetes.csv                  # Raw dataset
│   └── processed_data.csv            # Preprocessed dataset
├── models/                            # Trained model files
│   └── diabetes_ann_model.h5         # Trained ANN model
├── src/                               # Python source code
│   ├── preprocessing.py              # Data preprocessing functions
│   ├── model.py                      # ANN model architecture
│   └── utils.py                      # Utility functions
└── requirements.txt                   # Project dependencies
```

## Model Architecture

The ANN model follows this structure:

```
Input Layer (8 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU activation)
    ↓
Dropout Layer (0.2)
    ↓
Hidden Layer 2 (32 neurons, ReLU activation)
    ↓
Dropout Layer (0.2)
    ↓
Hidden Layer 3 (16 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
    ↓
Binary Classification (0 or 1)
```

### Model Specifications

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Regularization**: Dropout layers to prevent overfitting
- **Metrics**: Accuracy, Precision, Recall, AUC

## Installation & Setup

### Prerequisites

Ensure you have Python 3.6+ installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/adithyap15122005/Diabetes-Prediction-using-ANN-Classification.git
cd Diabetes-Prediction-using-ANN-Classification
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn jupyter
```

## Usage

### 1. Running Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the notebooks folder and open the desired notebook.

### 2. Training the Model

```python
from src.model import build_ann_model
from src.preprocessing import preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data('data/diabetes.csv')

# Build and train model
model = build_ann_model()
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### 3. Making Predictions

```python
# Predict on new data
new_patient = [[25, 1, 120, 70, 25, 100, 0.5, 2]]  # [age, gender, glucose, pressure, bmi, insulin, pedigree, pregnancies]
prediction = model.predict(new_patient)
print(f"Diabetes Risk: {prediction[0][0]:.2%}")
```

## Model Performance

The trained ANN model achieves the following performance metrics:

| Metric | Value |
|--------|-------|
| Accuracy | ~96% |
| Precision | ~95% |
| Recall | ~94% |
| F1-Score | ~94% |
| ROC-AUC | ~0.98 |

*(Metrics may vary based on data split and hyperparameters)*

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of predicted positive cases, how many are actually positive
- **Recall**: Of actual positive cases, how many are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Measure of model's ability to distinguish between classes

## Data Preprocessing Steps

1. **Handling Missing Values**: Remove or impute missing data
2. **Feature Scaling**: Normalize features using StandardScaler
3. **Train-Test Split**: 80-20 split for training and testing
4. **Feature Engineering**: Create new features if necessary
5. **Class Balancing**: Handle imbalanced datasets

## Hyperparameters

Key hyperparameters used in the model:

- **Batch Size**: 32
- **Epochs**: 100
- **Learning Rate**: 0.001 (default for Adam)
- **Dropout Rate**: 0.2
- **Hidden Layer Neurons**: [64, 32, 16]
- **Validation Split**: 20%

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Add cross-validation for robust evaluation
- [ ] Implement hyperparameter tuning (GridSearchCV, RandomSearchCV)
- [ ] Compare with other ML algorithms (Random Forest, SVM, XGBoost)
- [ ] Deploy as a Flask/Django web application
- [ ] Create a REST API for model predictions
- [ ] Build a mobile application
- [ ] Add real-time prediction dashboard
- [ ] Implement ensemble methods
- [ ] Add SHAP values for model interpretability

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Jupyter Notebook Not Starting
**Solution**: Install and start Jupyter
```bash
pip install jupyter
jupyter notebook
```

### Issue: Model Not Converging
**Solution**: Adjust hyperparameters or data preprocessing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

⚠️ **Important**: This project is for educational and research purposes only. The predictions made by this model should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## References & Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [UCI Machine Learning Repository - Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes)
- [Diabetes Prediction with ML - Research Papers](https://scholar.google.com/)

## Contact & Support

For questions, suggestions, or issues:

- **GitHub**: [@adithyap15122005](https://github.com/adithyap15122005)
- **Email**: adithyaparigi.15122005@gmail.com

---

**Last Updated**: June 2026

If this project helped you, please consider giving it a ⭐ star on GitHub!
