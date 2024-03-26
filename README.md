README
Breast Cancer Analysis with PCA and Logistic Regression
This Python code performs exploratory data analysis (EDA) and classification on the Wisconsin Breast Cancer dataset using Principal Component Analysis (PCA) and Logistic Regression.
Requirements:
•	Python 3.x
•	NumPy (pip install numpy)
•	Pandas (pip install pandas)
•	Matplotlib (pip install matplotlib)
•	Scikit-learn (pip install scikt-learn)
Instructions:
1.	Save the code as a Python script (e.g., breast_cancer_analysis.py).
2.	Ensure you have the required libraries installed.
3.	Run the script from your terminal: python breast_cancer_analysis.py
Code Overview:
1.	Imports: Necessary libraries are imported for data manipulation, visualization, machine learning algorithms, and model evaluation.
2.	Load Data: The Wisconsin Breast Cancer dataset is loaded using sklearn.datasets. The features are stored in a pandas DataFrame (X), and the target variable (cancer diagnosis) is stored in a pandas Series (y).
3.	Data Standardization: The data is standardized using StandardScaler to ensure features are on a similar scale before PCA.
4.	Dimensionality Reduction: PCA is applied to reduce the number of features to two principal components (X_pca) that capture the most variance in the data.
5.	Visualization: The reduced dataset is visualized using a scatter plot, where points are colored based on the target variable (benign or malignant).
6.	Train-Test Split: The data is split into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split for model evaluation.
7.	Logistic Regression (Bonus): A Logistic Regression model is trained on the training data (logreg.fit) to predict cancer diagnosis. Predictions are made on the testing data (y_pred), and the model's accuracy is evaluated using accuracy_score.
Interpretation:
•	The visualization (scatter plot) helps to identify potential patterns or clusters in the data, which can be indicative of separability between the benign and malignant classes.
•	The accuracy of the Logistic Regression model provides a measure of how well the model generalizes to unseen data.
