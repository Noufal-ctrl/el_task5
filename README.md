**Task 5: Decision Trees and Random Forests**

**Goal:**
Learn how to use Decision Trees and Random Forests for classification and regression.

**Tools Used:**

* Python
* Scikit-learn
* Matplotlib
* Seaborn

**Dataset:**

* Heart Disease UCI Dataset
* Target column: `target` (1 = has disease, 0 = no disease)

**Steps:**

1. Load the dataset using pandas.
2. Split the data into features (`X`) and target (`y`), then into training and testing sets.
3. Train a Decision Tree Classifier and show the tree diagram.
4. Control overfitting by setting a maximum depth (like `max_depth=3`).
5. Train a Random Forest Classifier and check if it's more accurate.
6. Show which features are most important using a bar chart.
7. Use 5-fold cross-validation to test model performance.
