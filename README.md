# 1st Month Tasks

## Task 1 – SMS Spam Detection using NLP and Machine Learning
- Preprocessed and cleaned the **Spam dataset** (removing nulls, renaming columns, text cleaning).
- Applied **TF-IDF vectorization** and handled class imbalance with **SMOTE**.
- Trained a **Naive Bayes classifier** and evaluated with **accuracy**, **confusion matrix**, and **classification report**.
- Built a complete **Scikit-learn Pipeline** including **preprocessing + classification**, with a real-time **spam_check()** function to classify custom inputs as spam or not.
- Achieved **~97% accuracy**

## Task 2 – MNIST Handwritten Digit Recognition using CNN
- Used the builtin **MNIST** dataset for handwritten digit classification.
- Built a deep **CNN model** with Conv2D, MaxPooling, Dense, and Dropout layers.
- Implemented **Early Stopping** to prevent overfitting.
- Evaluated the model on test data, achieving **strong accuracy**.
- Visualized training and **validation curves** (accuracy & loss) alongside test performance with custom plots.
- Achieved **99% accuracy** in all train,test,validation datsets.

# 2nd Month Tasks

## Task 3 – Housing Price Prediction
- Loaded and explored the **California Housing dataset** to predict median house values.
- Performed data cleaning, feature inspection, and **correlation** analysis using heatmaps.
- Selected **key features** (MedInc, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
- Standardized the features with **StandardScaler** to improve model performance.
- Trained a **Random Forest Regressor** and evaluated performance using **Mean Squared Error** and **R² score**.
- Visualized **actual vs predicted prices** through scatter plots for model validation.
- Achieved strong model performance with high **R²** and **low MSE** on test data.

## Task 4 – Iris Flower Classification
- Loaded and explored the **Iris dataset** for **classifying flower species** based on petal and sepal dimensions.
- Conducted data **analysis** and **visualization** with **pairplots** to observe class separations.
- Applied **StandardScaler** for **feature normalization** to enhance model performance.
- Trained a **Support Vector Machine** (SVM) classifier with RBF kernel for **multi-class classification**.
- Evaluated the model using **accuracy score**, **classification report**, and **confusion matrix visualization**.
- Achieved **~96–98% accuracy** across test data with excellent precision and recall for all species.
