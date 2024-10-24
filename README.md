# Flood-Prediction-using-Random-Forest
This project focuses on predicting flood probabilities using a Random Forest Regressor. The model was trained on historical data to predict the likelihood of floods based on various environmental and geographical factors. The dataset includes variables such as Monsoon Intensity, River Management, Urbanization, and more.
![output](https://github.com/user-attachments/assets/85c9c56d-1e0f-4a2d-9e2b-203388b3dcc9)
Project Structure
The repository contains the following files:

flood_prediction.ipynb: The Jupyter Notebook containing the full code for data preprocessing, model training, validation, and test predictions.
train.csv: The dataset used for training the model.
test.csv: The dataset used for making predictions.
submission.csv: The final predictions for the test set, formatted for submission.
random_forest_model.pkl: The saved trained model using joblib, so the model doesn't need to be retrained each time.
flood_prediction.zip: A zipped version of the entire project for ease of sharing.
Dataset
The data used for this project includes various features that affect flood probability, such as:

Monsoon Intensity
Deforestation
Urbanization
Siltation
Population Score
Drainage Systems
Political Factors
And more.
The training dataset contains 1,117,957 entries and 22 columns, with FloodProbability as the target variable.

Model Used
Random Forest Regressor: A powerful ensemble learning method that builds multiple decision trees and averages their outputs for robust predictions. The Random Forest model was tuned to optimize performance by adjusting parameters such as:
n_estimators: The number of trees in the forest (set to 50).
max_depth: The maximum depth of the trees (set to 10 to prevent overfitting).
Key Steps:
Data Preprocessing:
Handled missing data and scaled features using StandardScaler.
Model Training:
The Random Forest model was trained using the RandomForestRegressor from sklearn.
Cross-Validation:
5-fold cross-validation was used to evaluate the stability and performance of the model.
Test Predictions:
The trained model was used to generate flood probability predictions for the test dataset.
Feature Importance:
The importance of each feature was visualized to understand which factors contributed the most to the predictions.
Evaluation Metrics
Mean Squared Error (MSE): The model was evaluated using MSE on both the validation set and during cross-validation. A low MSE of 0.0016 was achieved across multiple folds, indicating good performance.
How to Run the Project
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/flood-prediction.git
Install the required Python libraries:
Copy code
pip install -r requirements.txt
Open the Jupyter Notebook flood_prediction.ipynb and run the cells in order.
You can also load the trained model from random_forest_model.pkl to avoid re-training:
python
Copy code
import joblib
model = joblib.load('random_forest_model.pkl')
Results
The model successfully predicts flood probabilities based on various features. These predictions can help in disaster preparedness and planning, especially in flood-prone regions.
Contributing
If you'd like to contribute to this project, feel free to create a pull request or open an issue with suggestions.

License
This project is licensed under the MIT License - see the LICENSE file for details.
