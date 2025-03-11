House Price Prediction
This project aims to predict house prices using various features such as square footage, number of bedrooms, and the age of the house. A linear regression model is built to make predictions and evaluate the model's performance.

Project Structure
house-price-prediction/ ├── data/
│ └── imoveis.csv # Dataset
├── src/
│ └── data_preprocessing.py # Data loading and cleaning
│ └── model_training.py # Model creation, training, and evaluation
│ └── visualization.py # Visualization of results
├── notebooks/
│ └── exploratory_analysis.ipynb # Jupyter notebook for EDA
├── requirements.txt # Project dependencies
├── README.md # Project description and instructions
└── model.pkl # Saved model

Overview
The project uses a dataset (imoveis.csv) containing information about houses, such as their square footage, number of bedrooms, and age. The goal is to predict the house price using these features.

Key Features:
SquareFeet: The size of the house in square feet.
Bedrooms: The number of bedrooms in the house.
Age: The age of the house.
Price: The target variable that represents the price of the house.
What You Can Do:
Load and clean the dataset.
Perform Exploratory Data Analysis (EDA) to understand the data and its relationships.
Build a linear regression model to predict house prices.
Evaluate the model using metrics like Mean Squared Error (MSE).
Visualize the data and results using matplotlib and seaborn.
Installation
Clone the repository and install the dependencies by running:

git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt

Usage
1. Load and Preprocess the Data
The data_preprocessing.py script contains functions to load and clean the data. This step handles missing values and prepares the data for model training.

Example:
from src.data_preprocessing import load_data, preprocess_data
df = load_data('data/imoveis.csv')
df = preprocess_data(df)

2. Train the Model
The model_training.py script builds and trains a linear regression model. It evaluates the model using the Mean Squared Error (MSE) and saves the trained model to a file (model.pkl).

Example:
from src.model_training import train_model
model = train_model(df)

3. Evaluate the Model
You can evaluate the model performance by checking the MSE value printed during training, which gives an indication of how well the model is performing on the test data.

4. Visualize the Results
The visualization.py script allows you to generate plots for better understanding and presentation of your data and model results.

Example:
from src.visualization import plot_data, plot_model_results
plot_data(df) # Visualizes the relationship between features and price

5. Exploratory Data Analysis (EDA)
You can also perform further data exploration in the Jupyter notebook (notebooks/exploratory_analysis.ipynb), which allows you to interactively analyze the dataset.

To start the notebook, use:
jupyter notebook notebooks/exploratory_analysis.ipynb

Dependencies
This project requires the following Python packages:

pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
Install all dependencies at once using the following command:
pip install -r requirements.txt

Model Evaluation
After training the model, you can evaluate its performance using the Mean Squared Error (MSE), which is printed out during training. The model's predictions are compared to the actual house prices to assess accuracy.

Saving and Loading the Model
The trained model is saved to a .pkl file using joblib. You can load the saved model for later use like this:

import joblib
model = joblib.load('model.pkl')

License
This project is licensed under the MIT License - see the LICENSE file for details.
