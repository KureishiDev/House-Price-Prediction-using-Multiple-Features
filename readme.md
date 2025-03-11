# House Price Prediction

This project aims to predict house prices using various features like square footage, number of bedrooms, and the age of the house. We build a linear regression model and evaluate its performance.

## Project Structure

- **data/**: Contains the dataset (`imoveis.csv`).
- **src/**: Python source files for data preprocessing, model training, and visualization.
- **notebooks/**: Jupyter notebook for exploratory data analysis (EDA).
- **model.pkl**: The trained machine learning model.
- **requirements.txt**: Python dependencies for the project.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt

Usage
Load and preprocess the data: The data can be loaded and cleaned using the data_preprocessing.py module.
Train the model: Run model_training.py to train a linear regression model.
Evaluate the model: The model performance can be evaluated using mean squared error (MSE).
Visualize the results: Use visualization.py to visualize the relationship between features and price, as well as the comparison between predicted and actual prices.
License
MIT License - see the LICENSE file for details.

