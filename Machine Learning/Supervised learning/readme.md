This repo contains the implementation of a supervised learning task using Python.
The main objective is to import a tabular dataset, examine its structure, prepare it for modelling, train regression algorithms, and evaluate their performance on unseen data.

## Project Description

The dataset used in this project is stored in a CSV file and includes multiple predictor variables along with a target column named "y". The task involves predicting the target variable using regression models.The project follows a standard supervised learning workflow which includes:

1. Loading and inspecting the dataset
2. Visualising the data to understand feature relationships
3. Splitting the dataset into training and testing sets
4. Training at least two different regression models
5. Generating predictions for the test dataset
6. Evaluating model performance using appropriate metrics
7. Implementing one optional enhancement or additional feature

This work was carried out in Google Colab using Python.

## Tools and Libraries

The following libraries were used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data Preparation

The dataset was imported from a CSV file using pandas. The structure of the data was examined to identify numerical features, missing values, and the distribution of variables.
At least one figure was generated to visualise important relationships or patterns within the data.

The dataset was then divided into training and testing subsets using scikit-learn's `train_test_split` function. The target column "y" was separated from the input variables.

## Regression Models

Two regression models were trained and compared. Example models include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
- Support Vector Regression

Each model was fitted on the training data and used to generate predictions for the test set.

## Evaluation

The performance of each model was evaluated using metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared score

## Additional Feature

An optional enhancement was implemented. Examples of such features include:

- A feature importance visualisation
- Cross-validation
- Hyperparameter tuning
- A residual analysis graph

## How to Run the Notebook

1. Open the `.ipynb` file in Google Colab or Jupyter Notebook.
2. Upload the dataset file when prompted.
3. Run all cells in sequence.
4. Review outputs, plots, and evaluation metrics.
