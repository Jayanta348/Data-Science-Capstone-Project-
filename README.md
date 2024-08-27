# Data Science Capstone Project

## Project Overview

This repository contains the code and data for the Data Science Capstone Project. The goal of this project is to apply various machine learning techniques to a car details dataset to predict the selling price of cars.

## Project Structure

- **data/**: Contains the dataset files.
- **notebooks/**: Jupyter notebooks for data exploration and analysis.
- **scripts/**: Python scripts for data preprocessing, modeling, and evaluation.
- **model/**: Saved machine learning models.
- **README.md**: This file.

## Dataset

The dataset used for this project includes the following columns:
- `name`: Car model name
- `year`: Year of manufacture
- `selling_price`: Selling price of the car
- `km_driven`: Kilometers driven
- `fuel`: Fuel type
- `seller_type`: Type of seller
- `transmission`: Transmission type
- `owner`: Number of owners

## Project Steps

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
2. **Exploratory Data Analysis (EDA)**: Analyzing the data using visualizations and statistical methods.
3. **Model Training**: Applying various machine learning models including Linear Regression, Decision Tree, Random Forest, Bagging, SVR, and KNN.
4. **Model Evaluation**: Evaluating models using metrics such as Mean Squared Error (MSE) and R2 Score.
5. **Model Saving and Loading**: Saving the best performing model and loading it for future use.
6. **Testing on Sample Data**: Testing the saved model on a randomly selected subset of the dataset.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Data-Science-Capstone-Project.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd Data-Science-Capstone-Project
    ```

3. **Install Dependencies**:
    Ensure you have the required Python packages installed. You can use `requirements.txt` or install manually:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

4. **Run the Code**:
    You can run the scripts or Jupyter notebooks provided in the `scripts/` and `notebooks/` directories.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to [your-email@example.com](mailto:your-email@example.com).
