# Walmart Sales Forecasting

This project focuses on forecasting weekly sales for various Walmart stores, leveraging historical sales data and relevant external factors. The goal is to build a robust machine learning model that can accurately predict future sales, identifying key drivers of sales performance.

## Project Overview

The core objective of this project is to develop an effective sales forecasting model. The analysis and modeling cover the following aspects:
- **Exploratory Data Analysis (EDA)**: Understanding sales trends over time, distribution of sales, and performance across different stores.
- **Feature Engineering**: Creating insightful features from date information and historical sales data (lagged sales, rolling averages) to capture temporal patterns and dependencies. Crucially, **granular holiday features** are engineered to account for significant sales spikes around major holiday periods.
- **Model Training and Evaluation**: Building and comparing multiple machine learning regression models (Random Forest, Ridge Regression, Decision Tree, XGBoost, LightGBM) to identify the best performer.
- **Hyperparameter Tuning**: Optimizing the best-performing model (LightGBM) using Randomized Search to further enhance its predictive accuracy.
- **Feature Importance Analysis**: Identifying which factors are most influential in predicting weekly sales.

## Dataset

The project utilizes the Walmart Sales data, publicly available on Kaggle. This dataset contains historical sales data for 45 Walmart stores, spanning from February 2010 to November 2012. It includes key attributes that can influence sales:
- `Store`: Store ID.
- `Date`: The week of sales.
- `Weekly_Sales`: Weekly sales for the given store.
- `Holiday_Flag`: Binary indicator if the week is a holiday week.
- `Temperature`: Average temperature in the region.
- `Fuel_Price`: Cost of fuel in the region.
- `CPI`: Consumer Price Index.
- `Unemployment`: Unemployment rate in the region.

**Dataset Link:** [Walmart Sales Data on Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

## Technologies Used

- **Python**: The primary language for data processing, analysis, and model building.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: Used for data visualization and presenting insights in the form of charts and graphs.
- **Scikit-learn**: For machine learning model implementation, including preprocessing, model selection, and evaluation metrics.
- **XGBoost**: A powerful gradient boosting library for robust predictions.
- **LightGBM**: Another highly efficient gradient boosting framework, often outperforming other boosting algorithms.
- **Google Colab**: The project is developed and executed in Google Colab for easy sharing and execution in an online environment.

## Project Structure

The project's main components are:
- `Walmart.csv`: The primary dataset containing the sales and related information.
- `Sales_Forecasting_Project.ipynb`: The Jupyter notebook containing all the code for data loading, EDA, feature engineering, model training, evaluation, and visualization.
- `README.md`: This documentation file.

## How to Run

This project can be executed either in a local Python environment (using Jupyter Notebook) or directly in Google Colab.

### Local Environment Setup and Execution:

1.  **Clone the Repository:**
    Open your terminal or command prompt and run the following command to download the project files:
    ```bash
    git clone https://github.com/nivvi1106/Sales-Forecasting.git
    ```

2.  **Navigate to Project Directory:**
    Change your current directory to the cloned project folder:
    ```bash
    cd Sales-Forecasting
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install Required Libraries:**
    Install all necessary Python packages using pip.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
    ```
    Alternatively, you can create a `requirements.txt` file (if you haven't already, run `pip freeze > requirements.txt` after installing all libraries) and then simply run:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Jupyter Notebook:**
    Once all libraries are installed, launch Jupyter Notebook:
    ```bash
    jupyter notebook Sales_Forecasting_Project.ipynb
    ```
    This will open the notebook in your web browser.

7.  **Verify Data Path:**
    Ensure the `Walmart.csv` file is located in the same directory as the `Sales_Forecasting_Project.ipynb` notebook. The notebook will automatically find it if it's in the correct relative path.

8.  **Execute Cells:**
    Run all cells sequentially within the notebook to perform the analysis and generate the model predictions.

### Google Colab Execution:

1.  **Open the Notebook:**
    Go to [Google Colab](https://colab.research.google.com/) and upload `Sales_Forecasting_Project.ipynb` (File -> Upload notebook).

2.  **Install Missing Libraries (if any):**
    Google Colab comes with many libraries pre-installed. However, you might need to install `lightgbm` explicitly. Run the following cell at the beginning of your notebook if you encounter import errors:
    ```python
    !pip install lightgbm
    ```

3.  **Upload Dataset to Google Drive & Mount:**
    * Upload the `Walmart.csv` dataset to your Google Drive (e.g., into a folder like `My Drive/Colab Notebooks/Store_Sales/`).
    * Add the following lines at the beginning of your notebook to mount your Google Drive:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```

4.  **Update File Path:**
    Ensure the `file_path` variable in your data loading cell (`file_path = '/content/drive/MyDrive/Colab Notebooks/Store_Sales/Walmart.csv'`) correctly points to the location where you uploaded `Walmart.csv` in your Google Drive.

5.  **Execute Cells:**
    Run all cells sequentially within the notebook to perform the analysis and generate the model predictions.
     
## Key Steps Performed in the Project

1.  **Data Loading & Preprocessing**:
    * Loaded the `Walmart.csv` dataset into a Pandas DataFrame.
    * Converted the `Date` column to datetime objects and sorted the data chronologically.
    * Checked for missing values (none found).

2.  **Exploratory Data Analysis (EDA)**:
    * Visualized total weekly sales over time to identify trends and seasonality (e.g., holiday spikes).
    * Analyzed the distribution of weekly sales and total sales by store.

3.  **Feature Engineering**:
    * Extracted time-based features from the `Date` column: `Year`, `Month`, `WeekOfYear`, `DayOfWeek`.
    * **Created specific binary features for key holiday weeks:** `Is_Thanksgiving_Week`, `Is_Christmas_Week`, `Is_NewYear_Week`, `Is_SuperBowl_Week`, `Is_Easter_Week` to capture their distinct impact on sales.
    * Generated lag features (`Weekly_Sales_Lag1`) to capture previous week's sales dependency.
    * Calculated rolling mean features (`Weekly_Sales_Roll_Mean4`) to incorporate recent sales trends.
    * Handled NaN values introduced by lag/rolling features by dropping corresponding rows.

4.  **Train/Test Split**:
    * Performed a time-based split of the dataset into training and testing sets to ensure no data leakage from the future. Data before July 1, 2012, was used for training, and data from July 1, 2012, onwards for testing.

5.  **Model Training & Evaluation**:
    * Trained and evaluated several regression models, including a baseline Random Forest, Ridge Regression, Decision Tree, XGBoost, and LightGBM.
    * Used Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) as evaluation metrics.

6.  **Hyperparameter Tuning (LightGBM)**:
    * Performed `RandomizedSearchCV` to find the optimal hyperparameters for the best-performing LightGBM model, further enhancing its accuracy.

7.  **Feature Importance Analysis**:
    * Identified the most influential features for sales prediction, confirming the high impact of engineered lag and rolling mean features, as well as specific holiday indicators.

## Project Achievements & Results

The project successfully developed a highly accurate sales forecasting model. Key achievements include:

-   A significant reduction in prediction error compared to a baseline model, demonstrating the effectiveness of feature engineering and model selection.
-   The **Tuned LightGBM model** emerged as the best performer, achieving:
    -   **MAE: ~$40,580.37**
    -   **RMSE: ~$59,097.44**
-   This represents approximately a **52% reduction in Mean Absolute Error** compared to the baseline Random Forest model (MAE: ~$84,633).
-   Feature importance analysis revealed that `Weekly_Sales_Roll_Mean4`, `Weekly_Sales_Lag1`, and specific holiday weeks (`Is_Thanksgiving_Week`, `Is_NewYear_Week`, `Is_Christmas_Week`) are the most critical factors influencing weekly sales.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
