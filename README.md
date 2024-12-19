# Demand Forecasting with Tabular, Textual, and Image Data

This project implements demand forecasting for items in a store using a combination of data modalities: tabular, textual, and image data. The project explores multiple machine learning and deep learning approaches, comparing their performance on a Kaggle dataset. The key goal is to predict demand with the lowest possible RMSE (Root Mean Square Error).

![image](https://github.com/user-attachments/assets/60db5da6-157c-450e-bda0-9d30f5ecb713)

## Objective
To predict item demand by leveraging:
- Historical sales data (tabular data)
- Textual information
- Visual data (images) for crowd counting using YOLO

## Models Implemented

### 1. **SARIMAX**
Seasonal AutoRegressive Integrated Moving Average with eXogenous variables, a classical time-series model.

### 2. **LSTM**
Long Short-Term Memory neural networks, a deep learning model for sequential data.

### 3. **Prophet**
A time-series forecasting tool by Facebook that handles seasonality.

### 4. **CNN**
Convolutional Neural Networks applied for:
   - **Time-only data**
   - **Time + YOLO-based crowd counting**

### 5. **CNN-LSTM**
A hybrid model combining CNNs and LSTMs for capturing both spatial and temporal features.

## Results
Validation RMSE (Root Mean Square Error) for each model:

| Model                | Validation RMSE |
|----------------------|-----------------|
| CNN (Time only)      | **8.64**        |
| CNN (Time + YOLO)    | **8.16**        |
| Fb Prophet           | 45.103         |
| SARIMAX              | 60.72          |
| LSTM                 | 98.87          |

## Dataset
The dataset includes:
- **Tabular data:** Historical sales data.
- **Images:** Used YOLO for crowd counting.
- **Textual data:** Features derived from textual information.

## Project Structure
```
TSA_CNN/
├── models/                  # Saved model files
├── result_plots/            # RMSE plots, other evaluation metrics
├── CNN-ts                   # CNN-based implementation for tabular data
├── RMSE.png                 # RMSE comparison plot
├── TimeSeries_CNN           # Implementation of CNN for time-series data
├── TimeSeries_CNN_LSTM_IMPROVED  # Improved CNN-LSTM implementation
├── TSA_prophet.ipynb        # Prophet model implementation
├── TSA_SARIMAX.ipynb        # SARIMAX implementation
├── TSA_LSTM.ipynb           # LSTM model implementation
├── TSA-prophet.ipynb        # Prophet model detailed analysis
```

## Highlights
- **YOLO Integration:** Used YOLO for image-based crowd counting, which significantly improved forecasting when combined with tabular data.
- **Best Performing Model:** CNN (Time + YOLO) achieved the lowest RMSE of 8.16.

## Requirements
To run the project, install the following dependencies:
```bash
pip install numpy pandas matplotlib tensorflow keras scikit-learn statsmodels yfinance fbprophet
```

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd TSA_CNN
   ```
3. Run any of the Jupyter Notebooks to reproduce the results.

## Future Work
- Explore Transformer models for time-series forecasting.
- Experiment with additional image processing techniques for feature extraction.
- Incorporate advanced text analysis methods for textual data.



## Results
The **CNN (Time + YOLO)** model achieved the lowest RMSE of 8.16.

![image](https://github.com/user-attachments/assets/c2f6380b-a9fa-4d23-b8e9-24478f9c8fbc)

![image](https://github.com/user-attachments/assets/6f5db772-b359-4823-bdcf-fe666ede625f)

![image](https://github.com/user-attachments/assets/d44e2c4a-7ec7-4eaf-90dd-055de585ea4d)


### RESULTS[RMSE] COMPARISON
![image](https://github.com/user-attachments/assets/8c907667-1d47-4632-819b-5305fd848b8a)

