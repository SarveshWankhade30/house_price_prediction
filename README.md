
# House Price Prediction Using Machine Learning

[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This project aims to predict house prices based on features such as the number of bedrooms, square footage, lot size, neighborhood quality, and other property attributes. The application combines machine learning and an interactive web interface to provide real-time price predictions for users. 

The web app is designed for ease of use, making it accessible to non-technical users and real estate professionals.

## Features
- Predict house prices based on user-provided property details.
- Interactive web UI developed using **Streamlit**.
- Trained on multiple machine learning algorithms with hyperparameter tuning for the best results.
- Real-time prediction using a pre-trained model.

## Technologies Used
- **Python**: Core programming language for development.
- **pandas**: Data manipulation and preprocessing.
- **numpy**: Numerical computations.
- **scikit-learn**: Building and optimizing machine learning models.
- **streamlit**: Creating the web-based user interface.
- **pickle**: Saving and loading the trained ML model.

## Algorithms Implemented
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting (e.g., XGBoost)**
4. **Support Vector Regression (SVR)**

## Installation and Usage
Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/sarveshwankhade30/house_price_prediction.git
cd house_price_prediction
```

### 2. Install Dependencies
Ensure you have Python installed. Install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to retrain the model, run:
```bash
python training.py
```
This will preprocess the data, train the model, and save the best model as `best_model.pkl`.

### 4. Run the Web App
Start the Streamlit app using the following command:
```bash
streamlit run app.py
```

## Project Structure
```
house_price_prediction/
├── app.py                # Streamlit web app script
├── training.py           # Script to train the ML model
├── data.csv              # Dataset for training the model
├── requirements.txt      # Dependencies for the project
├── best_model.pkl        # Pre-trained model (generated after training)
└── README.md             # Project documentation
```

## How It Works
1. **Input Property Details**: Enter features like the number of bedrooms, lot size, neighborhood quality, etc., into the web app.
2. **Model Prediction**: The app uses the trained model to process inputs and predict house prices.
3. **Output**: Display the estimated price in real-time on the app.

## Demo
You can access the live app [here](https://github.com/sarveshwankhade30/house_price_prediction).
https://housepricepredictionmodel.streamlit.app/

## Contributing
Contributions are welcome! Feel free to fork this repository, create a branch, and submit a pull request.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

**Created by Sarvesh Wankhade**  
Connect with me on [LinkedIn](https://www.linkedin.com/in/sarvesh-wankhade/)
