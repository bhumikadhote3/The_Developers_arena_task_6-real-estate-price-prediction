# The_Developers_arena_task_6-real-estate-price-prediction
Industry-ready real estate price prediction system using machine learning and FastAPI
🏠 Real Estate Price Prediction System
📌 Project Overview

This project is an industry-ready capstone project that predicts real estate property prices using machine learning. It demonstrates a complete end-to-end data science workflow, including data preprocessing, model training, experiment tracking, and deployment as a REST API using FastAPI.

The system is designed to simulate a production-ready real estate price prediction service that can be consumed by web or mobile applications.

🎯 Objectives

Build a machine learning model to predict real estate prices

Implement a complete ML pipeline from data loading to prediction

Deploy the trained model using a FastAPI backend

Track experiments and models using MLflow

Expose prediction functionality through a REST API

🛠️ Tech Stack

Programming Language: Python

Backend Framework: FastAPI

Machine Learning: Scikit-learn

Experiment Tracking: MLflow

Data Handling: Pandas, NumPy

API Server: Uvicorn

📂 Project Structure
real-estate-capstone/
│
├── app.py                  # Main FastAPI application
├── house_prices.csv        # Dataset used for training
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── data/                   # Data-related files
├── notebooks/              # EDA and experimentation notebooks
├── ml_pipeline/            # ML pipeline scripts
├── docs/                   # Screenshots and documentation
├── frontend/               # Frontend (optional / future scope)
├── backend/                # Backend modules
├── tests/                  # Test cases
└── .gitignore              # Ignored files and folders

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone [https://github.com/yourusername/real-estate-capstone.git](https://github.com/bhumikadhote3/The_Developers_arena_task_6-real-estate-price-prediction.git
)
cd The_Developers_arena_task_6-real-estate-price-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
python app.py


The API will start at:

http://localhost:8000

🚀 API Usage
🔹 Swagger UI

Access interactive API documentation:

http://localhost:8000/docs

🔹 Prediction Endpoint

POST /predict

Sample Request Body:

{
  "Area": 1200,
  "Bedrooms": 2,
  "Bathrooms": 2,
  "Age": 5,
  "Location": "Hyderabad",
  "Property_Type": "Apartment"
}


Sample Response:

{
  "predicted_price": 21919650.47
}

🧠 Machine Learning Pipeline

Data loading and validation

Feature preprocessing

Model training using regression techniques

Model evaluation (MAE, R² score)

Model registration and tracking using MLflow

Serving predictions through FastAPI

📊 Model Tracking

MLflow is used to:

Track experiments

Log metrics and parameters

Store trained models

(MLflow artifacts are excluded from GitHub using .gitignore)

📸 Screenshots

Screenshots of:

Running FastAPI server

Swagger API documentation

Successful prediction response

are available in the docs/ folder.

📈 Business Impact

This system helps:

Real estate companies estimate property prices

Buyers understand market value

Data-driven decision-making for pricing strategies

🔮 Future Enhancements

Add advanced models (XGBoost, Neural Networks)

Integrate Streamlit or React frontend

Dockerize the application

Add CI/CD pipeline

Deploy to cloud (AWS / Azure)

👤 Author
Bhumika Dhote
dhotebhumika3@gmail.com
