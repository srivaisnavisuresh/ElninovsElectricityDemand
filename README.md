# ElninovsElectricityDemand
My forecasting model for electricity demand based on machine learning that combines past electricity demand data with ENSO climate data to produce reliable 90-day forecasts. Includes a dynamic Flask dashboard along with visualization and Excel report download options.

REVIEW THE PROJECT.DOCX FOR BETTER UNDERSTANDING OF THE PROJECT

Installation & Setup
Follow the steps below to set up and run the project on your local machine.

##Step 1: Clone the Repository
bash
git clone https://github.com/ElNino-Electricity-Demand-Forecasting.git

Navigate to the project directory:
bash
cd ElNino-Electricity-Demand-Forecasting

## Step 2: Create a Virtual Environment (Recommended)
Windows
bash
python -m venv venv
venv\Scripts\activate

## Step 3: Install Required Libraries
Install all project dependencies using:
bash
pip install -r requirements.txt

## Step 4: Verify the Project Structure

Your project should have the following structure:

ElNino-Electricity-Demand-Forecasting/
│── app.py
│── train_model.py
│── requirements.txt
│── README.md
│── LICENSE
│── model.pkl
│
├── data/
│   └── elnino_electricity_1980_2025.xlsx
│
├── templates/
│   └── dashboard.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── report.html/

## Step 5: Train the Model 
Skip because electricity_demand_change_model.pkl is already included


## Step 6: Launch the Flask Application
Run the web application:
bash
python app.py

## Step 7: Open the Dashboard
Open your browser and visit:
http://127.0.0.1:5000

## Features Available
* View historical electricity demand trends
* Generate 90-day electricity demand forecasts
* Compare actual and predicted values
* Download forecast reports in Excel format
* Visualize forecasting results through interactive charts

## Troubleshooting
If you encounter missing package errors:
bash
pip install -r requirements.txt

If the application cannot locate the dataset or model, ensure that:

* electricity_demand_change_model.pkl is in the project root directory.
* The dataset is located inside the data/ folder.
* File paths in`app.py match your project structure.
