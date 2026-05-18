
# Flight Fare Prediction

An end-to-end ML pipeline for predicting flight fares (BDT) on Bangladesh routes.  
Includes automated retraining via Airflow and an interactive Streamlit app.

## Quick Start

1. **Download dataset** and place in `data/`:

```bash
mkdir data
mv Flight_Price_Dataset_of_Bangladesh.csv data/
````

2. **Setup environment:**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the pipeline:**

```bash
python main.py
python main.py --skip-eda
```

4. **Launch Streamlit UI:**

```bash
streamlit run app.py
```

5. **Retrain model on new CSV:**

```bash
python retrain.py --data path/to/new_data.csv
```

6. **Start Airflow (optional):**

```bash
cp .env.example .env
docker compose up --build -d
```

## Project Structure

```
flight_fare_prediction/
├── data/                          
├── src/                          
│   ├── config.py                  
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── models.py
│   ├── evaluation.py
│   └── logger.py
├── outputs/
│   ├── models/                    
│   ├── plots/                     
│   ├── reports/                   
│   └── pipeline.log
├── dags/
│   └── flight_fare_dag.py         
├── notebooks/
│   └── flight_fare_prediction.ipynb
├── docs/
│   ├── project_report.md
│   └── problem_definition.md
├── app.py                         
├── main.py                        
├── retrain.py                     
├── Dockerfile                     
├── docker-compose.yaml            
├── requirements.txt
```

## Pipeline Overview

| Stage | Description                                                         |
| ----- | ------------------------------------------------------------------- |
| 0     | Pre-validation: CSV schema & min fare ≥ 500 BDT                     |
| 1     | Load & inspect raw CSV                                              |
| 2     | Clean data: dtype corrections, range validation, deduplication      |
| 3     | Feature engineering: Month, Weekday, Stop_Type, Route → 73 features |
| 4     | EDA: optional, plots saved                                          |
| 5     | Train 7 regression models                                           |
| 6     | Evaluate: CV R², MAE, RMSE, residual std, 95% CI                    |
| 7     | Retrain best model on full dataset                                  |
| 8     | Save artefacts versioned                                            |
| 9     | Streamlit UI → interactive predictions                              |
| 10    | Airflow DAG → automated daily retraining                            |


## Model Results 

| Model             | CV R² Mean | Test R² | MAE (BDT) | RMSE (BDT) |
| ----------------- | ---------- | ------- | --------- | ---------- |
| XGBoost           | 0.8934     | 0.6514  | 28,278    | 47,624     |
| Random Forest     | 0.8932     | 0.6561  | 28,164    | 47,307     |
| Lasso             | 0.8931     | 0.6496  | 28,271    | 47,748     |
| LightGBM          | 0.8931     | 0.6544  | 28,216    | 47,418     |
| Linear Regression | 0.8930     | 0.6511  | 28,244    | 47,650     |
| Ridge             | 0.8930     | 0.6510  | 28,245    | 47,652     |
| Decision Tree     | 0.8923     | 0.6564  | 28,171    | 47,286     |

Residual std = 46,759 BDT → 95% CI ±91,647 BDT.

## Key Design Decisions

The pipeline applies a **log1p transformation** to the target (skew 1.554 → −0.172) to improve fit.
The **preprocessor is fitted only on training data** to prevent leakage, and categorical features use **`handle_unknown='ignore'`** for unseen values.
A **shared KFold** ensures consistent CV splits. The **best model is selected dynamically** by CV R² mean, and all artefacts are **versioned** for reproducibility.

## Conclusion

The Flight Fare Prediction project delivers a **robust, production-ready ML pipeline**.
It provides accurate fare prediction with cross-validated models, a retrainable workflow via Airflow, and an interactive Streamlit UI.
All pipeline steps, metrics, and artefacts are **validated and versioned**, making this solution reliable, reproducible, and ready for operational deployment.

