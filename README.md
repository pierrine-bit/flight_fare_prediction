
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

| Model             | CV R² Mean | CV R² Std | Test R² | MAE (BDT) | RMSE (BDT) |
| ----------------- | ---------- | --------- | ------- | --------- | ---------- |
| Random Forest     | 0.8932     | 0.0012    | 0.6561  | 28,164    | 47,307     |
| XGBoost           | 0.8932     | 0.0012    | 0.6518  | 28,272    | 47,597     |
| Lasso             | 0.8931     | 0.0013    | 0.6496  | 28,271    | 47,748     |
| LightGBM          | 0.8931     | 0.0012    | 0.6544  | 28,216    | 47,418     |
| Linear Regression | 0.8930     | 0.0013    | 0.6511  | 28,244    | 47,650     |
| Ridge             | 0.8930     | 0.0013    | 0.6510  | 28,245    | 47,652     |
| Decision Tree     | 0.8923     | 0.0010    | 0.6564  | 28,171    | 47,286     |

> **Best model is selected dynamically** at runtime by CV R² Mean → CV R² Std → RMSE (BDT).  
> Random Forest and XGBoost are **statistically equivalent** (within 0.0002 CV R²) — the winner varies by run depending on which hyperparameters RandomizedSearchCV finds. Both achieve CV R² ≈ 0.8932, MAE ≈ BDT 28,200, and 95% CI ±91,600 BDT.

Residual std ≈ 46,700 BDT → 95% CI ≈ ±91,600 BDT.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| log1p target transform | Reduces right skew (1.554 → −0.172); improves fit for all models |
| Preprocessor fitted on train only | Prevents leakage — test statistics never influence the pipeline |
| `handle_unknown='ignore'` in OHE | Unseen airlines/routes at inference silently map to zero vectors |
| Shared KFold across all models | Identical splits eliminate split-randomness as a confound |
| GridSearchCV for Ridge, Lasso, Decision Tree | Exhaustive search feasible at these grid sizes (15, 9, 60 combos) |
| RandomizedSearchCV n_iter=75 for ensembles | RF/XGBoost/LightGBM have 800k+ combinations — random sampling is the only practical approach |
| 4-level tiebreaker: CV R² → Std → RMSE → Name | RF and XGBoost are statistically tied; deterministic ordering prevents run-dependent results |
| Config-driven imputation with catch-all fallback | New dataset columns are handled automatically; only config needs updating |
| Versioned artefact saves | Every retrain preserves a timestamped rollback copy |

## Conclusion

The Flight Fare Prediction project delivers a **robust, production-ready ML pipeline**.
It provides accurate fare prediction with cross-validated models, a retrainable workflow via Airflow, and an interactive Streamlit UI.
All pipeline steps, metrics, and artefacts are **validated and versioned**, making this solution reliable, reproducible, and ready for operational deployment.

