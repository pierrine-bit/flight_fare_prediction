
# Flight Fare Prediction — Problem Definition

## 1. Business Context

Airline ticket pricing in Bangladesh is highly dynamic, fares vary based on airline, route, cabin class, travel season, booking lead time, and stopovers.  
Passengers and travel agencies currently have no reliable way to estimate total fare at booking.

## 2. Problem Statement

> Predict the total ticket fare (BDT) given flight attributes at booking time (Base Fare + Tax & Surcharge).

**Type:** Supervised Regression  
**Target:** `Total_Fare` (BDT)

## 3. Objectives

| # | Objective |
|---|-----------|
| 1 | Build a regression model to predict total fare |
| 2 | Identify key features driving fare variation |
| 3 | Compare 7 ML algorithms and select the best using CV R² |
| 4 | Deploy interactive Streamlit app with 95% prediction intervals |
| 5 | Automate daily retraining via Airflow DAG |

## 4. Success Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| CV R²  | ≥ 0.85 | 0.8934  |
| MAE    | ≤ 35,000 BDT | 28,278  |
| RMSE   | ≤ 55,000 BDT | 47,624 |

## 5. Input Features

| Feature | Description | Unique / Range |
|---------|-------------|----------------|
| Airline | Carrier name | 24 |
| Source | Origin airport | 8 |
| Destination | Destination airport | 20 |
| Class | Cabin class | Economy, Business, First |
| Stop_Type | Non-stop / With stopovers | Non-Stop: 36,642; With Stop: 20,358 |
| Season | Travel season | Regular: 44,525; Winter: 10,930; Hajj: 942; Eid: 603 |
| Duration_hrs | Flight duration in hours | 0.5 – 15.83 |
| Days_Before_Departure | Lead time from booking | 1 – 90 |
| Aircraft_Type | Aircraft model | 5 |
| Booking_Source | Booking channel | 3 |
| Month | Derived from departure date | 1 – 12 |
| Weekday | Derived from departure date | 0 – 6 |

> Base_Fare and Tax_Surcharge are excluded to prevent target leakage.

## 6. Assumptions

- Historical pricing patterns generalize to near-future bookings  
- Feature relationships remain consistent  
- Unseen airlines or routes handled via OHE (`handle_unknown='ignore'`)  
- Minimum fare in dataset: Base_Fare ≥ 500 BDT

## 7. Constraints

- No real-time pricing API used  
- Bangladesh-origin routes only  
- Static dataset; DAG retrains on same CSV unless a new CSV is provided

## 8. Scope

| Scope | Items |
|-------|-------|
| In Scope | Validation, cleaning, feature engineering, EDA, 7 ML models, Streamlit UI, Airflow DAG |
| Out of Scope | Seat availability prediction, multi-currency support |

## 9. Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Flight Price Dataset Bangladesh |
| Records | 57,000 |
| Columns | 17 |
| Target | Total_Fare (BDT) |
| Range after cleaning | 1,801 – 522,606 BDT · mean 70,348 |
