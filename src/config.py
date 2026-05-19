"""
Central configuration — all constants, paths, and schema definitions live here.
Import this module anywhere in the project instead of hard-coding values.
"""

from pathlib import Path

# Paths
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT_DIR / "data" / "Flight_Price_Dataset_of_Bangladesh.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR  = OUTPUTS_DIR / "models"   
PLOTS_DIR   = OUTPUTS_DIR / "plots"     
REPORTS_DIR = OUTPUTS_DIR / "reports"  

# Reproducibility 
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Cross-validation
CV_FOLDS      = 5    
GRID_CV_FOLDS = 5    
N_ITER        = 50   

#  Column rename map: raw CSV names → clean internal names 
COLUMN_RENAME_MAP: dict[str, str] = {
    "Base Fare (BDT)":        "Base_Fare",
    "Tax & Surcharge (BDT)":  "Tax_Surcharge",
    "Total Fare (BDT)":       "Total_Fare",
    "Departure Date & Time":  "Date",
    "Arrival Date & Time":    "Arrival_Date",
    "Duration (hrs)":         "Duration_hrs",
    "Stopovers":              "Stop_Raw",
    "Seasonality":            "Season",
    "Days Before Departure":  "Days_Before_Departure",
    "Source Name":            "Source_Name",
    "Destination Name":       "Destination_Name",
    "Aircraft Type":          "Aircraft_Type",
    "Booking Source":         "Booking_Source",
}

# ── Feature lists (after renaming) 
CATEGORICAL_FEATURES = [
    "Airline", "Source", "Destination",
    "Season", "Stop_Type",
    "Aircraft_Type", "Class", "Booking_Source",
]
NUMERICAL_FEATURES = [
    "Duration_hrs",
    "Days_Before_Departure", "Month", "Weekday",
]
TARGET = "Total_Fare"

# ── Expected column schema (after renaming) 
# Format: column_name → (expected_dtype_string, human_readable_description)
EXPECTED_DTYPES: dict[str, tuple[str, str]] = {
    "Airline":               ("object",   "Categorical — airline name"),
    "Source":                ("object",   "Categorical — origin IATA code"),
    "Destination":           ("object",   "Categorical — destination IATA code"),
    "Date":                  ("datetime", "Departure datetime"),
    "Base_Fare":             ("float64",  "Base ticket price (BDT)"),
    "Tax_Surcharge":         ("float64",  "Tax and surcharges (BDT)"),
    "Total_Fare":            ("float64",  "Total fare = Base + Tax (BDT)"),
    "Duration_hrs":          ("float64",  "Flight duration in hours"),
    "Stop_Raw":              ("object",   "Raw stopover label (Direct/1 Stop/2 Stops)"),
    "Season":                ("object",   "Travel season from dataset"),
    "Days_Before_Departure": ("int64",    "Days between booking and departure"),
    "Aircraft_Type":         ("object",   "Aircraft model"),
    "Class":                 ("object",   "Cabin class (Economy/Business/First)"),
    "Booking_Source":        ("object",   "Channel (Online/Travel Agency/Direct)"),
}

# ── Numeric value-range bounds: column → (min_valid, max_valid)
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Base_Fare":             (500,   500_000),   
    "Tax_Surcharge":         (0,     100_000),   
    "Total_Fare":            (500,   600_000),   
    "Duration_hrs":          (0.25,  20.0),      
    "Days_Before_Departure": (0,     365),       
}

#  Expected categorical values (None = don't enforce) 
# Used by validate_categorical_values() to detect data-entry anomalies.
EXPECTED_CATEGORIES: dict[str, list[str] | None] = {
    "Stop_Raw": ["Direct", "1 Stop", "2 Stops"],
    "Class":    ["Economy", "Business", "First"],
    "Season":   None,   # flexible — dataset may use different season labels
    "Airline":  None,
}

# ── Minimum valid base fare  
MIN_VALID_FARE = 500   

# ── Missing-value imputation strategies per column 
IMPUTATION_STRATEGY: dict[str, str] = {
    "Tax_Surcharge":         "median",
    "Base_Fare":             "median",
    "Duration_hrs":          "median",
    "Days_Before_Departure": "median",
    "Airline":               "mode",
    "Source":                "mode",
    "Destination":           "mode",
    "Season":                "mode",
    "Stop_Raw":              "mode",
    "Aircraft_Type":         "Unknown",   
    "Class":                 "mode",
    "Booking_Source":        "mode",
}

# ── Target transformation 

LOG_TRANSFORM_TARGET = True

# ── Plot style 
PLOT_STYLE   = "whitegrid"
PLOT_PALETTE = "muted"
PLOT_DPI     = 150
