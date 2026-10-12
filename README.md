# CH07 Food Delivery Insight (Synthetic Practice Project)

This project uses generated practice data inspired by the CH07 challenge. The CSVs are synthetic, not real QuickBite business records.

## Files
- orders.csv
- riders.csv
- routes.csv
- weather.csv
- delivery.csv
- analysis.py

## Run
Install dependencies:
`pip install pandas numpy matplotlib seaborn scikit-learn openpyxl`

Run from this folder:
`python analysis.py`

The script cleans and merges the CSV files, prints EDA summaries, saves charts, trains a Random Forest Regressor, reports MAE/RMSE/R², and exports cleaned_delivery.csv and delivery_analysis.xlsx.
