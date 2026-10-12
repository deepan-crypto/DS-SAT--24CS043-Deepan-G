import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load datasets
orders = pd.read_csv("orders.csv", parse_dates=["order_time"])
riders = pd.read_csv("riders.csv").drop_duplicates(subset="rider_id", keep="first")
routes = pd.read_csv("routes.csv")
weather = pd.read_csv("weather.csv")
delivery = pd.read_csv("delivery.csv", parse_dates=["pickup_time", "delivered_time"])

# 2. Clean orders
orders = orders.drop_duplicates(subset="order_id", keep="first")
orders["customer_zone"] = orders["customer_zone"].astype("string").str.strip().str.title()
orders = orders.dropna(subset=["order_time"])

# 3. Clean delivery timestamps and calculate delivery duration
delivery = delivery.drop_duplicates(subset="order_id", keep="first")
delivery = delivery.dropna(subset=["pickup_time", "delivered_time"])
delivery["delivery_time"] = (
    delivery["delivered_time"] - delivery["pickup_time"]
).dt.total_seconds() / 60
delivery = delivery[delivery["delivery_time"] > 0]

# 4. Combine datasets
data = orders.merge(routes, on="order_id", how="inner", suffixes=("", "_route"))
data = data.merge(weather, on="order_id", how="left")
data = data.merge(delivery[["order_id", "rider_id", "pickup_time", "delivered_time", "status", "delivery_time"]], on="order_id", how="inner")
data = data.merge(riders, on="rider_id", how="left")
data["weather"] = data["weather"].fillna("Unknown")
data["traffic"] = data["traffic"].fillna("Unknown")
data["rider_experience"] = data["rider_experience"].fillna(data["rider_experience"].median())
data["hour"] = data["order_time"].dt.hour
data["is_peak"] = data["hour"].between(12, 14) | data["hour"].between(18, 21)

# Keep sensible rows for analysis
data = data[data["delivery_time"].between(1, 240)].copy()

print("Combined cleaned dataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nSummary:\n", data[["distance", "delivery_time", "rider_experience"]].describe())

# 5. EDA outputs
print("\nAverage delivery time by zone:\n", data.groupby("customer_zone")["delivery_time"].mean().sort_values(ascending=False))
print("\nAverage delivery time by traffic:\n", data.groupby("traffic")["delivery_time"].mean().sort_values(ascending=False))
print("\nAverage delivery time by weather:\n", data.groupby("weather")["delivery_time"].mean().sort_values(ascending=False))
print("\nAverage delivery time by hour:\n", data.groupby("hour")["delivery_time"].mean().sort_values(ascending=False).head(5))

# 6. Charts
sns.barplot(data=data, x="customer_zone", y="delivery_time", errorbar=None, color="steelblue")
plt.title("Average Delivery Time by Zone")
plt.ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig("delivery_by_zone.png")
plt.close()

sns.barplot(data=data, x="traffic", y="delivery_time", order=["Low", "Medium", "High", "Unknown"], errorbar=None, color="steelblue")
plt.title("Average Delivery Time by Traffic Level")
plt.ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig("delivery_by_traffic.png")
plt.close()

sns.scatterplot(data=data, x="distance", y="delivery_time", hue="traffic", alpha=0.65)
plt.title("Distance vs Delivery Time")
plt.tight_layout()
plt.savefig("distance_vs_delivery.png")
plt.close()

sns.barplot(data=data, x="weather", y="delivery_time", errorbar=None, color="steelblue")
plt.title("Average Delivery Time by Weather")
plt.ylabel("Delivery time (minutes)")
plt.tight_layout()
plt.savefig("delivery_by_weather.png")
plt.close()

# 7. Machine learning: predict delivery time
features = ["distance", "traffic", "weather", "temperature", "visibility",
            "rider_experience", "shift", "customer_zone", "hour", "is_peak"]
X = pd.get_dummies(data[features], drop_first=True)
y = data["delivery_time"]
X = X.fillna(X.median(numeric_only=True)).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("MAE:", round(mean_absolute_error(y_test, pred), 2), "minutes")
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2), "minutes")
print("R2:", round(r2_score(y_test, pred), 3))

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features:\n", importance.head(10))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 8. Export deliverables
data.to_csv("cleaned_delivery.csv", index=False)
data.to_excel("delivery_analysis.xlsx", index=False)
print("\nSaved cleaned_delivery.csv and delivery_analysis.xlsx plus chart PNG files.")
