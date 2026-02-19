import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel("Online Retail.xlsx")

# Data preprocessing
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df.head())
print(df.describe())

# -----------------------------
# Sales by Country
# -----------------------------
country_sales = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
country_sales.head(10).plot(kind="bar")
plt.title("Top Countries by Sales")
plt.ylabel("Sales")
plt.show()

# -----------------------------
# Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["TotalPrice"], bins=50, kde=True)
plt.title("Distribution of Total Sales")
plt.show()

# -----------------------------
# Correlation
# -----------------------------
plt.figure(figsize=(6,4))
sns.heatmap(df[["Quantity","UnitPrice","TotalPrice"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# Scatter
# -----------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x="Quantity", y="TotalPrice", data=df)
plt.title("Quantity vs Total Price")
plt.show()

# -----------------------------
# Sales Over Time
# -----------------------------
sales_time = df.set_index("InvoiceDate").resample("M")["TotalPrice"].sum()

plt.figure(figsize=(8,5))
plt.plot(sales_time)
plt.title("Monthly Sales Trend")
plt.show()

# -----------------------------
# Regression
# -----------------------------
X = df[["Quantity"]]
y = df["TotalPrice"]

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R²:", model.score(X, y))
