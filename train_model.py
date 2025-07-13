import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("AmesHousing.csv")
df = df.rename(columns={
    "Lot Area": "LotArea",
    "Year Built": "YearBuilt",
    "Total Bsmt SF": "TotalBsmtSF",
    "Gr Liv Area": "GrLivArea",
    "Garage Cars": "GarageCars",
    "Overall Qual": "OverallQual",
    "Sale Price": "SalePrice"
})

df = df[["LotArea", "YearBuilt", "TotalBsmtSF", "GrLivArea", "GarageCars", "OverallQual", "SalePrice"]]
df.dropna(inplace=True)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("house_price_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved!")
