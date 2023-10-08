import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

new = pd.read_csv('featured_data.csv')

X = new[['Processor', 'Operating System', 'SSD', 'Display', 'Warranty', 'Generation', 'HDD', 'RAM In GB', 'RAM TYPE', 'Storage', 'Touch Display']]
y = new['MRP']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )



# Encode categorical variables using LabelEncoder
le_processor = LabelEncoder()
le_os = LabelEncoder()
le_gen = LabelEncoder()
le_ram_type = LabelEncoder()

le_processor.fit(new['Processor'])
le_os.fit(new['Operating System'])
le_gen.fit(new['Generation'])
le_ram_type.fit(new['RAM TYPE'])


X_train['Processor'] = le_processor.transform(X_train['Processor'])
X_train['Operating System'] = le_os.transform(X_train['Operating System'])
X_train['Generation'] = le_gen.transform(X_train['Generation'])
X_train['RAM TYPE'] = le_ram_type.transform(X_train['RAM TYPE'])


X_test['Processor'] = le_processor.transform(X_test['Processor'])
X_test['Operating System'] = le_os.transform(X_test['Operating System'])
X_test['Generation'] = le_gen.transform(X_test['Generation'])
X_test['RAM TYPE'] = le_ram_type.transform(X_test['RAM TYPE'])

# Create a column transformer for scaling numerical variables
ct = make_column_transformer((StandardScaler(), ['SSD', 'Display', 'Warranty', 'HDD', 'RAM In GB', 'Storage']), remainder='passthrough')

# Create a pipeline for pre-processing and modeling
rf = RandomForestRegressor(n_estimators=100)
model = make_pipeline(ct, rf)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict laptop prices on the testing data
y_pred = model.predict(X_test)

# Calculate mean absolute error on the testing data
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


joblib.dump(le_processor, 'le_processor.pkl')
joblib.dump(le_os, 'le_os.pkl')
joblib.dump(le_gen, 'le_gen.pkl')
joblib.dump(le_ram_type, 'le_ram_type.pkl')

joblib.dump(model, 'model.pkl')