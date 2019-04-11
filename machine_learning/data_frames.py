import pandas as pd
from sklearn.tree  import DecisionTreeRegressor

melbourne_data = pd.read_csv('data/melb_data.csv')

pruned_melbourne_data = melbourne_data.dropna(axis=0)

target = pruned_melbourne_data.Price
predictor_columns = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'YearBuilt']
features = pruned_melbourne_data[predictor_columns]


tree_predictor_model = DecisionTreeRegressor(random_state=1)
tree_predictor_model.fit(features, target)

print(tree_predictor_model.predict(features.head()))
