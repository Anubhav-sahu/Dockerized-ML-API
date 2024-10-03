from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Loading the dataset and training a model part
data = load_iris()
X, y = data.data, data.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# saving the created model. 
joblib.dump(model, 'model.pkl')
