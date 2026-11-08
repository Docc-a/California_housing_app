# California Housing Price Predictor

A machine learning app that predicts median house value in California block groups.

- Model: KNeighborsRegressor tuned with GridSearchCV (5-fold CV)
- Dataset: California Housing (scikit-learn)
- Test R²: 0.7221
- Built with Python, scikit-learn and Streamlit

Files:
- app.py: the Streamlit app
- california_knn_pipeline.joblib: the saved trained pipeline
- requirements.txt: libraries needed to run the app
