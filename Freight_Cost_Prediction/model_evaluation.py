from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train, max_depth = 5):
    model = DecisionTreeRegressor(
        max_depth = max_depth,
        random_state = 42
    )
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, max_depth = 6):
    model = RandomForestRegressor(
        max_depth = max_depth,
        random_state = 42
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, X_test, Y_test, model_name):
    preds = model.predict(X_test)

    mae = mean_absolute_error(Y_test, preds)
    rmse =root_mean_squared_error(Y_test,preds)
    r2 = r2_score(Y_test, preds)*100

    print(f"\n{model_name} Performance: ")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}%")

    return {
        "model_name": model_name,
        "MAE":mae,
        "RMSE":rmse,
        "R2":r2
    } 

    