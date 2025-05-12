import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

def get_user_input(X_columns):
    user_input = {}
    for column in X_columns:
        value = input(f"Введите значение для признака '{column}': ")
        user_input[column] = float(value)  # Предполагаем, что все значения числовые
    return user_input

def main():
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Linear Regression Prediction')
    parser.add_argument('data_file', type=str, help='Path to data file')
    args = parser.parse_args()

    # Загрузка данных из файла
    data = pd.read_csv(args.data_file)
    X = data.drop(columns=['Tensile Elastic Modulus', 'Tensile Strength'])
    y = data['Tensile Elastic Modulus']

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = train_model(X_train, y_train)

    # Оценка модели
    mse, r2, mae = evaluate_model(model, X_test, y_test)

    # Вывод результатов
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)

    # Получение пользовательского ввода и предсказание
    user_input = get_user_input(X.columns)
    user_input_df = pd.DataFrame([user_input])
    prediction = model.predict(user_input_df)
    print("Предсказанное значение:", prediction[0])

if __name__ == "__main__":
    main()
