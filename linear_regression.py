import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc_hat_y(x, w):
    '''
    Calcula y aproximada (y^ = w0*x0 + w1*x1 + ... w_n*x_n)
    '''
    hat_y = 0
    for i in range(len(x)):
        hat_y += w[i]*x[i]
    return hat_y


def calc_mse(y, x, w):
    '''
    Calcula el mean squared error de cada y^ con la y real
    '''
    n = len(y)
    sum = 0
    for i in range(n):
        hat_y = calc_hat_y(x[i], w)
        err = hat_y - y[i]
        sum += err**2
    mean_s_error = sum/n
    mse.append(mean_s_error)


def gradient_descent(alpha, x, y, w):
    '''
    Calcula los ajustes que se tienen que hacer a los parámetros (w) para la recta que
    aproxima los valores de y
    '''
    new_params = w
    for i in range(len(w)):
        sum = 0
        for j in range(len(x)):
            hat_y = calc_hat_y(x[j], w)
            error = (hat_y - y[j])
            sum += error*x[j][i]
        new_params[i] = w[i] - alpha * (1/len(x)) * sum
    return new_params


def linear_regresion(x, y, alpha, epochs):
    '''
    Regresión lineal
    '''
    m = len(x)
    # Inicializar todos los parámetros en 0
    w = [0 for i in range(len(x[0]))]
    for i in range(m):
        x[i] = [1]+x[i]  # Agregar x0=1 en todas las filas de las variables x
    w.insert(0, 1)  # Agregar w0

    for epoch in range(epochs):
        epochs_list.append(epoch)
        w = gradient_descent(alpha, x, y, w)
        calc_mse(y, x, w)

    print('Finished!')
    print('Predicted values for "y" params are:')
    print(f"Error: {mse[-1]}")
    for i, param in enumerate(w):
        print(f'x{i+1}: w{i+1} = {param}')
    return w, mse[-1]


if __name__ == '__main__':
    mse = []
    epochs_list = []
    '''
    Primer entrenamiento con dataset de entrenamiento train1.csv
    '''
    train1 = pd.read_csv('train1.csv')
    y_train1 = train1['width']
    X_train1 = train1.drop(columns='width')
    X_train1 = X_train1.values.tolist()
    y_train1 = y_train1.values.tolist()
    # X = [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9],
    #      [10, 11, 12],
    #      [0.3, 0.7, 0.2],
    #      [0.27, 0.35, 0.57]]
    # y = [2, 4, 7, 10, 0.5, 0.6]
    alpha = 0.0001
    epochs = 100000
    print("Model is training with dataset train1.csv")
    model, error_train1 = linear_regresion(X_train1, y_train1, alpha, epochs)
    xparams = model[1:]

    '''
    Primer prueba con dataset de prueba test1.csv
    '''
    prediction_test1 = []
    test1 = pd.read_csv('test1.csv')
    y_test1 = test1['width']
    X_test1 = test1.drop(columns=['width'])
    X_test1 = X_test1.values.tolist()
    y_test1 = y_test1.values.tolist()

    for i in range(len(y_test1)):
        sum = 0
        for j in range(len(xparams)):
            sum += xparams[j]*X_test1[i][j]
        prediction_test1.append(sum)

    prediction_test1 = np.array(prediction_test1)
    y_test1 = np.array(y_test1)
    y_test1_mean = y_test1.mean()
    numerador = 0
    denominador = 0
    for i in range(len(y_test1)):
        numerador += (prediction_test1[i] - y_test1_mean)**2
        denominador += (y_test1[i] - y_test1_mean)**2
    r_squared_test1 = numerador/denominador
    print("Test with dataset test1.csv")
    print(f"R Squared Test 1: {r_squared_test1}")

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.title('MSE throughout epochs: 1st train')
    plt.plot(epochs_list, mse)
    plt.show()

    '''
    VARIANDO SETS DE ENTRENAMIENTO Y TEST
    '''
    mse = []
    epochs_list = []
    '''
    Primer entrenamiento con dataset de entrenamiento train1.csv
    '''
    train2 = pd.read_csv('train2.csv')
    y_train2 = train2['width']
    X_train2 = train2.drop(columns='width')
    X_train2 = X_train2.values.tolist()
    y_train2 = y_train2.values.tolist()
    # X = [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9],
    #      [10, 11, 12],
    #      [0.3, 0.7, 0.2],
    #      [0.27, 0.35, 0.57]]
    # y = [2, 4, 7, 10, 0.5, 0.6]
    alpha = 0.01
    epochs = 1000
    print("\n\nModel is training with dataset train2.csv")
    model, error_train2 = linear_regresion(X_train2, y_train2, alpha, epochs)
    xparams = model[1:]

    '''
    Primer prueba con dataset de prueba test1.csv
    '''
    prediction_test2 = []
    test2 = pd.read_csv('test2.csv')
    y_test2 = test2['width']
    X_test2 = test2.drop(columns=['width'])
    X_test2 = X_test2.values.tolist()
    y_test2 = y_test2.values.tolist()

    for i in range(len(y_test2)):
        sum = 0
        for j in range(len(xparams)):
            sum += xparams[j]*X_test2[i][j]
        prediction_test2.append(sum)

    prediction_test2 = np.array(prediction_test1)
    y_test2 = np.array(y_test2)
    y_test2_mean = y_test2.mean()
    numerador = 0
    denominador = 0
    for i in range(len(y_test2)):
        numerador += (prediction_test2[i] - y_test2_mean)**2
        denominador += (y_test2[i] - y_test2_mean)**2
    r_squared_test2 = numerador/denominador
    print("Test with dataset test2.csv")
    print(f"R Squared Test 2: {r_squared_test2}")

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.title('MSE throughout epochs: 2nd train')
    plt.plot(epochs_list, mse)
    plt.show()

    '''
    PREDICCIONES
    '''
    data = pd.read_csv('predictions_data.csv')
    data_X = data.drop(columns=['width'])
    data_y = data['width']
    data_X = data_X.values.tolist()
    data_y = data_y.values.tolist()
    print('\n\nPREDICTIONS')
    for i in range(len(data_X)):
        sum = 0
        for j in range(len(xparams)):
            sum += xparams[j]*data_X[i][j]
        print(
            f'Predicted value: {round(sum,4)}    Real value: {round(data_y[i],4)}    Error: {round(round(sum,4)-round(data_y[i],4),4)}')
