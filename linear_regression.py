import numpy as np


def calc_hat_y(x, w):
    '''
    Calcula y aproximada (y^ = w0*x0 + w1*x1 + ... w_n*x_n)
    '''
    hat_y = 0
    for i in range(len(x)):
        hat_y += w[i]*x[i]
    return hat_y


def calc_mse(y, n, hat_y):
    '''
    Calcula el mean squared error de cada y^ con la y real
    '''
    sum = 0
    for i in range(n):
        squared_err = (y[i] - hat_y)**2
        sum += squared_err
    return sum/n


def gradient_descent(alpha, m, x, y, w):
    '''
    Calcula los ajustes que se tienen que hacer a los parámetros (w) para la recta que
    aproxima los valores de y
    '''
    errors = []
    new_params = w
    for i in range(len(w)):
        sum = 0
        for j in range(len(x)):
            hat_y = calc_hat_y(x[j], w)
            mse = calc_mse(y, m, hat_y)
            errors.append(mse)
            delta_mult = (hat_y - y[j])*x[j][i]
            sum += delta_mult
        new_params[i] = w[i] - ((alpha*sum)/m)
    return new_params


def linear_regresion(x, y, alpha, epochs):
    '''
    Regresión lineal
    '''
    m = len(x)
    # Inicializar todos los parámetros en 1
    w = [0.5 for i in range(len(x[0]))]
    for i in range(m):
        x[i] = [1]+x[i]  # Agregar x0=1 en todas las filas de las variables x
    w.insert(0, 1)

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        w = gradient_descent(alpha, m, x, y, w)
        print(f'Params: {w}\n')

    print('Finished!')
    print('Predicted values for "y" params are:')
    for i, param in enumerate(w):
        print(f'x{i+1}: w{i+1} = {param}')


if __name__ == '__main__':
    x = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    # y = [0.5, 1, 1.5, 2, 2.5]
    y = [15, 15, 15, 15]
    w = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha = 0.2
    m = len(x)
    linear_regresion(x, y, alpha, 10000)
