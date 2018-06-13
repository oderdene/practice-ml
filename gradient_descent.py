import numpy as np
import matplotlib.pyplot as plt
import xlrd
import requests

def gradient_descent_runner(points, starting_m, starting_b, learning_rate, num_iterations):
    # регрессийн цэгүүдийг зурж харуулах
    plt.xlabel('Гал гаралтын тоо хэмжээ')
    plt.ylabel('Гэмт хэргийн гаралтын тоо')
    plt.scatter(points[:, 0], points[:, 1])
    x_plot = np.linspace(0, 50, 100)
    plt.ion() # графикийг тусдаа процесстой цонх болгон харуулах

    m = starting_m
    b = starting_b

    for i in range(num_iterations):
        m, b = step_gradient(m, b, np.array(points), learning_rate)
        print("m=", m, " and b=", b)

        # регрессийн шулууныг w болон b утгуудыг ашиглан зурах
        plt.xlabel('Гал гаралтын тоо хэмжээ')
        plt.ylabel('Гэмт хэргийн гаралтын тоо')
        plt.scatter(points[:, 0], points[:, 1])
        plt.plot(x_plot, x_plot*m + b)
        plt.show()
        plt.pause(0.001)
        plt.gcf().clear()

    return m, b


def step_gradient(current_m, current_b, points, learning_rate):
    # gradient descent
    m_gradient = 0
    b_gradient = 0

    # calculate optimal values for model

    # to calculate the gradient we need to calculate
    # the partial derivative of m and b
    n = float(len(points))

    sum_m = 0
    sum_b = 0

    for point in points:
        sum_m += -1 * point[0] * (point[1] - (current_m * point[0] + current_b))
        sum_b += -1 * (point[1] - (current_m * point[0] + current_b))

    m_gradient = (2 / n) * sum_m
    b_gradient = (2 / n) * sum_b

    m_new = current_m - (learning_rate * m_gradient)
    b_new = current_b - (learning_rate * b_gradient)

    return m_new, b_new



def compute_error_for_given_points(m, b, points):
    # sum of squared errors
    sum_error = 0
    for i in range(len(points)):
        point = points[i]
        sum_error += (point[1] - (m*point[0] + b)) ** 2
    return sum_error / float(len(points))


def run():
    # өгөгдлүүдээ http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr05.html
    # хаяг дээрхи Чикаго хотын гал болон гэмт хэргийн гаралтын хамаарлын
    # судалгааг агуулсан excel файлаас татаж авч уншаад numpy массив болгож хадгална
    # data_url = "http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/excel/slr05.xls"
    # u = requests.get(data_url)
    # book = xlrd.open_workbook(file_contents=u.content, encoding_override="utf-8")
    # sheet = book.sheet_by_index(0)
    # data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

    # дээрх data г data.csv файлд хадгалав
    points = np.genfromtxt("data.csv", delimiter=",")

    # Hyperparameter - tuning knobs in ml for our model
    # learning rate is how fast our model learn
    # too low is will be slow to converge
    # too high it will never converge
    # converge means finding the optimal values for our function

    learning_rate = 0.001

    # y = mx + b (slope formula) m = slope and b = y-intercept
    initial_b = 0
    initial_m = 0
    initial_error = compute_error_for_given_points(initial_m, initial_b, points)

    print("Starting gradient descent m=", initial_m, " and b=", initial_b, " with an error=", initial_error)

    # depends on the size of the dataset
    num_iterations = 1000

    [m, b] = gradient_descent_runner(points, initial_m, initial_b, learning_rate, num_iterations)

    error = compute_error_for_given_points(m, b, points)

    print("After 1000 iterations m=", m, " and b=", b, " with an error=")

if __name__ == "__main__":
    print("opening from terminal")
    run()