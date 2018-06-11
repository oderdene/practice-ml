# #!/usr/bin/env python3
#!python3

def predict_sales(radio, weight, bias):
    return weight*radio + bias


def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (predict_sales(weight,radio[i], bias)))**2
    return total_error / companies
