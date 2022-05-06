# ------ libraries ------ #
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- Functions ---- #
def get_year_list(year):
    month = 1
    list = []
    for i in range(0, 12):
        if month < 10:
            string = str(year) + "-0" + str(month) + "-01"
            list.append(string)
            month += 1
        else:
            string = str(year) + "-" + str(month) + "-01"
            list.append(string)
            month += 1
    return list


def month_list_fun(year_list, num):
    vals = year_list[year_list['DATE'].dt.month == num]
    to_return = vals.loc[:, 'DEXMXUS']
    # remove values that are 0

    return to_return


def avg_month(year_list):
    count = 1
    year_avg = [0] * 12
    i = 0
    while count <= 12:
        vals = month_list_fun(year_list, count)
        val = np.sum(vals)
        val_avg = val / len(vals)
        year_avg[i] = val_avg
        count += 1
        i += 1
    return year_avg


def matrix_avg_min_max(year_list):
    count = 1
    avg_min_max = np.empty((12, 3))
    i = 0
    while count <= 12:
        vals = month_list_fun(year_list, count)
        min = np.amin(vals)
        max = np.amax(vals)
        val = np.sum(vals)
        val_avg = val / len(vals)
        avg_min_max[i, 0] = val_avg
        avg_min_max[i, 1] = min
        avg_min_max[i, 2] = max
        count += 1
        i += 1
    return avg_min_max


def x_var(years):
    amount = years * 12
    array = [0] * amount
    count = 1
    for i in range(amount):
        if count <= 12:
            array[i] = count
            count += 1
        else:
            array[i] = 1
            count = 2
    return array


def data():
    # USD to Pesos CSV file
    data = pd.read_csv("USD_to_Peso.csv")
    # format this correctly: we want to formate one column as the date
    # we want to convert the numbers into floats, so we can process them
    data['DATE'] = pd.to_datetime(data['DATE'].str.strip(), format='%Y/%m/%d')
    data['DEXMXUS'].replace({".": "0"}, inplace=True)
    data['DEXMXUS'] = pd.to_numeric(data['DEXMXUS'], downcast="float")

    # ensures that the dates which the DEXMXUS val is 0 is
    # dropped to avoid the min getting messed up
    target = data[data['DEXMXUS'] > 0]

    # separate data by year
    year_1994 = target[target['DATE'].dt.year == 1994]
    year_1995 = target[target['DATE'].dt.year == 1995]
    year_1996 = target[target['DATE'].dt.year == 1996]
    year_1997 = target[target['DATE'].dt.year == 1997]
    year_1998 = target[target['DATE'].dt.year == 1998]
    year_1999 = target[target['DATE'].dt.year == 1999]
    year_2000 = target[target['DATE'].dt.year == 2000]
    year_2001 = target[target['DATE'].dt.year == 2001]
    year_2002 = target[target['DATE'].dt.year == 2002]
    year_2003 = target[target['DATE'].dt.year == 2003]
    year_2004 = target[target['DATE'].dt.year == 2004]
    year_2005 = target[target['DATE'].dt.year == 2005]
    year_2006 = target[target['DATE'].dt.year == 2006]
    year_2007 = target[target['DATE'].dt.year == 2007]
    year_2008 = target[target['DATE'].dt.year == 2008]
    year_2009 = target[target['DATE'].dt.year == 2009]
    year_2010 = target[target['DATE'].dt.year == 2010]
    year_2011 = target[target['DATE'].dt.year == 2011]
    year_2012 = target[target['DATE'].dt.year == 2012]
    year_2013 = target[target['DATE'].dt.year == 2013]
    year_2014 = target[target['DATE'].dt.year == 2014]
    year_2015 = target[target['DATE'].dt.year == 2015]
    year_2016 = target[target['DATE'].dt.year == 2016]
    year_2017 = target[target['DATE'].dt.year == 2017]
    year_2018 = target[target['DATE'].dt.year == 2018]
    year_2019 = target[target['DATE'].dt.year == 2019]
    year_2020 = target[target['DATE'].dt.year == 2020]
    year_2021 = target[target['DATE'].dt.year == 2021]

    # this is the average + min + max of that month
    avgMinMax_1994 = matrix_avg_min_max(year_1994)
    avgMinMax_1995 = matrix_avg_min_max(year_1995)
    avgMinMax_1996 = matrix_avg_min_max(year_1996)
    avgMinMax_1997 = matrix_avg_min_max(year_1997)
    avgMinMax_1998 = matrix_avg_min_max(year_1998)
    avgMinMax_1999 = matrix_avg_min_max(year_1999)
    avgMinMax_2000 = matrix_avg_min_max(year_2000)
    avgMinMax_2001 = matrix_avg_min_max(year_2001)
    avgMinMax_2002 = matrix_avg_min_max(year_2002)
    avgMinMax_2003 = matrix_avg_min_max(year_2003)
    avgMinMax_2004 = matrix_avg_min_max(year_2004)
    avgMinMax_2005 = matrix_avg_min_max(year_2005)
    avgMinMax_2006 = matrix_avg_min_max(year_2006)
    avgMinMax_2007 = matrix_avg_min_max(year_2007)
    avgMinMax_2008 = matrix_avg_min_max(year_2008)
    avgMinMax_2009 = matrix_avg_min_max(year_2009)
    avgMinMax_2010 = matrix_avg_min_max(year_2010)
    avgMinMax_2011 = matrix_avg_min_max(year_2011)
    avgMinMax_2012 = matrix_avg_min_max(year_2012)
    avgMinMax_2013 = matrix_avg_min_max(year_2013)
    avgMinMax_2014 = matrix_avg_min_max(year_2014)
    avgMinMax_2015 = matrix_avg_min_max(year_2015)
    avgMinMax_2016 = matrix_avg_min_max(year_2016)
    avgMinMax_2017 = matrix_avg_min_max(year_2017)
    avgMinMax_2018 = matrix_avg_min_max(year_2018)
    avgMinMax_2019 = matrix_avg_min_max(year_2019)
    avgMinMax_2020 = matrix_avg_min_max(year_2020)
    avgMinMax_2021 = matrix_avg_min_max(year_2021)

    avgMinMaxAll = np.vstack(
        (avgMinMax_1994, avgMinMax_1995, avgMinMax_1996, avgMinMax_1997, avgMinMax_1998, avgMinMax_1999,
         avgMinMax_2000, avgMinMax_2001, avgMinMax_2002, avgMinMax_2003, avgMinMax_2004, avgMinMax_2005,
         avgMinMax_2006, avgMinMax_2007, avgMinMax_2008, avgMinMax_2009, avgMinMax_2010, avgMinMax_2011,
         avgMinMax_2012, avgMinMax_2013, avgMinMax_2014, avgMinMax_2015, avgMinMax_2016, avgMinMax_2017,
         avgMinMax_2018, avgMinMax_2019, avgMinMax_2020, avgMinMax_2021))

    x = x_var(28)
    x = np.array(x)

    date_1994 = get_year_list(1994)
    date_1995 = get_year_list(1995)
    date_1996 = get_year_list(1996)
    date_1997 = get_year_list(1997)
    date_1998 = get_year_list(1998)
    date_1999 = get_year_list(1999)
    date_2000 = get_year_list(2000)
    date_2001 = get_year_list(2001)
    date_2002 = get_year_list(2002)
    date_2003 = get_year_list(2003)
    date_2004 = get_year_list(2004)
    date_2005 = get_year_list(2005)
    date_2006 = get_year_list(2006)
    date_2007 = get_year_list(2007)
    date_2008 = get_year_list(2008)
    date_2009 = get_year_list(2009)
    date_2010 = get_year_list(2010)
    date_2011 = get_year_list(2011)
    date_2012 = get_year_list(2012)
    date_2013 = get_year_list(2013)
    date_2014 = get_year_list(2014)
    date_2015 = get_year_list(2015)
    date_2016 = get_year_list(2016)
    date_2017 = get_year_list(2017)
    date_2018 = get_year_list(2018)
    date_2019 = get_year_list(2019)
    date_2020 = get_year_list(2020)
    date_2021 = get_year_list(2021)

    dates = np.concatenate(
        (date_1994, date_1995, date_1996, date_1997, date_1998, date_1999, date_2000, date_2001, date_2002,
         date_2003, date_2004, date_2005, date_2006, date_2007, date_2008, date_2009, date_2010, date_2011,
         date_2012, date_2013, date_2014, date_2015, date_2016, date_2017, date_2018, date_2019, date_2020,
         date_2021), axis=None)

    df = pd.DataFrame(avgMinMaxAll)
    df.insert(0, "Month", x, True)
    df.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    df.drop(columns=['Month'], axis=1, inplace=True)
    df['Month'] = dates
    return df


def B_vals(x, y, w, eps, c_val):
    wT = w.reshape(-1, 1)
    xR = x.reshape(1, -1)
    if np.linalg.norm(np.abs(np.dot(wT, xR) - y)) <= eps:
        return 0
    if np.linalg.norm(np.dot(wT, xR) - y) > eps:
        return -c_val
    if np.linalg.norm(np.dot(wT, xR) - y) < -eps:
        return c_val


def e_loss(w, x, y, eps):
    wT = w.reshape(-1, 1)
    xR = x.reshape(1, -1)
    return np.max(np.linalg.norm(np.abs(np.dot(wT, xR)) - y) - eps, 0)


def SVR_cost(w, c, l, loss):
    w2 = w.reshape(1, -1)
    norm = np.linalg.norm(w2)
    return .5 * (norm * norm) + (c / l) * loss


def SVR_linear_sgd(x, y, eps, w, c_val, learning_rate, iterations):
    batches = len(x)
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    for j in range(iterations):
        for i in range(batches):
            batchTempX = x[i]
            batchTempY = y[i]
            current_error = rmse(batchTempX, w, batchTempY)
            print("the rsme is:", current_error)
            for j in range(len(batchTempX)):
                b = B_vals(batchTempX[j], batchTempY[j], w, eps, c_val)
                w = np.multiply((1 - learning_rate), w) + (b * batchTempX[j])
    return w


def each_data(x, w):
    return x * w[0] + w[1]


def rmse(x, w, y):
    pred = []
    for i in range(len(x)):
        pred.append(each_data(x[i], w))
    inside = y - pred
    total = np.linalg.norm(inside) ** 2
    return np.sqrt(total / len(x))


def prediction(x, w):
    pred = []
    for i in range(len(x)):
        pred.append(each_data(x[i], w))
    return pred


def SVR_linear_sgdNoBatch(x, y, eps, w, c_val, learning_rate, iterations):
    l = len(x)
    previous_cost = None
    previous_error = None
    count_iters = 0
    reached = False
    for i in range(iterations):
        current_error = rmse(x, w, y)
        if i > 1 and previous_error <= current_error:
            break
        for j in range(l):
            b = B_vals(x[j], y[j], w, eps, c_val)
            w = np.multiply((1 - learning_rate), w) + (b * x[j])
            loss = e_loss(w, x[j], y[j], eps)
            current_cost = SVR_cost(w, c_val, l, loss)
            if i > 1 and previous_cost <= current_cost:
                break
            previous_cost = current_cost
        previous_error = current_error
        # print("The Error:", current_error)
        count_iters += 1
    if iterations == count_iters:
        reached == True
    return w, reached, current_error, current_cost


def iterate_C_values(x, y, eps, w, start_c_val, learning_Rate, iters):
    count = 0
    rsme = []
    cost = []
    reached = []
    W_vects = []
    while start_c_val > .000000001:
        print(start_c_val)
        W, reach, CE, CC = SVR_linear_sgdNoBatch(x, y, eps, w, start_c_val, learning_Rate, iters)
        rsme.append(CE)
        cost.append(CC)
        reached.append(reach)
        W_vects.append(W)
        start_c_val = start_c_val * .1
        count += 1
    return rsme, cost, reached, W_vects, count


def iterate_LR_values(x, y, eps, w, c_val, min_LR, increment, max_LR, iters):
    count = 0
    rsme = []
    cost = []
    reached = []
    W_vects = []
    LR = []
    while min_LR < max_LR:
        LR.append(min_LR)
        W, reach, CE, CC = SVR_linear_sgdNoBatch(x, y, eps, w, c_val, min_LR, iters)
        rsme.append(CE)
        cost.append(CC)
        reached.append(reach)
        W_vects.append(W)
        min_LR = min_LR + increment
        count += 1
    return rsme, cost, reached, W_vects, count, LR


def iterate_C_adj_values(x, y, eps, w, learning_rate, min_C, increment, max_C, iters):
    count = 0
    rsme = []
    cost = []
    reached = []
    W_vects = []
    C_vals = []
    while min_C < max_C:
        C_vals.append(min_C)
        w, reach, CE, CC = SVR_linear_sgdNoBatch(x, y, eps, w, min_C, learning_rate, iters)
        rsme.append(CE)
        cost.append(CC)
        reached.append(reach)
        W_vects.append(w)
        min_C = min_C + increment
        count += 1
    return rsme, cost, reached, W_vects, count, C_vals


if __name__ == '__main__':
    currency = data()

    prediction_days = 12
    currency['Prediction'] = currency[['Average']].shift(-prediction_days)

    X = np.array(currency.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    X = X[:len(currency) - prediction_days]
    X_vals = np.array(X)
    Y = np.array(currency['Prediction'])
    Y = Y[:-prediction_days]
    Y_vals = np.array(Y)

    # ---- Checking the ideal value of C and the Learning Rate ---- #
    e = 0.00001
    w_ = [80, 100]
    w_0 = np.array(w_)
    LR = 0.1
    C = .1

    error1, cost1, statement1, support_vectors1, number1 = iterate_C_values(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(error1)
    print(cost1)
    print(support_vectors1)

    x_axis = []
    for i in range(number1 - 1):
        x_axis.append(i)

    C_vals_array = ['0.01', '1.0e-3', '1.0e-4', '1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9']

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error1[1:], label="LR = 0.1")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----
    LR = 0.01
    C = .1

    error01, cost01, statement01, support_vectors01, number01 = iterate_C_values(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(error01)
    print(cost01)
    print(support_vectors01)

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error01[1:], label="LR = 0.01")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----
    LR = 0.001
    C = .1

    error001, cost001, statement001, support_vectors001, number001 = iterate_C_values(X_vals, Y_vals, e, w_0, C, LR,
                                                                                      600)
    print(error001)
    print(cost001)
    print(support_vectors001)

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error001[1:], label="LR = 0.001")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----
    LR = 0.0001
    C = .1
    error0001, cost0001, statement0001, support_vectors0001, number0001 = iterate_C_values(X_vals, Y_vals, e, w_0, C,
                                                                                           LR, 600)
    print(error0001)
    print(cost0001)
    print(support_vectors0001)

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error0001[1:], label="LR = 0.0001")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----
    LR = 0.00001
    C = .1
    error00001, cost00001, statement00001, support_vectors00001, number00001 = iterate_C_values(X_vals, Y_vals, e, w_0,
                                                                                                C, LR, 600)
    print(error00001)
    print(cost00001)
    print(support_vectors00001)

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error00001[1:], label="LR = 0.00001")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----
    LR = 0.000001
    C = .1
    error000001, cost000001, statement000001, support_vectors000001, number000001 = iterate_C_values(X_vals, Y_vals, e,
                                                                                                     w_0, C, LR, 600)
    print(error000001)
    print(cost000001)
    print(support_vectors000001)

    plt.title(f'RSME - C val, Learning Rate {LR}')
    plt.plot(x_axis, error000001[1:], label="LR = 0.000001")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ----- Visual Representation ------ #
    x_axis = []
    for i in range(number00001):
        x_axis.append(i)

    C_vals_array = ['0.1', '0.01', '1.0e-3', '1.0e-4', '1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9']

    plt.title("RSME - C val, Learning Rates")
    plt.plot(x_axis, error000001, label="LR = 0.000001")
    plt.plot(x_axis, error00001, label="LR = 0.00001")
    plt.plot(x_axis, error0001, label="LR = 0.0001")
    plt.plot(x_axis, error001, label="LR = 0.001")
    plt.plot(x_axis, error01, label="LR = 0.01")
    plt.plot(x_axis, error1, label="LR = 0.1")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.legend()
    plt.show()

    x_axis = []
    for i in range(number00001-1):
        x_axis.append(i)

    C_vals_array = ['0.01', '1.0e-3', '1.0e-4', '1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9']

    plt.title("RSME - C val, Learning Rates")
    plt.plot(x_axis, error000001[1:], label="LR = 0.000001")
    plt.plot(x_axis, error00001[1:], label="LR = 0.00001")
    plt.plot(x_axis, error0001[1:], label="LR = 0.0001")
    plt.plot(x_axis, error001[1:], label="LR = 0.001")
    plt.plot(x_axis, error01[1:], label="LR = 0.01")
    plt.plot(x_axis, error1[1:], label="LR = 0.1")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.legend()
    plt.show()

    x_axis = []
    for i in range(number00001 - 3):
        x_axis.append(i)

    C_vals_array = ['1.0e-4', '1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9']

    plt.title("RSME - C val, Learning Rates")
    plt.plot(x_axis, error0001[3:], label="LR = 0.0001")
    plt.plot(x_axis, error001[3:], label="LR = 0.001")
    plt.plot(x_axis, error01[3:], label="LR = 0.01")
    plt.plot(x_axis, error1[3:], label="LR = 0.1")
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.legend()
    plt.show()

    x_axis = [0,1]
    C_vals_array = ['1.0e-4', '1.0e-5']
    plot = [error00001[3], error00001[4]]
    plt.title("RSME - C val, Learning Rate: 0.00001")
    plt.plot(x_axis, plot)
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    x_axis = [0, 1]
    C_vals_array = ['1.0e-3', '1.0e-4']
    plot = [error000001[2], error000001[3]]
    plt.title("RSME - C val, Learning Rate: 0.000001")
    plt.plot(x_axis, plot)
    plt.xticks(x_axis, C_vals_array)
    plt.xlabel("C Value")
    plt.ylabel("RSME Size")
    plt.show()

    # ---- Valuating the range of C and LR---- #
    e = 0.00001
    w_ = [80, 100]
    w_0 = np.array(w_)

    # --- C = 0.0001 , LR = 0.001 --- #
    LR = 0.001
    C = 0.00001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error,".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 5, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.00001 , LR = 0.001 --- #
    LR = 0.001
    C = 0.000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 5, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

# --- C = 0.000001 , LR = 0.001 --- #
    LR = 0.001
    C = 0.0000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error,".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 8, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # ================================#
    # --- C = 0.0001 , LR = 0.0001 --- #
    LR = 0.0001
    C = 0.00001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 5, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.00001 , LR = 0.0001 --- #
    LR = 0.0001
    C = 0.000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 5, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.000001 , LR = 0.0001 --- #
    LR = 0.0001
    C = 0.0000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 10, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # ================================#
    # --- C = 0.0001 , LR = 0.00001 --- #
    LR = 0.00001
    C = 0.0001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(16, 12.5, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.00001 , LR = 0.0001 --- #
    LR = 0.00001
    C = 0.00001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 10, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.000001 , LR = 0.0001 --- #
    LR = 0.00001
    C = 0.000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 50, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # ================================#
    # --- C = 0.0001 , LR = 0.000001 --- #
    LR = 0.000001
    C = 0.001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, -10, f'Cost: {cost}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.00001 , LR = 0.0001 --- #
    LR = 0.000001
    C = 0.0001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 6, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # --- C = 0.000001 , LR = 0.0001 --- #
    LR = 0.000001
    C = 0.00001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    error = format(error, ".2f")
    cost = format(cost, ".5f")

    Y_vals_pred = prediction(X_vals, W)

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * W[0] + W[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'TRAINING MODEL: Currency Exchange, C: {C}, LR: {LR}')
    plt.text(15, 200, f'RSME: {error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # there were our best
    C = 0.00001
    max_L = 0.0001  # under estimate
    min_L = 0.00001  # over estimate

    error, cost, statement, support_vectors, number, learning_rate_array = iterate_LR_values(X_vals, Y_vals, e, w_0, C, min_L, 0.000001, max_L, 600)

    x_axis = []
    for i in range(number):
        x_axis.append(i)

    plt.title("RSME- C = 1.0e-5, Learning Rates Ranging: 0.00001 - 0.0001")
    plt.plot(learning_rate_array, error)
    plt.xlabel("LR Value")
    plt.ylabel("RSME Size")
    plt.show()

    min_error = min(error)
    min_index_error = error.index(min_error)

    SV_error = support_vectors[min_index_error]
    SV_LR_error = learning_rate_array[min_index_error]

    min_error = format(min_error, ".2f")
    SV_LR_error = format(SV_LR_error, ".7f")

    x_m = []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M * SV_error[0] + SV_error[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'Min RSME W vector, C: {C}, LR: {SV_LR_error}')
    plt.text(13.5, 10, f'RSME: {min_error}')
    plt.text(13.5, 8, f'W: {SV_error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    # ---- other value of C ---- #
    C = 0.000001
    max_L = 0.0001  # under estimate
    min_L = 0.00001  # over estimate

    error, cost, statement, support_vectors, number, learning_rate_array = iterate_LR_values(X_vals, Y_vals, e, w_0, C, min_L, 0.000001, max_L, 600)

    plt.title("RSME- C = 1.0e-6, Learning Rates Ranging: 0.00001 - 0.0001")
    plt.plot(learning_rate_array, error)
    plt.xlabel("LR Value")
    plt.ylabel("RSME Size")
    plt.show()

    min_error = min(error)
    min_index_error = error.index(min_error)

    SV_error = support_vectors[min_index_error]
    SV_LR_error = learning_rate_array[min_index_error]

    min_error = format(min_error, ".2f")
    SV_LR_error = format(SV_LR_error, ".7f")

    equation = x_M * SV_error[0] + SV_error[1]
    plt.scatter(X_vals, Y_vals)
    plt.title(f'Min RSME W vector, C: {C}, LR: {SV_LR_error}')
    plt.text(13.5, 10, f'RSME: {min_error}')
    plt.text(13.5, 8, f'W: {SV_error}')
    plt.plot(x_M, equation, label="Regression", color="red")
    plt.ylabel("Today's value")
    plt.xlabel("Previous Day value")
    plt.legend()
    plt.show()

    X_v = currency[['Average']]
    pred_vals = X_v[324:]
    pred_vals = np.array(pred_vals)

    C = 0.00001
    LR = 0.0000110
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    x_m = []
    for i in range(12):
        x_m.append(i)
    x_M = np.array(x_m)

    plt.title(f'Currency Rate for 2021, C: {C}, LR: {LR}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    C = 0.000001
    LR = 0.0000220
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    x_m = []
    for i in range(12):
        x_m.append(i)
    x_M = np.array(x_m)

    plt.title(f'Currency Rate for 2021, C: {C}, LR: {LR}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

# ---- try different values of epsilon---- #
    C = 0.000001
    LR = 0.0000220
    e = 0.1
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    x_m = []
    for i in range(12):
        x_m.append(i)
    x_M = np.array(x_m)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    e = 0.01
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    e = 0.001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    e = 0.0001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    e = 0.00001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)
    print(prediction_values)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

    e = 0.000001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)

    plt.title(f'Currency Rate for 2021, epsilon: {e}')
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label="Our Predicted values", color="red")
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()

# ----- final ----- #

    C = 0.000001
    LR = 0.0000220
    e = 0.0001
    W, state, error, cost = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)

    prediction_values = prediction(pred_vals, W)

    x_m = []
    for i in range(12):
        x_m.append(i)
    x_M = np.array(x_m)

    y_real = np.array(pred_vals)
    y_hat = np.array(prediction_values)

    inside = y_real - y_hat
    total = np.linalg.norm(inside) ** 2
    rsme = np.sqrt(total / len(y_real))
    rsme = format(rsme, ".7f")

    plt.title(f'Currency Rate for 2021, C: {C}, LR: {LR}, epsilon: {e}', size = 13)
    plt.plot(x_M, pred_vals, label="The Actual values", color="green")
    plt.plot(x_M, prediction_values, label='Our Predicted values', color="red")
    plt.text(3.5, 20.75, f'RSME: {rsme}', size=13)
    plt.ylabel("Pesos per Dollar")
    plt.xlabel("Months")
    plt.legend()
    plt.show()



