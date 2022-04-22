# ------ libraries ------ #
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

    # make average of months for each year by month
    avg_1994 = avg_month(year_1994)
    avg_1995 = avg_month(year_1995)
    avg_1996 = avg_month(year_1996)
    avg_1997 = avg_month(year_1997)
    avg_1998 = avg_month(year_1998)
    avg_1999 = avg_month(year_1999)
    avg_2000 = avg_month(year_2000)
    avg_2001 = avg_month(year_2001)
    avg_2002 = avg_month(year_2002)
    avg_2003 = avg_month(year_2003)
    avg_2004 = avg_month(year_2004)
    avg_2005 = avg_month(year_2005)
    avg_2006 = avg_month(year_2006)
    avg_2007 = avg_month(year_2007)
    avg_2008 = avg_month(year_2008)
    avg_2009 = avg_month(year_2009)
    avg_2010 = avg_month(year_2010)
    avg_2011 = avg_month(year_2011)
    avg_2012 = avg_month(year_2012)
    avg_2013 = avg_month(year_2013)
    avg_2014 = avg_month(year_2014)
    avg_2015 = avg_month(year_2015)
    avg_2016 = avg_month(year_2016)
    avg_2017 = avg_month(year_2017)
    avg_2018 = avg_month(year_2018)
    avg_2019 = avg_month(year_2019)
    avg_2020 = avg_month(year_2020)
    avg_2021 = avg_month(year_2021)

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

    # convert to the same base year price
    # avgMinMaxAll = basePrice(avgMinMaxAll)
    # print(avgMinMaxAll)

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
    for j in range(iterations):
        for i in range(batches):
            batchTempX = x[i]
            batchTempY = y[i]
            for j in range(len(batchTempX)):
                b = B_vals(batchTempX[j], batchTempY[j], w, eps, c_val)
                w = np.multiply((1 - learning_rate), w) + (b * batchTempX[j])
                loss = e_loss(w, batchTempX[j], batchTempY[j], eps)
                l = len(batchTempX)
                print("The cost is:", SVR_cost(w, c_val, l, loss))
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
    for i in range(iterations):
        current_error = rmse(x, w, y)
        if i > 1 and previous_error <= current_error:
            break
        for j in range(l):
            b = B_vals(x[j], y[j], w, eps, c_val)
            w = np.multiply((1 - learning_rate), w) + (b * x[j])
            loss = e_loss(w, x[j], y[j], eps)
            current_cost = SVR_cost(w, c_val, l, loss)
            if i >1 and previous_cost <= current_cost:
                break
            previous_cost = current_cost
            # print("The cost:", current_cost)
        previous_error = current_error
        # Error = rmse(x, w, y)
        print("The Error:", current_error)
    return w


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

    '''
    The best one: 
    e = 0.00001
    w_ = [80, 100]
    w_0 = np.array(w_)
    C = .00001
    LR = 0.00001'''

    e = 0.00001
    w_ = [70, 100]
    w_0 = np.array(w_)
    C = .000001
    LR = 0.00001

    X_batch = np.array_split(X_vals, 26)
    X_batch = np.array(X_batch)
    Y_batch = np.array_split(Y_vals, 26)
    Y_batch = np.array(Y_batch)

    W = SVR_linear_sgdNoBatch(X_vals, Y_vals, e, w_0, C, LR, 600)
    print(W)

    X_vals = X_vals.reshape(1, -1)
    W = W.reshape(-1, 1)

    Y_vals = Y_vals.reshape(-1, 1)

    rmse = rmse(X_vals, W, Y_vals)
    print("The rmse:", rmse)

    Y_vals_pred = prediction(X_vals, W)

    x_m= []
    for i in range(25):
        x_m.append(i)
    x_M = np.array(x_m)

    equation = x_M*W[0] + W[1]

    plt.scatter(X_vals, Y_vals)
    plt.plot(x_M, equation, label = "Regression", color = "red")
    plt.legend()
    plt.show()

    plt.plot(Y_vals, label="Actual Values", color = "green")
    plt.plot(Y_vals_pred[0], label="My Prediction", color = "blue")
    plt.legend()
    plt.show()
