# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras


def main():
    a_min = 15
    a_max = 80
    s_min = 1 #Thousand
    s_max = 15 #Thousands
    J = 15000
    K = 10000
    age = np.random.randint(a_min, a_max + 1, K+J)
    salary = np.random.uniform(s_min, s_max, K+J)
    sal_selfemp = np.random.randint(0, 2, K+J)
    chi = np.random.uniform(0   ,   1,  K+J)
    vec = []
    for i in range(K+J):
        vec.append([age[i], salary[i], sal_selfemp[i]])

    print("The empirical means are", ex01(vec))
    #The case a_1 = a_2 = a_3 = 0 implies that the function p(x) becomes constant and is equal to sigmoid(a0). The models based on these features will have no predictive power.The Roc curve will hence be diagonal.
    vec_train = vec[:J]
    vec_test = vec[J:]
    y=ex02_1(age, salary, sal_selfemp, chi)
    y_train = y[:J]
    y_test = y[J:]
    p_test = p
    print("Default rate", np.mean(y)*100)
    log_Reg(vec_train, y_train, vec_test, y_test)
    ex02_2(vec_train, y_train, vec_test, y_test)

def ex01(vec_J):
    mean_values_J = np.array(vec_J).mean(axis=0)
    return mean_values_J

def ex02_1(age_K, salary_K, sal_selfemp_K, chi):
    y=[]
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    a0 = -10
    a1 = 3
    a2 = -7
    a3 = 15
    p=sigmoid(get_p(age_K, salary_K, sal_selfemp_K, a0, a1, a2, a3))
    print(get_p(age_K, salary_K, sal_selfemp_K, a0, a1, a2, a3))
    for i in range(len(p)):
        y.append(y_Val(chi[i], p[i]))

    return y
def ex02_2(vec_train, y_train,  vec_test, y_test):
    # Create and train neural network

    # build the nn model with 2 hidden layers
    model_NN = keras.Sequential(
        [
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu', name="hidden_layer_1"),
            keras.layers.Dense(32, activation='relu', name="hidden_layer_2"),
            keras.layers.Dense(1, activation='sigmoid', name="output_layer")
        ]
    )

    # the loss function
    def total_deviance(y_true, y_pred):
        return tf.math.reduce_mean(-y_true * tf.math.log(y_pred) - (1. - y_true) * tf.math.log(1. - y_pred))

    # pick an optimizer
    model_NN.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=total_deviance
    )

    # shuffle the data during training
    train_dataset = tf.data.Dataset.from_tensor_slices((vec_train, np.float32(y_train)))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(256)

    # fit the model
    model_NN.fit(
        train_dataset,
        batch_size=256,
        epochs=10
    )
    scores_NN_train = model_NN(vec_train)
    scores_NN = model_NN(vec_test)

    print('total deviance for training data = ' ,log_loss(y_train, scores_NN_train))
    print('total deviance for test data = ' ,log_loss(y_test, scores_NN))
    comp_NN = np.stack((scores_NN.numpy()[:, 0], y_test), axis=-1)
    select = (comp_NN[:, 0] <= 0.04) & (comp_NN[:, 0] >= 0.03)
    comp_NN_select = comp_NN[select, :]
    print('fraction of y_j=1: ' + str(np.mean(comp_NN_select[:, 1])))
def ex03(K):
    #Apparently the Neural Network model has a lower deviance , hence we weill use the NN model
    p = K
    loan = 1000
    interest = 0.04

    defaults = np.float32(np.random.uniform(low=0., high=1., size=(p, 50000)) < p_test[:portfolio_size, :])
    defaultR_ins = np.mean(y[:J])
    threshold = np.quantile(scores_NN_train, 1. - 2 * defaultR_ins)
    # threshold = 0.2 # To compare to an arbitrary threshold
    print(threshold)
    p_and_l_NN = np.sum(loan_amount * (rate * (1. - defaults) - defaults) * np.float32(scores_NN < threshold), axis=0)

    pct_VaR = 95
    index = int(((100 - pct_VaR) * n_replic / 100) - 1)
    print('Expected P&L (NN): {:.2f}'.format(np.mean(p_and_l_NN)))

    p_and_l_b_NN = np.sort(p_and_l_NN)
    print('VaR (NN): {:.2f}'.format(-p_and_l_b_NN[index]))

    def to_opt(threshold):
        to_give_a_loan = scores_NN_train.numpy() < threshold
        pred_not_default = to_give_a_loan[to_give_a_loan] * 1  # Looks weird is correct
        # print(pred_not_default.shape)
        true_defaults = y_train[to_give_a_loan[:, 0]]
        return np.absolute(np.average((pred_not_default - true_defaults) == 0))

    res = minimize(to_opt, np.array(0.12), bounds=((0, 1),), method="SLSQP", options={'eps': 0.01})
    print(res)
    p_and_l_NN = np.sum(loan_amount * (rate * (1. - defaults) - defaults) * np.float32(scores_NN < res.x), axis=0)

    plt.figure(2)
    plt.hist(p_and_l_NN, bins=100, label='NN')
    plt.legend()
    pct_VaR = 95
    index = int(((100 - pct_VaR) * n_replic / 100) - 1)
    print('Expected P&L (NN): {:.2f}'.format(np.mean(p_and_l_NN)))

    p_and_l_b_NN = np.sort(p_and_l_NN)
    print('VaR (NN): {:.2f}'.format(-p_and_l_b_NN[index]))
def log_Reg(vec_train, y_train, vec_test, y_test):

    # logitic regression for the linear dataset
    model_LR = LogisticRegression().fit(vec_train, y_train)

    # a look at the coefficients
    print(model_LR.coef_)
    print(model_LR.intercept_)
    # calculate the scores on the linear testset
    scores_LR_train = model_LR.predict_proba(vec_train)
    scores_LR = model_LR.predict_proba(vec_test)

    # calculate total deviance for training and test data.
    # Note that scores_LR_1_train[:, 1] is the predicted default probability
    print('total deviance for training data = ' + str(log_loss(y_train, scores_LR_train[:, 1])))
    print('total deviance for test data = ' + str(log_loss(y_test, scores_LR[:, 1])))
    # plot the roc curve
    conc_LR = np.stack((scores_LR[:, 1], y_test), axis=-1)
    select = (conc_LR[:, 0] <= 0.04) & (conc_LR[:, 0] >= 0.03)
    comp_LR_select = conc_LR[select, :]
    print('fraction of y_j=1: ' + str(np.mean(comp_LR_select[:, 1])*100) + '%')

def y_Val(chi,p):
    if chi <= p:
        y = 1
    else :
        y = 0
    return y

def get_p(x0,x1,x2,a0,a1,a2,a3):
    px = (a0+a1*np.sqrt(np.abs(x0-55))+a2*x1+a3*x2)
    return px

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
