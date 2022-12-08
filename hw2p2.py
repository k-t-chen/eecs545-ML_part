import numpy as np
import numpy.matlib

def hw2():
    train_x = np.load("hw2p2_train_x.npy")
    train_y = np.load("hw2p2_train_y.npy")
    test_x = np.load("hw2p2_test_x.npy")
    test_y = np.load("hw2p2_test_y.npy")


    train_n, d = train_x.shape 
    test_n = test_x.shape[0]

    train_y = np.stack((1-train_y, train_y), axis=1)
    y_counts = train_y.sum(axis=0)
    log_y_priors = np.log(y_counts) - np.log(train_n)
    feature_counts = np.dot(train_y.T, train_x,)
    feature_counts = feature_counts + 1.0
    class_counts = feature_counts.sum(axis=1)
    

    log_feature_counts = np.log(feature_counts)
    log_class_counts = np.log(class_counts.reshape(2,1)) 
    log_feature_prob = log_feature_counts - log_class_counts

    log_class_factors = np.dot(test_x, log_feature_prob.T)

    test_likelihoods = log_class_factors + log_y_priors
    pred_test_y = np.argmax(test_likelihoods, axis=1)

    test_error = np.sum(pred_test_y != test_y) / test_n 

    print("The test error based on Naive Bayes classifier is {:.4f}".format(test_error * 100.0))
    if log_y_priors[1] >= log_y_priors[0]:
        maj_test_error = (1-test_y).sum() / test_n
    else:
        maj_test_error = test_y.sum() / test_n
        
    print("The test error based on majority vote is {:.4f}, by choosing class {}".format(maj_test_error * 100.0, 1 if log_y_priors[1]>=log_y_priors[0] else 0))
    return test_error
test_error = hw2()