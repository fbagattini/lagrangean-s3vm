from __future__ import division
#
import sklearn.utils as skutils
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
import numpy as np

def lagrangian_s3vm_train(xtrain_l,
                          ytrain_l,
                          xtrain_u,
                          svc,
                          r=.5,
                          batch_size=2000,
                          annealing_sequence=[0.1, 0.25, 0.5, 1.0],
                          balance_tolerance=.005,
                          rdm=np.random.RandomState()):
    """
    xtrain_l: a numpy (n=samples, d=feature) array containing labeled data
    ytrain_l: a numpy (n=samples,) array containing the labels of labeled data
    xtrain_u: a numpy (n=samples, d=feature) array containing unlabeled data
    r: float percentage of unlabeled samples to be classified as positive
    batch_size: limits the training batch size in large scale scenarios
    annealing_sequence: sequence of values for C_star (wrt C, so that '0.25' means 'C_star=0.25*C')
    rdm: controls the randomness of the process

    returns: a sklearn.svm.SVC object fitted using both labeled and unlabeled samples
    """
    l = xtrain_l.shape[0]; u = xtrain_u.shape[0]
    beta = min(u, batch_size - l)*(2*r - 1)#constant term
    max_violation = min(u, batch_size - l)*balance_tolerance
    for factor in annealing_sequence:
        #sample: shuffle + slice
        xtrain_u = skutils.shuffle(xtrain_u, random_state=rdm)
        u_batch = xtrain_u[:batch_size - l] if xtrain_u.shape[0] >= batch_size - l else xtrain_u
        #compute labels on the batch
        distances = svc.decision_function(u_batch)
        y_u = lagrangian_heuristic(distances, u_batch.shape[0], beta, max_violation=max_violation)
        #concatenate with batch
        if type(xtrain_l) == csr_matrix : xtrain = vstack([xtrain_l, u_batch])
        else : xtrain = np.concatenate((xtrain_l, u_batch)) 
        ytrain = np.concatenate((ytrain_l, y_u))
        #refit
        sample_weight = l*[1]+min(u, batch_size - l)*[factor]
        svc.fit(xtrain, ytrain, sample_weight=sample_weight)                  
    return svc

def lagrangian_heuristic(distances,        
                         u,                 
                         beta,
                         iterations=100,
                         lam_0=0.0,
                         theta_0=1.0,
                         max_violation=1.0):
    """
    returns the best labeling of the unlabeled points wrt to
    - their distance from the separating hyperplane
    - the relaxed balance constraint
    """
    lam = lam_0
    theta = theta_0
    y_a, y_b = None, None
    max_violation = max(1, max_violation)
    for k in xrange(iterations):
        best_labels = get_best_label(distances, lam)
        violation = np.sum(best_labels) - beta
        if abs(violation) <= max_violation : break
        if violation < 0 : y_a = best_labels  
        else             : y_b = best_labels
        if y_a is not None and y_b is not None : lam = planes_intersection(y_a, y_b, u, distances, beta)
        else:
            lam += theta*violation
            theta = update_theta(theta) 
    return best_labels

def planes_intersection(y_a, y_b, u, distances, beta):
    emp_err_on_y_a = np.sum(np_hinge(np.multiply(distances, y_a)))
    emp_err_on_y_b = np.sum(np_hinge(np.multiply(distances, y_b)))    
    numerator = emp_err_on_y_b - emp_err_on_y_a
    #
    violation_on_y_a = np.sum(y_a)# - beta
    violation_on_y_b = np.sum(y_b)# - beta
    denominator = violation_on_y_a - violation_on_y_b 
    return numerator/denominator
    
def get_best_label(distances, lam):
    plus_1 = np_hinge(distances) + lam
    minus_1 = np_hinge(-distances) - lam
    return 2*(minus_1 > plus_1).astype(int)-1

def update_theta(theta):
    return theta*0.9

def hinge(t):
    return max(0, 1-t)

np_hinge = np.vectorize(hinge, otypes=[np.float])
