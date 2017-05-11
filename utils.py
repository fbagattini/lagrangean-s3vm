from __future__ import division
#
import numpy as np
from PIL import Image
#
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
from sklearn.metrics import euclidean_distances
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

def get_coil(objects,
             l,
             u,
             rdm=np.random.RandomState(),
             source='datasets/COIL/'):
    """
    loads a COIL20 classification task (distinguish between objects[0] and objects[1])
    """
    label = -1
    samples = []
    for obj in objects:
        obj_folder_name = source+'%d/' % obj
        for image_index in range(72):
            obj_img = Image.open(obj_folder_name+'obj%d__%d.png' % (obj, image_index))
            rescaled = obj_img.resize((20,20))
            pixels_values = [float(x) for x in list(rescaled.getdata())]
            sample = np.array(pixels_values + [label])
            samples.append(sample)
        label+=2
    rdm.shuffle(samples)
    samples = np.array(samples)
    data = samples[:, :-1]
    targets = samples[:, -1]
    targets = np.array([int(y) for y in targets])
    mmsc = MinMaxScaler(feature_range=(0,1))
    data = mmsc.fit_transform(data)
    return data[:l], targets[:l], data[l:l+u], data[l+u:], targets[l+u:]

def get_best_estimator_by_cv(data, targets, folds, C=[2**i for i in range(0, 5)],
                                                   gamma=[],
                                                   kernel=['rbf']):
    """
    used to initialize the inner supervised routine;
    returns a cross-validated SVC object which has been fitted with the selected hyper-parameters
    """
    if not gamma : gamma += [1/data.shape[1]]
    positives = len([y for y in targets if y==1])
    if positives == 0 or positives == len(data) : return SVC(C=1, gamma=1/len(data[0]))
    params_grid = [
      {'C': C,
       'gamma': gamma,
       'kernel': kernel}
    ]
    gs = GridSearchCV(SVC(), params_grid, n_jobs=-1, cv=folds)
    gs.fit(data, targets)
    best_estimator = gs.best_estimator_
    return best_estimator