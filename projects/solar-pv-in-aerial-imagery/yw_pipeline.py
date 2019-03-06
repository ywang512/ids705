'''
Pipeline
------------------------------------------------
load data 		|	fin
preprocess 		|	R


Yifei Wang
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import scipy.signal as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten

plt.close()


dir_train_images  = './data/training/'
dir_test_images   = './data/testing/'
dir_train_labels  = './data/training_labels/labels_training.csv'
dir_test_ids      = './data/sample_submission.csv'

SUBMISSION = False
KFOLD = 20
IMG_SIZE = 101



def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory 

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.tif'
        image     = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids


def preprocess_and_extract_features(data):
    '''Preprocess data and extract features
    
    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    # Here, we do something trivially simple: we take the average of the RGB
    # values to produce a grey image, transform that into a vector, then
    # extract the mean and standard deviation as features.
    
    # Make the image grayscale

    return data
    '''
    R = data[:,:,:,0]
    Nimg = R.shape[0]
    Npix = R.shape[1] * R.shape[2]

    data = R.reshape(Nimg, Npix)

    return data
	'''

def preprocess_red(data, red=False):
	Nimg = data.shape[0]
	Npix = data.shape[1] * data.shape[2]
	Npixcolor = data.shape[1] * data.shape[2] * data.shape[3]
	if red == True:
		R = data[:,:,:,0]
		data = R.reshape(Nimg, Npix)
		return data
	elif red == False:
		return data.reshape(Nimg, Npixcolor)

def add_RelLum(data):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  returns[:,:,:,-1] = (0.2126 * data[:,:,:,0] + 0.7152 * data[:,:,:,1] + 0.0722 * data[:,:,:,2]) / 255.0
  #print(returns.shape)
  return returns

def Gaussian(sigma):
    alpha = math.ceil(3.5 * sigma)
    n = np.arange(-alpha, alpha+1)
    g = np.exp(-0.5 * n**2 / sigma**2)
    k = 1/np.sum(g)
    return k*g

def dGaussian(sigma):
    alpha = math.ceil(3.5 * sigma)
    n = np.arange(-alpha, alpha+1)
    d = n * np.exp(-0.5 * n**2 / sigma**2)
    k = -1/np.dot(n, d)
    return k*d

def gradient(img, sigma=0.1):
    '''img is a 2-dimensional numpy array representing a black-and-white image'''
    
    g = Gaussian(sigma)
    gx = np.expand_dims(g, 0)
    gy = np.expand_dims(g, 1)
    d = dGaussian(sigma)
    dx = np.expand_dims(d, 0)
    dy = np.expand_dims(d, 1)
    
    # derivative along x
    Ix = ss.convolve2d(ss.convolve2d(img, dx, mode = 'same'), gy, mode = 'same')

    # derivative along y
    Iy = ss.convolve2d(ss.convolve2d(img, gx, mode = 'same'), dy, mode = 'same')
    
    return (Ix, Iy)

def add_gradient(data, sigma=0.1):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  for i, img in enumerate(data):
    Ix, Iy = gradient(img[:,:,0], sigma=sigma)
    G = np.sqrt(Ix**2+Iy**2)
    returns[i,:,:,-1] = G
  #print(returns.shape)
  return returns


# Not a great idea for using PCA
def classifier_pca(X, n_components):
	'''used for seeing pricipal components'''
	clf = PCA(n_components = 150)
	clf.fit_transform(X)
	#print(clf.explained_variance_ratio_)
	#print(np.sum(clf.explained_variance_ratio_))
	#print(len(clf.explained_variance_ratio_))
	#print(clf.singular_values_)
	V = clf.components_
	'''
	fig, axs = plt.subplots(3,3, figsize = (9, 9))
	axf = axs.flatten()
	for v, ax in zip(V, axf):
		ax.imshow(v.reshape(101,101))
	plt.show()
	'''
	return clf




def set_classifier(classifier):

	'''Shared function to select the classifier for both performance evaluation
	and testing '''

	model = Sequential()
	model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2))) 
	model.add(BatchNormalization())

	model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(2, activation = 'softmax'))


	model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=['accuracy'])

	return model
	'''
    if classifier == "pca":
    	return PCA()
    else:
        return KNeighborsClassifier(n_neighbors=7)
	'''

def pca_lr_cv_assessment(X,y,n,k,red=False, RelLum=False, Grad=False):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.zeros(y.shape[0],dtype='float')
    prediction_class = np.zeros(y.shape[0],dtype='int')

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    #ACC = []
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train, y_val   = y[train_index], y[val_index]
        
        # Train the classifier
        # X_train_features = preprocess_and_extract_features(X_train)
        if RelLum:
        	X_train = add_RelLum(X_train)
        if Grad:
        	X_train = add_gradient(X_train)
        X_train = preprocess_red(X_train, red=red)
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X_train)
        lr = LogisticRegression(solver='lbfgs', max_iter=500).fit(X_pca, y_train)

        # Test the classifier on the validation data for this fold
        if RelLum:
        	X_val = add_RelLum(X_val)
        if Grad:
        	X_val = add_gradient(X_val)
        X_val = preprocess_red(X_val, red=red)
        X_val_features   = pca.transform(X_val)

        prediction_scores[val_index] = lr.predict_proba(X_val_features)[:,1]
        prediction_class[val_index] = lr.predict(X_val_features)

        #print(acc*100)
        #ACC.append(acc)
        # Save the predictions for this fold
        #prediction_scores[val_index] = cpred[:,1]
    return prediction_scores, prediction_class


def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = roc_curve(labels, prediction_scores, pos_label=1)
    auc = roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)
   
    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_RGB(ids, imgs, labels):
	n = len(ids)
	fig, axs = plt.subplots(3, n, figsize = (3*n,9))
	axs = axs.T

	for i, ax in zip(ids, axs):
		img0 = imgs[i,:,:,0]
		img1 = imgs[i,:,:,1]
		img2 = imgs[i,:,:,2]
		img = [img0, img1, img2]
		
		for x, imgg in zip(ax, img):
			x.imshow(imgg, cmap = 'gray')
			title = str(i) + "_" + str(labels[i])
			x.set_title(title)
			x.axis('off')
	plt.tight_layout()
	plt.show()


def output_submission():
	# Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features              = preprocess_and_extract_features(training_data)
    clf                            = set_classifier()
    clf.fit(training_features,training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features  = preprocess_and_extract_features(test_data)
    test_scores    = clf.predict_proba(test_features)[:,1]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                   'score':  test_scores})
    submission_file.to_csv('submission.csv',
                           columns=['id','score'],
                           index=False)

def extract_pca(X):
	clf = PCA(n_components = 150)
	clf.fit(X)
	return clf



def rf_cv_assessment(X,y,n,k, red=False, RelLum=False, Grad=False):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.zeros(y.shape[0],dtype='float')
    prediction_class = np.zeros(y.shape[0],dtype='int')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    #ACC = []
    cnt=0
    for train_index, val_index in kf.split(X, y):
        cnt+=1
        if cnt % 2 == 0:
	        print("starting %d-fold:" % cnt)
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train, y_val   = y[train_index], y[val_index]
        
        # Train the classifier
        if RelLum:
        	X_train = add_RelLum(X_train)
        if Grad:
        	X_train = add_gradient(X_train)
        X_train = preprocess_red(X_train, red=red)
        rf = RandomForestClassifier(n_estimators=n).fit(X_train, y_train)

        if RelLum:
        	X_val = add_RelLum(X_val)
        if Grad:
        	X_val = add_gradient(X_val)
        X_val = preprocess_red(X_val, red=red)
        prediction_scores[val_index] = rf.predict_proba(X_val)[:,1]
        prediction_class[val_index] = rf.predict(X_val)

        #print(acc*100)
        #ACC.append(acc)
        # Save the predictions for this fold
        #prediction_scores[val_index] = cpred[:,1]
    return prediction_scores, prediction_class




#====================================================================================================

def main():

	imgs, labels = load_data(dir_train_images, dir_train_labels, training=True)
	data = preprocess_red(imgs)


	''' # pca + logistic + cv = {ACC: 0.68, AUC: 0.68}
	for i in np.arange(10,100,10):
		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10,red=False)
		print("no preprocess, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)

		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10,red=True)
		print("take red channel, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)
	'''


	'''
	for i in np.arange(225,301,25):
		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10)
		print("Original, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)

		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10, RelLum=True)
		print("add RelLum, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)

		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10, Grad=True)
		print("add Gradient, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)

		pre_score, pre_class = pca_lr_cv_assessment(imgs,labels,n=i,k=10, Grad=True, RelLum=True)
		print("add Gradient, pca = ",i)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*50)
	'''


	''' # random forest = 100~300{ACC: 0.715~0.728, AUC: 0.77~0.787, time: 125~377}
	n = 300
	pre_score, pre_class = rf_cv_assessment(data,labels,n,k=10)
	print("Random Forest, trees = ",n)
	print("ACC:   ", accuracy_score(labels, pre_class))
	print("AUC:   ", roc_auc_score(labels, pre_score))
	'''

	print("STARTED")
	print("-"*40)
	for n in range(300, 301, 100):
		pre_score, pre_class = rf_cv_assessment(imgs,labels,n,k=10)
		print("RF - trivial, trees = ",n)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("-"*40)

		pre_score, pre_class = rf_cv_assessment(imgs,labels,n,k=10, RelLum=True)
		print("RF - RelLum, trees = ",n)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("-"*40)

		pre_score, pre_class = rf_cv_assessment(imgs,labels,n,k=10, Grad=True)
		print("RF - Gradient, trees = ",n)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("-"*40)

		pre_score, pre_class = rf_cv_assessment(imgs,labels,n,k=10, RelLum=True, Grad=True)
		print("RF - RelLum + Gradient, trees = ",n)
		print("ACC:   ", accuracy_score(labels, pre_class))
		print("AUC:   ", roc_auc_score(labels, pre_score))
		print("="*40)


	if SUBMISSION:
		output_submission()


if __name__ == '__main__':
	main()




















