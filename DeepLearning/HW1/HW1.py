import os
os.chdir('C:/Users/theov/anaconda3/.spyder-py3/NCKU/DeepLearning/HW1/TinyImageNet')
import numpy as np
from numpy import linalg as LA
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from skimage import exposure
from natsort import natsorted
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras import Sequential
import warnings

# dd=os.listdir('TIN')
# f1 = open('train.txt', 'w')
# f2 = open('test.txt', 'w')
# for i in range(len(dd)):
#     d2 = os.listdir ('TIN/%s/images/'%(dd[i]))
#     d2 = natsorted(d2)
#     for j in range(200): #len(d2)-2
#         str1='TIN/%s/images/%s'%(dd[i], d2[j])
#         f1.write("%s %d\n" % (str1, i))
#     str1='TIN/%s/images/%s'%(dd[i], d2[-1])
#     f2.write("%s %d\n" % (str1, i))

# f1.close()
# f2.close()

def create_codebook(f, num_visual_words=128):
    f = open(f)
    lines = f.readlines()
    features = []

    for i in range(0, len(lines), 5):
        fn, label = lines[i].split(' ')
        img = cv2.imread(fn)
        img = cv2.resize(img, (256, 256))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        descriptors = brisk_feature_extraction(img_gray)
        if descriptors is None:
            continue
        features.append(descriptors)

    # Use KMeans to cluster features into visual words (codebook)
    print("KMeans")
    kmeans = KMeans(n_clusters=num_visual_words)
    gg = [item for sublist in features for item in sublist]
    kmeans.fit(gg)

    # Get the cluster centers as the visual words (codebook)
    codebook = kmeans.cluster_centers_

    return codebook

def brisk_feature_extraction(image):
    # Create BRISK detector with specific parameters
    brisk = cv2.BRISK_create() #, scaleFactor=scaleFactor, nlevels=nlevels,
                        #, firstLevel=firstLevel, WTA_K=WTA_K, patchSize=patch_size
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = brisk.detectAndCompute(image, None)
    
    # Return the image with keypoints (or descriptors if needed)
    return descriptors

def load_img(f, processing_choice=None, vocabulary_size=128, codebook=None):
    f = open(f)
    lines = f.readlines()
    imgs, lab = [], []
    
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        
        im1 = cv2.imread(fn)
        
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        ==============================='''
        if processing_choice == 0:
            # Apply HoG feature extraction
            im1 = cv2.resize(im1, (64, 64))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im1 = hog_feature_extraction(im1_gray)
        elif processing_choice == 1:
            im1 = cv2.resize(im1, (256, 256))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            descriptors = brisk_feature_extraction(im1_gray)
            if descriptors is None:
                descriptors = np.zeros((1, 64))
            
            # Assign descriptors to visual words in the codebook
            labels = np.argmin(np.linalg.norm(descriptors[:, None, :] - codebook[None, :, :], axis=2), axis=1)
        
            # Build histogram representing BoW feature
            hist, _ = np.histogram(labels, bins=np.arange(vocabulary_size + 1), density=True)
            
            im1 = hist
            
        elif processing_choice == 2:
            # Apply color histogram feature extraction
            im1 = cv2.resize(im1, (256, 256))
            im1 = color_histogram_extraction(im1)
            
        elif processing_choice == 3:
            imgs.append(im1) 
            lab.append(int(label))
            continue
           
        vec = np.reshape(im1, [-1])
        imgs.append(vec) 
        lab.append(int(label))
            
    imgs = np.asarray(imgs, np.float32)
    lab = np.asarray(lab, np.int32)
    return imgs, lab

def hog_feature_extraction(image):
    # Set HoG parameters (you may adjust these based on your requirements)
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Create HoG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HoG features
    features = hog.compute(image)
    
    # Return the HoG features
    return features

# Function to compute color histogram features
def color_histogram_extraction(image):
    # Split the image into RGB channels
    channels = cv2.split(image)

    # Initialize variables to store histogram features
    hist_features = []

    # Compute histogram for each channel
    for channel in channels:
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist_features.extend(hist.flatten())

    # Convert the histogram features to a numpy array
    hist_features = np.array(hist_features)

    return hist_features

# Load and preprocess data
X_train_hog, y_train = load_img('train.txt', processing_choice=0)  # HOG feature extraction
X_test_hog, y_test = load_img('test.txt', processing_choice=0)

# To fix different dimension problem, has to first create codebook
vocabulary_size=200
codebook = create_codebook('train.txt', vocabulary_size)
X_train_brisk, _ = load_img('train.txt', processing_choice=1, vocabulary_size=vocabulary_size, codebook=codebook)  # ORB feature extraction
X_test_brisk, _ = load_img('test.txt', processing_choice=1, vocabulary_size=vocabulary_size, codebook=codebook)

X_train_hist, _ = load_img('train.txt', processing_choice=2)  # Color histogram feature extraction
X_test_hist, _ = load_img('test.txt', processing_choice=2)

##Apply PCA
num_pca_components = 100  # Adjust the number of components as needed

# Reshape HOG features to be suitable for PCA
X_train_reshaped_hog = X_train_hog.reshape((X_train_hog.shape[0], -1))
X_test_reshaped_hog = X_test_hog.reshape((X_test_hog.shape[0], -1))
# Apply PCA
pca_hog = PCA(n_components=num_pca_components)
X_train_pca_hog = pca_hog.fit_transform(X_train_reshaped_hog)
X_test_pca_hog = pca_hog.transform(X_test_reshaped_hog)
cumulative_variance_ratio = np.cumsum(pca_hog.explained_variance_ratio_)
print(f"Cum Variance Ratio (HoG): {cumulative_variance_ratio[-1]: .3f}")

# X_train_reshaped_orb = X_train_orb.reshape((X_train_orb.shape[0], -1))
# X_test_reshaped_orb = X_test_orb.reshape((X_test_orb.shape[0], -1))
# # Apply PCA
# pca_orb = PCA(n_components=num_pca_components)
# X_train_pca_orb = pca_orb.fit_transform(X_train_reshaped_orb)
# X_test_pca_orb = pca_orb.transform(X_test_reshaped_orb)
# cumulative_variance_ratio = np.cumsum(pca_orb.explained_variance_ratio_)
# print(f"Cum Variance Ratio (ORB): {cumulative_variance_ratio[-1]: .3f}")

X_train_reshaped_hist = X_train_hist.reshape((X_train_hist.shape[0], -1))
X_test_reshaped_hist = X_test_hist.reshape((X_test_hist.shape[0], -1))
# Apply PCA
pca_hist = PCA(n_components=num_pca_components)
X_train_pca_hist = pca_hist.fit_transform(X_train_reshaped_hist)
X_test_pca_hist = pca_hist.transform(X_test_reshaped_hist)
cumulative_variance_ratio = np.cumsum(pca_hist.explained_variance_ratio_)
print(f"Cum Variance Ratio (Hist): {cumulative_variance_ratio[-1]: .3f}")

# Concatenate features
X_train = np.concatenate((X_train_pca_hog, X_train_brisk, X_train_pca_hist), axis=1)
X_test = np.concatenate((X_test_pca_hog, X_test_brisk, X_test_pca_hist), axis=1)

# Classification models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'SVM': SVC(random_state=42),
    'MLP': MLPClassifier(max_iter=10000, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    with warnings.catch_warnings():
        # Suppress FutureWarning from scipy.stats.mode
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # Train with HOG features
        # print(f'Model: {model_name}')
        # model.fit(X_train_pca_hog, y_train)
        # predictions_hog = model.predict(X_test_pca_hog)
        # accuracy_hog = accuracy_score(y_test, predictions_hog)
        # f1_hog = f1_score(y_test, predictions_hog, average='weighted')
        # print(f'{model_name} Accuracy with HOG Features: {accuracy_hog}')
        # print(f'{model_name} F1-score with HOG Features: {f1_hog:.3f}')
    
        # # Train with BRISK features
        # model.fit(X_train_brisk, y_train)
        # predictions_brisk = model.predict(X_test_brisk)
        # accuracy_brisk = accuracy_score(y_test, predictions_brisk)
        # f1_brisk = f1_score(y_test, predictions_brisk, average='weighted')
        # print(f'{model_name} Accuracy with BRISK Features: {accuracy_brisk}')
        # print(f'{model_name} F1-score with BRISK Features: {f1_brisk:.3f}')
        
        # # Train with color histograms
        # model.fit(X_train_pca_hist, y_train)
        # predictions_hist = model.predict(X_test_pca_hist)
        # accuracy_hist = accuracy_score(y_test, predictions_hist)
        # f1_hist = f1_score(y_test, predictions_hist, average='weighted')
        # print(f'{model_name} Accuracy with Color Histograms: {accuracy_hist}')
        # print(f'{model_name} F1-score with Color Histograms: {f1_hist:.3f}')
        
        # Train with concatenated features
        model.fit(X_train, y_train)
        predictions_conc = model.predict(X_test)
        accuracy_conc = accuracy_score(y_test, predictions_conc)
        f1_conc = f1_score(y_test, predictions_conc, average='weighted')
        print(f'{model_name} Accuracy with Concatenated Feature: {accuracy_conc}')
        print(f'{model_name} F1-score with Concatenated Feature: {f1_conc:.3f}')

# Learning-based feature extraction methods. Using CNN
X_train_cnn, y_train = load_img('train.txt', processing_choice=3)
X_test_cnn, y_test = load_img('test.txt', processing_choice=3)
# Preprocess the data (normalize pixel values to be between 0 and 1)
X_train_cnn, X_test_cnn = X_train_cnn / 255.0, X_test_cnn / 255.0
# Split the data into training and testing sets
train_images, val_images, train_labels, val_labels = train_test_split(X_train_cnn, y_train, test_size=0.3, random_state=42)

# Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(200, activation='softmax'))  # Adjust output size based on your task
    return model

# Create the CNN model
input_shape = train_images.shape[1:]
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))

# Predict probabilities for the test set
probabilities_cnn = model.predict(X_test_cnn)

# Convert probabilities to class labels
predictions_cnn = np.argmax(probabilities_cnn, axis=1)
accuracy_cnn = accuracy_score(y_test, predictions_cnn)
f1_cnn = f1_score(y_test, predictions_cnn, average='weighted')
print(f'CNN Accuracy: {accuracy_cnn}')
print(f'CNN F1-score: {f1_cnn:.3f}')