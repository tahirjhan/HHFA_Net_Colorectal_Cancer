# Import libraries
import tensorflow.compat.v1 as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
seed = 232
np.random.seed(seed)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("CUDA working?...")
print(tf.test.is_built_with_cuda())

from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

print("Data directory...")
cfolder = os.getcwd()
print(cfolder)
input_path = cfolder

def process_data(img_dims, batch_size):
    if len(img_dims)>1:
        r = img_dims[0]; c = img_dims[1]
    else:
        r = img_dims, c = img_dims

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_gen = test_datagen.flow_from_directory(
        directory=os.path.join(input_path, 'Testing'),
        target_size=(r, c),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    return test_gen

img_dims = [150,150]
batch_size = 32

test_gen = process_data(img_dims, batch_size)

np.set_printoptions(precision=4)
num_batches = test_gen.samples // batch_size
accm = np.zeros([num_batches])
model = load_model('xyz.h5')
#model= model.load_weights('xyz.hdf5')
for i in range(num_batches):
    print('testing batch number: {}/{}'.format(i + 1, num_batches))
    x, y = test_gen.next()
    yp = model.predict(x)
    ya = np.argmax(y, axis=1)
    ypa = np.argmax(yp, axis=1)
    if i == 0:
        ytrue = y;
        ypred = yp
    else:
        ytrue = np.concatenate((ytrue, y))
        ypred = np.concatenate((ypred, yp))
    n_true = np.where(ya == ypa)[0]
    accm[i] = len(n_true) / len(ya)

preds = np.argmax(ypred,axis=1)
test_lab = np.zeros([ytrue.shape[0]])
for i in range(ytrue.shape[0]):
    test_lab[i] = np.where(ytrue[i,:]==1)[0]

accuracy = accm.mean()
print('Accuracy: '+ str(accuracy))
conf_mat = confusion_matrix(test_lab, preds)
print("Confusion Matrix ...")
print(conf_mat)

# Print f1, precision, and recall scores
precision = precision_score(test_lab, preds , average='weighted')
recall = recall_score(test_lab, preds , average='weighted')
F1 = 2 * ((precision * recall) / (precision + recall))

print('overall F1 score is: '+ str(F1))

#weighted F score
wighted_fscore= precision_recall_fscore_support(test_lab, preds, average='weighted')

print('Weighted F score is: '+ str(wighted_fscore))

print("Print classification report:")
target_names = ['Benign','Complex Stroma','Debris','Inflammatory','Muscle','Stroma','Tumor']
print(classification_report(test_lab,preds,target_names=target_names))