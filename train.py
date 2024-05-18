import numpy as np
import random
import cv2
from glob import glob
import matplotlib.pyplot as plt
from imutils import paths
import os
from PIL import Image
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.image import img_to_array
import skimage
import skimage.transform
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
#perform denoising
def denoise(image):

    #denoising using Non-local mean algorithm
    out = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return out


data = []
label = []
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("dataset")))
random.shuffle(img_dir)
print("[INFO]  Preprocessing...")
print("total-->",len(img_dir))
tot=len(img_dir)
count=0
for i in img_dir:
    img = cv2.imread(i)
    img=cv2.resize(img,(128,128))

    # print(img.shape)
    img=denoise(img)
    # convert it to grayscale
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

    # apply histogram equalization 
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # cv2.imshow("equalizeHist", np.hstack((img, hist_eq)))
    # cv2.waitKey(0)
 
    data.append(hist_eq) 
    lb=i.split(os.path.sep)[-2]
    if lb=='MildDemented':
        label.append(0)
    elif lb=='ModerateDemented':
        label.append(1)
    elif lb=='NonDemented':
        label.append(2)
    elif lb=='VeryMildDemented':
        label.append(3)
    print(count,"/",tot)
    count+=1




# pickle.dump(data,open('feats.pkl','wb'))
# pickle.dump(label,open('labels.pkl','wb'))


data=pickle.load(open('feats.pkl','rb'))
labels=pickle.load(open('labels.pkl','rb'))
print("****************")
print(len(data))
print(len(labels))

data=np.array(data)
labels=np.array(labels) 

print(data.shape)
print(labels.shape)


from sklearn.model_selection import train_test_split
#perform train-test splitting
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=0, shuffle=True,test_size=0.2)
print("\nTraining Set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting Set")
print(x_test.shape)
print(y_test.shape)

#perform normalization
x_train=x_train/255
x_test=x_test/255

get_a=x_train.shape[0]
get_b=x_train.shape[1]
get_c=x_train.shape[2]
get_d=x_train.shape[3]

x_train=x_train.reshape(get_a,get_b*get_c*get_d)
print(x_train.shape)


from collections import Counter
from imblearn.combine import SMOTETomek

print('Original dataset shape %s' % Counter(y_train))
smt = SMOTETomek(random_state=42)
x_train, y_train = smt.fit_resample(x_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train))

#perform Label encoding (return binary)
from sklearn.preprocessing import LabelBinarizer
#initialize
label_as_binary = LabelBinarizer()
y_train = label_as_binary.fit_transform(y_train)
y_test = label_as_binary.fit_transform(y_test)

x_train=x_train.reshape(8164,128,128,3)

from model import Custom
#function call
model=Custom()

print(model.summary())

#compiling the mode;
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])

#saving the model
checkpoint=ModelCheckpoint("Project_Saved_Models/trained_model.h5",
                           monitor="acc",
                           save_best_only=True,
                           verbose=1)

#training
history= model.fit(x_train,y_train,epochs=30,batch_size=8,validation_data=(x_test,y_test),callbacks=[checkpoint])

#plotting
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("Project_Extra/acc_plot.png")
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("Project_Extra/loss_plot.png")
plt.show()



# final_loss, final_accuracy = model.evaluate(x_test, y_test)
#### print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))


y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


disease_types=['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented']
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)
plt.show()