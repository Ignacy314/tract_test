import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import backend as K
import shutil
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6'
dev=tf.config.list_physical_devices('GPU')
print(len(dev))


source_directory = '/mnt/d/wsl/spectrograms_48kHz_5s'

source_test_directory = '/mnt/d/wsl/spectrograms_48kHz_5s_test'

train_directory = '/mnt/d/wsl/wsl_1/train_data'

test_directory = '/mnt/d/wsl/wsl_1/test_data'

nb_current_drone_train_samples=50000
nb_current_background_train_samples=50000
nb_current_drone_test_samples=5000
nb_current_background_test_samples=5000

def sampleData( source_directory, source_test_directory, train_directory, test_directory,
                nb_current_drone_train_samples, nb_current_background_train_samples, 
                nb_current_drone_test_samples, nb_current_background_test_samples
               ):
    print("Copying data samples")
    nb_train_drone_samples = len(os.listdir(source_directory+"/1"))
    nb_train_background_samples = len(os.listdir(source_directory+"/0"))
        
    #nb_current_drone_train_samples = int( nb_train_drone_samples/5)
    #nb_current_background_train_samples = int( nb_train_background_samples/5 )

    train_drone_indices = np.random.randint(0,nb_train_drone_samples,size=nb_current_drone_train_samples)
    train_background_indices = np.random.randint(0,nb_train_background_samples,size=nb_current_background_train_samples)    
      

    nb_test_drone_samples = len(os.listdir(source_test_directory+"/1"))
    nb_test_background_samples = len(os.listdir(source_test_directory+"/0"))
    #nb_current_drone_test_samples = int(nb_current_drone_train_samples/5)
    #nb_current_background_test_samples = int(nb_current_background_train_samples/5)

    test_drone_indices = np.random.randint(0,nb_test_drone_samples,size=nb_current_drone_test_samples)
    test_background_indices = np.random.randint(0,nb_test_background_samples,size=nb_current_background_test_samples)
      
    i=-1
    image_list=os.listdir(source_directory+"/0/")
    for  img_name in image_list:
        i+=1
        img=image.load_img(source_directory+"/0/"+img_name)
        if (img.width == 224 and img.height == 224) and (i in train_background_indices) :
            shutil.copyfile(source_directory+"/0/"+img_name,train_directory+"/0/"+img_name)
            
    i=-1
    image_list=os.listdir(source_directory+"/1/")
    for  img_name in image_list:
        i+=1
        img=image.load_img(source_directory+"/1/"+img_name)
        if (img.width == 224 and img.height == 224) and (i in train_drone_indices) :
            shutil.copyfile(source_directory+"/1/"+img_name,train_directory+"/1/"+img_name)
            
    i=-1
    image_list=os.listdir(source_test_directory+"/1/")
    for  img_name in image_list:
        i+=1
        img=image.load_img(source_test_directory+"/1/"+img_name)
        if (img.width == 224 and img.height == 224) and (i in test_drone_indices) :
            shutil.copyfile(source_test_directory+"/1/"+img_name,test_directory+"/1/"+img_name)
       
    i=-1
    image_list=os.listdir(source_test_directory+"/0/")
    for  img_name in image_list:
        i+=1
        img=image.load_img(source_test_directory+"/0/"+img_name)
        if (img.width == 224 and img.height == 224) and (i in test_drone_indices) :
            shutil.copyfile(source_test_directory+"/0/"+img_name,test_directory+"/0/"+img_name)
    print("Data samples are copied")
'''
sampleData(source_directory, source_test_directory, train_directory, test_directory,
    nb_current_drone_train_samples, nb_current_background_train_samples, 
    nb_current_drone_test_samples, nb_current_background_test_samples)


image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = image_generator.flow_from_directory(batch_size = 40, directory = train_directory, shuffle = True, target_size = (224,224), class_mode = 'categorical', subset = 'training')

validation_generator = image_generator.flow_from_directory(batch_size = 40, directory = train_directory, shuffle = True, target_size = (224,224), class_mode = 'categorical', subset = 'validation')

train_images, train_labels = next(train_generator)

#train_images.shape

#train_labels

label_names = { 0: 'background', 1: 'drone'}

L = 6
W = 6

fig , axes = plt.subplots(L,W,figsize = (12,12))
axes = axes.ravel() #ravel used to flatten the axis

for i in np.arange(0, L * W):
  axes[i].imshow(train_images[i])
  axes[i].set_title(label_names[np.argmax(train_labels[i])])
  axes[i].axis('off')

plt.subplots_adjust(wspace = 0.5)

basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (224,224,3)))

basemodel.summary()

for layer in basemodel.layers[:-10]:
  layers.trainable = False
  
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
headmodel = Flatten(name = 'flatten')(headmodel)
headmodel = Dense(128, activation = 'relu')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = 'relu')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(2, activation = 'softmax')(headmodel)

model = Model(inputs = basemodel.input,outputs = headmodel)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 1e-4), metrics = ['accuracy'])

# using early stopping to exit training if validation loss is not decreasing after certain number of epochs (patience)
earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = 'weights.keras', save_best_only=True)

train_generator = image_generator.flow_from_directory(batch_size = 4, directory = train_directory, shuffle = True, target_size = (224,224), class_mode = 'categorical', subset = 'training')
val_generator = image_generator.flow_from_directory(batch_size = 4, directory = train_directory, shuffle = True, target_size = (224,224), class_mode = 'categorical', subset = 'validation')

history = model.fit(train_generator, epochs = 25, validation_data = val_generator, callbacks = [checkpointer, earlystopping])

history.history.keys()

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy and Loss')

plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')  

plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

prediction = []
original = []
image = []

for i in range(len(os.listdir(test_directory))):
  for item in os.listdir(os.path.join(test_directory,str(i))):
    img = cv2.imread(os.path.join(test_directory,str(i),item))
    img = cv2.resize(img,(224,224))
    image.append(img)
    img = img/255
    img = img.reshape(-1,224,224,3)
    predict = model.predict(img)
    predict = np.argmax(predict)
    prediction.append(predict)
    original.append(i)
    
score = accuracy_score(original, prediction)
print('Test Accuracy: {}'.format(score))   

L = 5
W = 5

fig , axes = plt.subplots(L,W,figsize = (12,12))
axes = axes.ravel() #ravel used to flatten the axis

for i in np.arange(0, L * W):
  axes[i].imshow(train_images[i])
  axes[i].set_title('Guess{}\nTrue={}'.format(str(label_names[prediction[i]]),str(label_names[original[i]])))
  axes[i].axis('off')

plt.subplots_adjust(wspace = 1.2)

print(classification_report(np.asarray(original), np.asarray(prediction)))

cm = confusion_matrix(np.array(original),np.array(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')

model.save("./r50_spec.keras")
'''


model = tf.keras.models.load_model("./r50_spec.keras")


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

prediction = []
original = []
image = []

for i in range(len(os.listdir(test_directory))):
  for item in os.listdir(os.path.join(test_directory,str(i))):
    img = cv2.imread(os.path.join(test_directory,str(i),item))
    img = cv2.resize(img,(224,224))
    image.append(img)
    img = img/255
    img = img.reshape(-1,224,224,3)
    predict = model.predict(img)
    predict = np.argmax(predict)
    prediction.append(predict)
    original.append(i)
    
score = accuracy_score(original, prediction)
print('Test Accuracy: {}'.format(score))   
