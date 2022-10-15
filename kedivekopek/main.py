# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import  to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random
import os
from glob import glob

# klasör yolunun gösterilmesi
train=os.listdir("archive (2)\\training_set\\training_set\\kedi_köpek_train")
test=os.listdir("archive (2)\\test_set\\test_set\\kedi_köpek_test")

print(len(train))
print(len(test))

# gerekli sabitlerin belirlenmesi
image_width=128
image_height=128
image_size=(image_width,image_height)
image_channel=3

# eğitim verisinin hazırlanması
filenames=os.listdir("archive (2)\\training_set\\training_set\\kedi_köpek_train")


categories=[]
for filename in filenames:
    category=filename.split(".")[0]
    if category=="dog":
        categories.append(1)
    else:
        categories.append(0)

df=pd.DataFrame({"filename":filenames,"category":categories})

# veriyi görselleştirme
print(df["category"].value_counts())#.plot.bar())

# rasgele bir görüntünün seçilmesi

sample=random.choice(filenames)
image=load_img("archive (2)\\training_set\\training_set\\kedi_köpek_train\\"+sample)
plt.imshow(image)
plt.show()


# modelin oluşturulması

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout, Dense, Activation, BatchNormalization,Flatten

model=Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(image_width,image_height,image_channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))
model.summary()

# modelin derlenmesi
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

# verinin hazırlanması
df["category"]=df["category"].replace({0:"cat",1:"dog"})
train_df,validate_df=train_test_split(df,test_size=0.2)
train_df=train_df.reset_index(drop=True)
validate_df=validate_df.reset_index(drop=True)

# kategorilerin yapılması
train_df["category"].value_counts().plot.bar() # plot gösteriyor

# eğitim ve doğrulama verilerin hazırlanması

total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15

# eğitim verilerin çoğaltılması
train_datagen=ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator=train_datagen.flow_from_dataframe(
    train_df,
    "archive (2)\\training_set\\training_set\\kedi_köpek_train\\",
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode="categorical",
    batch_size=batch_size
)

# doğrulama verilerini çoğaltılması

validation_datagen=ImageDataGenerator(
    rescale=1./255
)

validation_generator=validation_datagen.flow_from_dataframe(
    validate_df,
    "archive (2)\\training_set\\training_set\\kedi_köpek_train\\",
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode="categorical",
    batch_size=batch_size
)

# çoğaltılan veriye bakmak

example_df=train_df.sample(n=1).reset_index(drop=True)
example_generator= train_datagen.flow_from_dataframe(
    example_df,
    "archive (2)\\training_set\\training_set\\kedi_köpek_train\\",
    x_col="filename",
    y_col="category",
    target_size=image_size,
    class_mode="categorical"
)


plt.figure(figsize=(12,12))
for i in range(0,15):
    plt.subplot(5,3,i+1)
    for X_batch in example_generator:
        image=X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# derlenmiş olan modelin eğitilmesi

epoch=3

history=model.fit_generator(
    generator=train_generator,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=total_validate,
    steps_per_epoch=total_train)

# eğitim ve doğrulama verilerinin görüntülenmesi

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,12))
ax1.plot(history.history["loss"],color="b",label="eğitim kaybı")
ax1.plot(history.history["val_loss"],color="r",label="doğrulama kaybı")
ax1.set_xticks(np.arange(1,epoch,1))
ax1.set_yticks(np.arange(0,1,0.1))

ax2.plot(history.history["accuracy"],color="b",label="eğitim başarımı")
ax2.plot(history.history["val_accuracy"],color="r",label="doğrulama başarımı")
ax2.set_xticks(np.arange(1,epoch,1))

legend=plt.legend(log="best",shadow=True)
plt.tight_layout()
plt.show()

# test verisinin hazırlanması

test_filenames=os.listdir("archive (2)\\training_set\\training_set\\kedi_köpek_test\\")
test_df=pd.DataFrame({"filename":test_filenames})
nb_sample=test_df.shape[0]

# test verisinin çoğaltılması

test_gen=ImageDataGenerator(
    rescale=1./255
)

test_generator=validation_datagen.flow_from_dataframe(
    test_df,
    "archive (2)\\training_set\\training_set\\kedi_köpek_test\\",
    x_col="filename",
    y_col="None",
    target_size=image_size,
    class_mode=None,
    batch_size=batch_size,
    shuffle=False
)

# tahmin işleminin yapılması

predict=model.predict_generator(test_generator,steps=np.ceil(nb_sample/batch_size))

# tahmin işleminin hangi kategorye ait olduğunu belirleme

test_df["category"]=np.argmax(predict,axis=-1)
label_map=dict((v,k) for k,v in train_generator.class_indices.items())
test_df["category"]=test_df["category"].replace(label_map)

# tahmin değerlendirmesi

sample_test=test_df.head(18)
sample.head()
plt.figure(figsize=(12,24))

for index, row in sample_test.iterrows():
    filename=row["filename"]
    category=row["category"]
    img=load_img("archive (2)\\training_set\\training_set\\kedi_köpek_test\\"+filename,target_size=image_size)
    plt.subplot(6,3, index+1)
    plt.imshow(img)
    plt.xlabel(filename+"("+"{}".format(category)+")")

plt.tight_layout()
plt.imshow()





























