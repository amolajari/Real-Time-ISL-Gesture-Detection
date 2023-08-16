from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import  Adam

sz = 250

# CNN Arch.
classifier = Sequential()

# 1st convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# 2nd convolution layer and pooling
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())
# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=96, activation='relu'))
# classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))

# Compiling the CNN

learning_rate = 0.0001
adam = Adam(lr=learning_rate)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data and training the model
classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(sz, sz),
                                                 batch_size=3,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(sz, sz),
                                            batch_size=3,
                                            color_mode='grayscale',
                                            class_mode='categorical')

batch_size = 3

history = classifier.fit(training_set,
        steps_per_epoch=int(300/batch_size),
        epochs=5,
        validation_data=test_set,
        validation_steps=int(150/batch_size))

# Saving the model
classifier.save('model-bw.h5')
print('Saved')