import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

EPOCHS = 4
SAMPLE_DIR = './resources/example_data'
SAMPLE_MULTIPLIER = 4
STEERING_ANGLE_CORRECTION = 0.25

# Load sample data
def get_driving_log():
    with open(SAMPLE_DIR + '/driving_log.csv') as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        header = driving_log[0]
        return driving_log[1:len(driving_log) - 1], header  # strip header


driving_log, header = get_driving_log()

# Split off validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_log, test_size=0.4)
validation_samples, test_samples = train_test_split(validation_samples, test_size=0.5)

print("Original train: {}, validation: {}, test {}".format(len(train_samples), len(validation_samples), len(test_samples)))

# Create generators for train and validation data


def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size // SAMPLE_MULTIPLIER
    total = 0
    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # print(offset, offset + batch_size)
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Add center image
                center_image = cv2.imread(SAMPLE_DIR + '/IMG/' + batch_sample[0].split('/')[-1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Add left and right camera images
                left_angle = center_angle + STEERING_ANGLE_CORRECTION
                right_angle = center_angle - STEERING_ANGLE_CORRECTION
                angles.extend([left_angle, right_angle])

                left_image = cv2.imread(SAMPLE_DIR + '/IMG/' + batch_sample[1].split('/')[-1])
                right_image = cv2.imread(SAMPLE_DIR + '/IMG/' + batch_sample[2].split('/')[-1])
                images.extend([left_image, right_image])

                # Add flipped version
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

            X = np.array(images)
            y = np.array(angles)
            total += len(images)
            yield shuffle(X, y)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
test_generator = generator(validation_samples)

# Account for data generation
train_samples_per_epoch = len(train_samples) * SAMPLE_MULTIPLIER
validation_samples_per_epoch = len(validation_samples) * SAMPLE_MULTIPLIER
test_samples_per_epoch = len(test_samples) * SAMPLE_MULTIPLIER

print("Augmented train: {}, validation: {}, test {}".format(train_samples_per_epoch, validation_samples_per_epoch, test_samples_per_epoch))

# Define simple model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# Train the model
history = model.fit_generator(train_generator, samples_per_epoch=train_samples_per_epoch, validation_data=validation_generator, nb_val_samples=validation_samples_per_epoch, nb_epoch=EPOCHS)

# Save
model.save('model.h5')

# Create some graphs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./accuracy.png', format='png')

# summarize history for loss
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./loss.png', format='png')
