import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

# Load sample data
SAMPLE_DIR = './resources/example_data'

def get_driving_log():
    with open(SAMPLE_DIR + '/driving_log.csv') as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        header = driving_log[0]
        return driving_log[1:len(driving_log)], header  # strip header


driving_log, header = get_driving_log()


# Split off validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)


# Create generators for train and validation data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = SAMPLE_DIR + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


# Define simple model
from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mse')

# Train the model
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

# Save
model.save('model.h5')