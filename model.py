import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Cropping2D
import matplotlib

# Make sure the plots work on a headless machine
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse

# Constants
SAMPLE_MULTIPLIER = 4


def get_driving_log(sample_dir):
    with open(sample_dir + '/driving_log.csv') as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        header = driving_log[0]
        return driving_log[1:len(driving_log)], header  # strip header


def create_data_sets(driving_log):

    # Split off validation set
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(driving_log, test_size=0.4)
    validation_samples, test_samples = train_test_split(validation_samples, test_size=0.5)

    return train_samples, validation_samples, test_samples


def generator(samples, sample_dir, batch_size=32, steering_angle_correction=.25):
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
                center_image = cv2.imread(sample_dir + '/IMG/' + batch_sample[0].split('/')[-1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Add left and right camera images
                left_angle = center_angle + steering_angle_correction
                right_angle = center_angle - steering_angle_correction
                angles.extend([left_angle, right_angle])

                left_image = cv2.imread(sample_dir + '/IMG/' + batch_sample[1].split('/')[-1])
                right_image = cv2.imread(sample_dir + '/IMG/' + batch_sample[2].split('/')[-1])
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


def compile_model(model):
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    return model


def create_model():
    model = Sequential()
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # Crop image to show only the road
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(1))

    return compile_model(model)


def load_model_with_fallback(filename):
    try:
        return load_model(filename)
    except ValueError as err:
        print("Could not load model as is:", err)
        print("Loading without the optimizer state")
        import h5py
        f = h5py.File(filename, mode='r')

        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model = model_from_json(model_config.decode('utf-8'))
        model.load_weights(filename)
        return compile_model(model)


def train_model(model, train_generator, train_samples_per_epoch, validation_generator, validation_samples_per_epoch, epochs):
    # Train the model
    return model.fit_generator(train_generator, samples_per_epoch=train_samples_per_epoch, validation_data=validation_generator, nb_val_samples=validation_samples_per_epoch, nb_epoch=epochs)


def save_model(model, name):
    # Save
    model.save(name)


def plot_training_graphs(history):
    # Create some graphs

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


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Manipulate model')
    parser.add_argument(
        'action',
        type=str,
        choices=['train', 'fine_tune', 'test'],
        default='',
        help='main action'
    )
    parser.add_argument(
        '--sample_data_dir',
        '-d',
        type=str,
        default='./resources/example_data',
        help='Training data directory')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='model.h5',
        help='Model input file')
    parser.add_argument(
        '--model_out',
        '-o',
        type=str,
        default='model-out.h5',
        help='Model output file')
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=4,
        help='Number of epochs')
    parser.add_argument(
        '--steering_angle_correction',
        '-a',
        type=float,
        default=.25,
        help='Steering angle correction for left/right camera images')
    return parser.parse_args(), parser


def main():

    # Deal with the cmd line arguments
    args, parser = parse_cmd_line_args()

    args_valid = True
    if not os.path.isdir(args.sample_data_dir):
        print("Need to specify the sample dir")
        args_valid = False

    if not args_valid:
        parser.print_help()

    # Load sample data and split
    driving_log, header = get_driving_log(args.sample_data_dir)
    train_samples, validation_samples, test_samples = create_data_sets(driving_log)
    print("Original train: {}, validation: {}, test {}".format(len(train_samples), len(validation_samples), len(test_samples)))

    # Create generators for train and validation data
    train_generator = generator(train_samples, sample_dir=args.sample_data_dir, steering_angle_correction=args.steering_angle_correction)
    validation_generator = generator(validation_samples, sample_dir=args.sample_data_dir, steering_angle_correction=args.steering_angle_correction)

    # Account for data generation
    train_samples_per_epoch = len(train_samples) * SAMPLE_MULTIPLIER
    validation_samples_per_epoch = len(validation_samples) * SAMPLE_MULTIPLIER
    print("Augmented train: {}, validation: {}".format(train_samples_per_epoch, validation_samples_per_epoch))

    if args.action == 'train':
        # Train a new model
        print("Training model", args.model_out)
        model = create_model()
        history = train_model(model, train_generator, train_samples_per_epoch, validation_generator, validation_samples_per_epoch, epochs=args.epochs)
        save_model(model, args.model_out)
        plot_training_graphs(history)
        print(model.summary())

    if args.action =='fine_tune':
        # Fine tune an existing model
        print("Fine tuning ", args.model)
        # Load
        model = load_model_with_fallback(args.model)
        print("Input model:")
        print(model.summary())

        # Freeze the layers except the Fully connected layers
        for layer in model.layers:
            if not type(layer) == Dense:
                layer.trainable = False
            else:
                print("Keeping {} trainable".format(layer))

        # Train
        history = train_model(model, train_generator, train_samples_per_epoch, validation_generator, validation_samples_per_epoch, epochs=args.epochs)

        # Set all layers to trainable again
        for layer in model.layers:
            layer.trainable = True

        # Save
        save_model(model, args.model_out)
        plot_training_graphs(history)
        print("Output model:")
        print(model.summary())

    elif args.action == 'test':
        # Test a model
        print("Testing model", args.model)
        model = load_model_with_fallback(args.model)
        print(model.summary())

        test_generator = generator(validation_samples, sample_dir=args.sample_data_dir,
                                   steering_angle_correction=args.steering_angle_correction)
        test_samples_nb = len(test_samples) * SAMPLE_MULTIPLIER
        print("Testing on {} samples".format(test_samples_nb))
        result = model.evaluate_generator(test_generator, val_samples=test_samples_nb)
        print("Test result:", result)

if __name__ == '__main__':
    main()