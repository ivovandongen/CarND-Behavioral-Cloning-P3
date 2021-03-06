import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Cropping2D
import matplotlib
from keras.utils.visualize_util import plot

# Make sure the plots work on a headless machine
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse

# Constants
SAMPLE_MULTIPLIER = 4
RANDOM_STATE = 42

## Image manipulation


def grayscale_image(image):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(image)


## Data set construction


def get_driving_log(sample_dir):
    with open(sample_dir + '/driving_log.csv') as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        header = driving_log[0]
        lines = driving_log[1:len(driving_log)]

        # Correct paths
        for line in lines:
            for i in range(3):
                line[i] = sample_dir + '/IMG/' + line[i].split('/')[-1]

        return lines, header

def create_data_sets(driving_log):

    # Split off validation set
    from sklearn.model_selection import train_test_split
    log_shuffled = shuffle(driving_log, random_state=RANDOM_STATE)
    train_samples, validation_samples = train_test_split(log_shuffled, test_size=0.4, random_state=RANDOM_STATE)
    validation_samples, test_samples = train_test_split(validation_samples, test_size=0.5, random_state=RANDOM_STATE)

    return train_samples, validation_samples, test_samples


def generator(samples, batch_size=32, steering_angle_correction=.25):
    num_samples = len(samples)
    batch_size = batch_size // SAMPLE_MULTIPLIER
    total = 0
    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples, random_state=RANDOM_STATE)
        for offset in range(0, num_samples, batch_size):
            # print(offset, offset + batch_size)
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Add center image
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Add left and right camera images
                left_angle = center_angle + steering_angle_correction
                right_angle = center_angle - steering_angle_correction
                angles.extend([left_angle, right_angle])

                left_image = cv2.imread(batch_sample[1])
                right_image = cv2.imread(batch_sample[2])
                images.extend([left_image, right_image])

                # Add flipped version
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

            X = np.array(images)
            y = np.array(angles)
            total += len(images)
            yield shuffle(X, y, random_state=RANDOM_STATE)


## Model creation


def compile_model(model):
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    return model


def create_model():
    model = Sequential()
    # Grayscale
    model.add(Lambda(grayscale_image, input_shape=(160, 320, 3)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
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
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(1))

    return compile_model(model)


def load_partial_model(filename, first_layer=None, last_layer=None):
    print("Loading partial model:", filename)
    import h5py
    f = h5py.File(filename, mode='r')

    # instantiate model
    model_config = f.attrs.get('model_config')
    if model_config is None:
        raise ValueError('No model found in config file.')
    model = model_from_json(model_config.decode('utf-8'))
    model.load_weights(filename)

    new_model = Sequential()

    start_adding = first_layer is None
    for layer in model.layers:
        if first_layer is not None and layer.name == first_layer:
            start_adding = True

        if start_adding:
            print ("Adding", layer.name)
            new_model.add(layer)
        else:
            print("Skipping", layer.name)

        if last_layer is not None and layer.name == last_layer:
            print ("Done")
            break

    return compile_model(new_model)

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


## Visualization


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


def plot_model_output(model, input, cmap='gray', nb_filters=5, image_prefix="activation"):
    prediction = model.predict(input)
    prediction = np.squeeze(prediction, axis=0)

    columns = min(nb_filters, prediction.shape[2])
    rows = 1
    fig = plt.figure(figsize=(columns * 3, rows * 3))
    plt.axis('off')

    for i in range(0, columns):
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.title('FeatureMap ' + str(i))
        p = prediction[:, :, i]
        plt.axis('off')
        ax.imshow(p, interpolation='None', cmap=cmap)
    fig.tight_layout()

    plt.savefig('./{}.png'.format(image_prefix), format='png')


## Main


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Manipulate model')
    parser.add_argument(
        'action',
        type=str,
        choices=['train', 'fine_tune', 'test', 'plot', 'visualize'],
        default='',
        help='main action'
    )
    parser.add_argument(
        '--sample_data_dir',
        '-d',
        nargs = '+',
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
    parser.add_argument(
        '--input_image',
        '-i',
        nargs ='+',
        type=str,
        help='Input image for activation map visualization')
    parser.add_argument(
        '--conv_layers',
        '-c',
        type=int,
        default=1,
        help='Number of layers to show for activation map visualization')
    return parser.parse_args(), parser


def main():

    # Deal with the cmd line arguments
    args, parser = parse_cmd_line_args()

    ## Visualization targets

    if args.action == 'plot':
        print("Plotting model", args.model)
        model = load_model_with_fallback(args.model)
        print(model.summary())
        plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)
        return

    elif args.action == 'visualize':
        print("Visualizing model activations", args.model)
        images = args.input_image if type(args.input_image) == list else [args.input_image]
        for image in images:
            for i in range(1, args.conv_layers + 1):
                model = load_partial_model(args.model, last_layer='convolution2d_{}'.format(i))
                plot_model_output(model, np.array([cv2.imread(image)]), image_prefix="examples/{}-{}".format(image.split('/')[-1], model.layers[-1].name), cmap='jet')
        return

    ## Train / test targets

    sample_data_dirs = args.sample_data_dir if type(args.sample_data_dir) == list else [args.sample_data_dir]

    args_valid = True
    if not all([os.path.isdir(dir) for dir in sample_data_dirs]):
        print("Need to specify the sample dirs correctly")
        args_valid = False

    if not args_valid:
        parser.print_help()

    print("Reading resources from:")
    for dir in sample_data_dirs:
        print(dir)

    # Load sample data and split
    driving_log = sum([log[0] for log in [get_driving_log(dir) for dir in sample_data_dirs]], [])
    print("Total samples:", len(driving_log), "Including left/right cameras", len(driving_log) * 3)

    train_samples, validation_samples, test_samples = create_data_sets(driving_log)
    print("Original train: {}, validation: {}, test {}".format(len(train_samples), len(validation_samples), len(test_samples)))

    # Create generators for train and validation data
    train_generator = generator(train_samples, steering_angle_correction=args.steering_angle_correction)
    validation_generator = generator(validation_samples, steering_angle_correction=args.steering_angle_correction)

    # Account for data generation
    train_samples_per_epoch = len(train_samples) * SAMPLE_MULTIPLIER
    validation_samples_per_epoch = len(validation_samples) * SAMPLE_MULTIPLIER
    print("Augmented train: {}, validation: {}".format(train_samples_per_epoch, validation_samples_per_epoch), "Total:", len(driving_log) * SAMPLE_MULTIPLIER)

    if args.action == 'train':
        # Train a new model
        print("Training model", args.model_out)
        model = create_model()
        history = train_model(model, train_generator, train_samples_per_epoch, validation_generator, validation_samples_per_epoch, epochs=args.epochs)
        save_model(model, args.model_out)
        plot_training_graphs(history)
        print(model.summary())

    elif args.action =='fine_tune':
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

        test_generator = generator(validation_samples, steering_angle_correction=args.steering_angle_correction)
        test_samples_nb = len(test_samples) * SAMPLE_MULTIPLIER
        print("Testing on {} samples".format(test_samples_nb))
        result = model.evaluate_generator(test_generator, val_samples=test_samples_nb)
        print("Test result:", result)


if __name__ == '__main__':
    main()

