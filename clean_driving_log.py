import os
import csv
import argparse

# Constants
SAMPLE_MULTIPLIER = 4


def get_driving_log(sample_dir, name='driving_log.csv'):
    filename = sample_dir + '/' + name
    print("Loading", filename)
    with open(filename) as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        print("Read {} lines".format(len(driving_log)))
        header = driving_log[0]
        return driving_log[1:len(driving_log)], header  # strip header


def clean_driving_log(driving_log, sample_dir):
    cleansed_driving_log = []
    for entry in driving_log:
        images = [sample_dir + '/IMG/' + item.split('/')[-1] for item in entry[0:3]]
        if all([os.path.exists(img) for img in images]):
            cleansed_driving_log.append(entry)
        else:
            # Cleanup
            for img in [img for img in images if os.path.exists(img)]:
                print("deleting", img)
                os.remove(img)

    return cleansed_driving_log


def save_driving_log(sample_dir, header, entries, name='driving_log.csv'):
    filename = sample_dir + '/' + name
    print("Saving log to file", filename)
    with open(filename, 'w') as file:
        print("Writing {} rows".format(len(entries) + 1))
        writer = csv.writer(file, lineterminator=os.linesep)
        writer.writerow(header)
        writer.writerows(entries)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Clean driving log file')
    parser.add_argument(
        '--sample_data_dir',
        '-d',
        required=True,
        type=str,
        help='Training data directory')
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

    # Load/Clean/Save
    driving_log, header = get_driving_log(args.sample_data_dir)
    driving_log = clean_driving_log(driving_log, args.sample_data_dir)
    save_driving_log(args.sample_data_dir, header, driving_log)


if __name__ == '__main__':
    main()
