import os
import csv
import argparse
import shutil


def get_driving_log(sample_dir, name='driving_log.csv'):
    filename = sample_dir + '/' + name
    print("Loading", filename)
    with open(filename) as file:
        reader = csv.reader(file)
        driving_log = [line for line in reader]
        print("Read {} lines".format(len(driving_log)))
        header = driving_log[0]
        return driving_log[1:len(driving_log)], header  # strip header


def save_driving_log(sample_dir, header, entries, name='driving_log.csv'):
    filename = sample_dir + '/' + name
    print("Saving log to file", filename)
    with open(filename, 'w') as file:
        print("Writing {} rows".format(len(entries) + 1))
        writer = csv.writer(file, lineterminator=os.linesep)
        writer.writerow(header)
        writer.writerows(entries)


def copy_image_files(input_dir, output_dir):
    for image in [img for img in os.listdir(input_dir) if img.endswith(".jpg")]:
        print("Copy", image, output_dir + '/' + image.split('/')[-1])
        shutil.copyfile(input_dir + '/' + image, output_dir + '/' + image.split('/')[-1])


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Clean driving log file')
    parser.add_argument(
        '--d1',
        '-d1',
        required=True,
        type=str,
        help='Training data directory 1')
    parser.add_argument(
        '--d2',
        '-d2',
        required=True,
        type=str,
        help='Training data directory 3')
    parser.add_argument(
        '--output_dir',
        '-o',
        required=True,
        type=str,
        help='Output dir')
    return parser.parse_args(), parser


def main():
    # Deal with the cmd line arguments
    args, parser = parse_cmd_line_args()

    args_valid = True
    if not os.path.isdir(args.d1) or not os.path.isdir(args.d2):
        print("Need to specify the sample directories")
        args_valid = False

    if not args_valid:
        parser.print_help()

    # Load/Clean/Save
    driving_log1, header = get_driving_log(args.d1)
    driving_log2, _ = get_driving_log(args.d2)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_driving_log(args.output_dir, header, driving_log1 + driving_log2)

    if not os.path.exists(args.output_dir + '/IMG'):
        os.mkdir(args.output_dir + '/IMG')

    copy_image_files(args.d1 + '/IMG', args.output_dir + '/IMG')
    copy_image_files(args.d2 + '/IMG', args.output_dir + '/IMG')


if __name__ == '__main__':
    main()
