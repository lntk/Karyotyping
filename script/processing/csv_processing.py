"""
    @author: lntk
"""

from script.util import image_utils, general_utils, data_utils


def image_and_bounding_boxes_to_csv(image_files, bounding_boxes, labels, annotations_file):
    annotations = list()
    num_file = len(image_files)
    for idx in range(num_file):
        image_file = image_files[idx]
        box = bounding_boxes[idx]
        label = str(labels[idx])
        annotations.append([image_file, *box, label])

    general_utils.write_list_to_csv(annotations, annotations_file)


def annotated_file_to_bounding_box_csv(file_in, csv_dir, image_dir, prefix=""):
    image_files, _, labels = data_utils.read_annotated_file(file_in, image_dir)
    image_files_to_bounding_box_csv(image_files, labels, csv_dir, prefix=prefix)


def directory_to_bounding_box_csv(directory, csv_dir, prefix="", verbose=False):
    image_files, labels, _, _ = data_utils.directory_to_images_files_and_labels(directory, verbose=verbose)
    image_files_to_bounding_box_csv(image_files, labels, csv_dir, prefix=prefix, verbose=verbose)


def image_files_to_bounding_box_csv(image_files, labels, csv_dir, prefix="", verbose=False):
    general_utils.create_directory(csv_dir)

    boxes = list()
    for image_file in image_files:
        image = image_utils.read_image(image_file, cmap="rgb")
        box = image_utils.get_chromosome_bounding_box(image)
        boxes.append(box)

    train_data, val_data, test_data = data_utils.split_multiple_data_lists([image_files, boxes, labels])

    annotations_train_file = csv_dir + "/" + prefix + "annotations_train.csv"
    image_and_bounding_boxes_to_csv(*train_data, annotations_train_file)

    annotations_val_file = csv_dir + "/" + prefix + "annotations_val.csv"
    image_and_bounding_boxes_to_csv(*val_data, annotations_val_file)

    annotations_test_file = csv_dir + "/" + prefix + "annotations_test.csv"
    image_and_bounding_boxes_to_csv(*test_data, annotations_test_file)

    class_mapping = [[str(i), i] for i in range(len(labels))]
    class_mapping_file = csv_dir + "/" + prefix + "class_mapping.csv"
    general_utils.write_list_to_csv(class_mapping, class_mapping_file)

    if verbose:
        print("Train: ", annotations_train_file)
        print("Validation: ", annotations_val_file)
        print("Test: ", annotations_test_file)
        print("Class mapping:", class_mapping_file)
