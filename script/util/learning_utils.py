from script.util import image_utils
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
import argparse


def points_augmentation(points, rotation_angle, horizontal_shift, vertical_shift, shape=(512, 512)):
    new_points = image_utils.upscale_points(points, shape)

    new_points = image_utils.translate_points(new_points, horizontal_shift=horizontal_shift,
                                              vertical_shift=vertical_shift)
    new_points = image_utils.rotate_points(new_points, rotation_angle, shape=shape)

    new_points = image_utils.downscale_points(new_points, shape)
    return new_points


def data_augmentation(X, y_skeleton, y_clf, rotation_angle, horizontal_shift, vertical_shift, iteration=10):
    new_X = list()
    new_y_skeleton = list()
    new_y_clf = list()
    num_data = X.shape[0]
    data_shape = (X.shape[1], X.shape[2])

    # Add current data
    for idx in range(num_data):
        new_X.append(X[idx].copy())
        new_y_skeleton.append(y_skeleton[idx])
        new_y_clf.append(y_clf[idx])

    for i in range(iteration):
        for idx in range(num_data):
            image = X[idx].copy()
            class_id = y_clf[idx]
            points = y_skeleton[idx]

            # process image of X
            r = np.random.randint(rotation_angle) + 1
            h = np.random.randint(horizontal_shift) + 1
            v = np.random.randint(vertical_shift) + 1

            new_image = image_utils.translate_image(image, h, v)
            new_image = image_utils.rotate_image(new_image, r)
            new_X.append(new_image)

            # process points of y_skeleton
            new_points = points_augmentation(points, -r, h, v, data_shape)
            new_y_skeleton.append(new_points)

            # process class of y_clf
            new_y_clf.append(class_id)

    new_X = np.asarray(new_X)
    new_y_skeleton = np.asarray(new_y_skeleton)
    new_y_clf = np.asarray(new_y_clf)
    return new_X, new_y_skeleton, new_y_clf


def data_split_and_generation(X, y_skeleton, y_clf, out_file, train_ratio=0.8, val_ratio=0.1,
                              rotation_angle=90,
                              horizontal_shift=20, vertical_shift=20,
                              iteration=10, white_object=True, image_type='gray', shuffle=True):
    """
    :param X: training images
    :param y_skeleton: label - points along the skeleton
    :param y_clf: label - chromosome class
    :param out_file: file to save augmented and splitted data
    :param train_ratio: percentage of data for training
    :param val_ratio: percentage of data for validation
    :param rotation_angle:
    :param horizontal_shift:
    :param vertical_shift:
    :param iteration: #new_data = #data * iteration
    :param white_object: whether the chromosome is white/black
    :param image_type: string representing the type of image: "gray", "rgb"
    :param shuffle: whether data is shuffled before splitting
    :return: augmented and splitted data
    """
    if train_ratio + val_ratio >= 1:
        raise Exception("Total ratio is greater than 1")
    num_data = X.shape[0]
    ratio_train = train_ratio
    ratio_train_val = train_ratio + val_ratio

    if shuffle:
        shuffled_ids = np.random.permutation(num_data)
        X = X[shuffled_ids]
        y_skeleton = y_skeleton[shuffled_ids]
        y_clf = y_clf[shuffled_ids]

    if white_object:
        X_train, X_val, X_test = np.split(X.copy(), [int(ratio_train * num_data), int(ratio_train_val * num_data)])
    else:
        X_train, X_val, X_test = np.split(255 - X.copy(),
                                          [int(ratio_train * num_data), int(ratio_train_val * num_data)])

    y_skeleton_train, y_skeleton_val, y_skeleton_test = np.split(y_skeleton.copy(), [int(ratio_train * num_data),
                                                                                     int(ratio_train_val * num_data)])
    y_clf_train, y_clf_val, y_clf_test = np.split(y_clf.copy(),
                                                  [int(ratio_train * num_data), int(ratio_train_val * num_data)])

    if image_type == 'gray':
        new_X_train, new_y_skeleton_train, new_y_clf_train = data_augmentation(X_train, y_skeleton_train,
                                                                               y_clf_train, rotation_angle,
                                                                               horizontal_shift, vertical_shift,
                                                                               iteration=iteration)
        new_X_val, new_y_skeleton_val, new_y_clf_val = data_augmentation(X_val, y_skeleton_val, y_clf_val,
                                                                         rotation_angle, horizontal_shift,
                                                                         vertical_shift, iteration=iteration)
        new_X_test, new_y_skeleton_test, new_y_clf_test = data_augmentation(X_test, y_skeleton_test,
                                                                            y_clf_test, rotation_angle,
                                                                            horizontal_shift, vertical_shift,
                                                                            iteration=iteration)
    else:
        new_X_train, new_y_skeleton_train, new_y_clf_train = data_augmentation(X_train, y_skeleton_train, y_clf_train,
                                                                               rotation_angle, horizontal_shift,
                                                                               vertical_shift, iteration=iteration)
        new_X_val, new_y_skeleton_val, new_y_clf_val = data_augmentation(X_val, y_skeleton_val, y_clf_val,
                                                                         rotation_angle, horizontal_shift,
                                                                         vertical_shift, iteration=iteration)
        new_X_test, new_y_skeleton_test, new_y_clf_test = data_augmentation(X_test, y_skeleton_test, y_clf_test,
                                                                            rotation_angle, horizontal_shift,
                                                                            vertical_shift, iteration=iteration)

    print(new_X_train.shape)
    print(new_y_skeleton_train.shape)
    print(new_y_clf_train.shape)
    print(new_X_val.shape)
    print(new_y_skeleton_val.shape)
    print(new_y_clf_val.shape)
    print(new_X_test.shape)
    print(new_y_skeleton_test.shape)
    print(new_y_clf_test.shape)

    # dump vectors to file
    # data = {'X_train': new_X_train, 'y_skeleton_train': new_y_skeleton_train, 'y_clf_train': new_y_clf_train,
    #         'X_val': new_X_val, 'y_skeleton_val': new_y_skeleton_val, 'y_clf_val': new_y_clf_val,
    #         'X_test': new_X_test, 'y_skeleton_test': new_y_skeleton_test, 'y_clf_test': new_y_clf_test}

    train_data = {"X": new_X_train, "y_skel": new_y_skeleton_train, "y_clf": new_y_clf_train}
    val_data = {"X": new_X_val, "y_skel": new_y_skeleton_val, "y_clf": new_y_clf_val}
    test_data = {"X": new_X_test, "y_skel": new_y_skeleton_test, "y_clf": new_y_clf_test}

    data = {"train": train_data, "val": val_data, "test": test_data}
    pickle.dump(data, open(out_file, 'wb'))
    print("Done saving data to " + out_file)

    return data


# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #     print(cm)

    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def show_confusion_matrix(y_test, y_pred, class_names, title='Confusion matrix'):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    plt.show()

# def custom_parse_args(args, annotations_file, class_mapping_file, main_dir=""):
#     """ Parse the arguments.
#     """
#     parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
#     subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
#     subparsers.required = True
#
#     coco_parser = subparsers.add_parser('coco')
#     coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')
#
#     pascal_parser = subparsers.add_parser('pascal')
#     pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
#
#     kitti_parser = subparsers.add_parser('kitti')
#     kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
#
#     def csv_list(string):
#         return string.split(',')
#
#     oid_parser = subparsers.add_parser('oid')
#     oid_parser.add_argument('main_dir', help='Path to dataset directory.', default=main_dir)
#     oid_parser.add_argument('--version', help='The current dataset version is v4.', default='v4')
#     oid_parser.add_argument('--labels-filter', help='A list of labels to filter.', type=csv_list, default=None)
#     oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
#     oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)
#
#     csv_parser = subparsers.add_parser('csv')
#     csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.',
#                             default=annotations_file)
#     csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.',
#                             default=class_mapping_file)
#     csv_parser.add_argument('--val-annotations',
#                             help='Path to CSV file containing annotations for validation (optional).')
#
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument('--snapshot', help='Resume training from a snapshot.')
#     group.add_argument('--imagenet-weights',
#                        help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
#                        action='store_const', const=True, default=True)
#     group.add_argument('--weights', help='Initialize the model with weights from a file.')
#     group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights',
#                        action='store_const', const=False)
#
#     parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
#     parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
#     parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
#     parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
#     parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.',
#                         action='store_true')
#     parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
#     parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
#     parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-5)
#     parser.add_argument('--snapshot-path',
#                         help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
#                         default='./snapshots')
#     parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
#     parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
#     parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
#                         action='store_false')
#     parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
#     parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
#     parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
#                         default=800)
#     parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
#                         type=int, default=1333)
#     parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
#     parser.add_argument('--weighted-average',
#                         help='Compute the mAP using the weighted average of precisions among classes.',
#                         action='store_true')
#
#     # Fit generator arguments
#     parser.add_argument('--workers',
#                         help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0',
#                         type=int, default=1)
#     parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int,
#                         default=10)
#
#     return train.check_args(parser.parse_args(args))
