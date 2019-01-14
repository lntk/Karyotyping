import os
import pickle
import numpy as np
import cv2
import functools
from random import shuffle

from script.util import learning_utils, general_utils, image_utils


def process_annotated_data(file_in, file_out, image_dir, file_hard=None, verbose=True):
    """
    :param file_in: An annotated file, which contains a dictionary of image files, points, classes  
    :param file_out: A file to store output
    :param file_hard: A file containing image files that are considered difficult
    :param image_dir: The current directory contains above image files
    :param verbose: A boolean value indicates whether info is displayed
    :return: 
    """

    if os.path.isfile(file_in):
        annotated_data = pickle.load(open(file_in, 'rb'))

        if verbose:
            print("Dictionary keys: ", annotated_data.keys())

        image_files = annotated_data['fns']
        skeleton = annotated_data['skeleton']
        clf = annotated_data['class']
    else:
        raise Exception("file not found")

    if verbose:
        print("Show 10 random data:")
        for i in range(10):
            idx = np.random.randint(len(image_files))
            print(image_files[idx], skeleton[idx], clf[idx])

    # Convert annotated data to numpy arrays
    if file_hard is not None:
        hard_image_files = general_utils.read_lines(file_hard)
    else:
        hard_image_files = None

    X = list()
    y_skeleton = list()
    y_clf = list()

    X_hard = list()
    y_skeleton_hard = list()
    y_clf_hard = list()

    for idx in range(len(image_files)):
        image_file = image_files[idx]
        image_file = image_file[7:]
        image = cv2.imread(image_dir + "/" + image_file, 0)
        i_skeleton = np.asarray(skeleton[idx])
        i_clf = int(np.asarray(clf[idx]))

        if file_hard is not None and image_file in hard_image_files:
            X_hard.append(image)
            y_skeleton_hard.append(i_skeleton)
            y_clf_hard.append(i_clf)
        else:
            X.append(image)
            y_skeleton.append(i_skeleton)
            y_clf.append(i_clf)

    X = np.asarray(X)
    y_skeleton = np.asarray(y_skeleton)
    y_clf = np.asarray(y_clf)

    if file_hard is not None:
        X_hard = np.asarray(X_hard)
        y_skeleton_hard = np.asarray(y_skeleton_hard)
        y_clf_hard = np.asarray(y_clf_hard)

    if verbose:
        print("Images shape: ", X.shape)
        if file_hard is not None:
            print("Hard images shape: ", X_hard.shape)
        print("Skeletons shape: ", y_skeleton.shape)
        print("Classes shape: ", y_clf.shape)

    # Dump numpy arrays to file
    if file_hard is not None:
        data = {'X': X, 'y_skeleton': y_skeleton, 'y_clf': y_clf, 'X_hard': X_hard, 'y_skeleton_hard': y_skeleton_hard,
                'y_clf_hard': y_clf_hard}
    else:
        data = {'X': X, 'y_skeleton': y_skeleton, 'y_clf': y_clf}
    pickle.dump(data, open(file_out, 'wb'))

    if file_hard is not None:
        return X, y_skeleton, y_clf, X_hard, y_skeleton_hard, y_clf_hard
    else:
        return X, y_skeleton, y_clf


def annotated_data_to_image_ids_and_points(file_in, file_out, image_dir, verbose=False, debug=False):
    if os.path.isfile(file_in):
        if verbose:
            print("Opening file: ", file_in)
        annotated_data = pickle.load(open(file_in, 'rb'))

        if verbose:
            print("Dictionary keys: ", annotated_data.keys())

        image_files = annotated_data['fns']
        skeleton = annotated_data['skeleton']
        clf = annotated_data['class']
    else:
        raise Exception("file not found")

    if verbose:
        print("Show 10 random data:")
        for i in range(10):
            idx = np.random.randint(len(image_files))
            print(image_files[idx], skeleton[idx], clf[idx])

    image_ids = list()
    points_list = list()

    for idx in range(len(image_files)):
        image_file = image_files[idx]
        image_file = image_file[7:]
        points = np.asarray(skeleton[idx])

        image_ids.append(image_dir + "/" + image_file)
        points_list.append(points)

    if debug:
        view_sample_data_list(image_ids, points_list, num_sample=5)

    data = {'image_files': image_ids, 'points_list': points_list}
    if file_out is not None:
        pickle.dump(data, open(file_out, 'wb'))
        if verbose:
            print("Data is dumped to file: ", file_out)
    else:
        print("Output file is not specified, data was not dumped.")

    return image_ids, points_list


def read_annotated_file(annotated_file, image_dir, file_out=None, verbose=False, debug=False):
    if os.path.isfile(annotated_file):
        if verbose:
            print("Opening file: ", annotated_file)
        annotated_data = pickle.load(open(annotated_file, 'rb'))

        if verbose:
            print("Dictionary keys: ", annotated_data.keys())

        image_files = annotated_data['fns']
        skeleton = annotated_data['skeleton']
        clf = annotated_data['class']
    else:
        raise Exception("file not found")

    if verbose:
        print("Show 10 random data:")
        for i in range(10):
            idx = np.random.randint(len(image_files))
            print(image_files[idx], skeleton[idx], clf[idx])

    image_ids = list()
    points_list = list()
    labels = list()

    for idx in range(len(image_files)):
        image_file = image_files[idx]
        image_file = image_file[7:]
        points = np.asarray(skeleton[idx])
        label = int(clf[idx]) - 1  # TODO: Tell Tuan to reduce this value by 1

        image_ids.append(image_dir + "/" + image_file)
        points_list.append(points)
        labels.append(label)

    if debug:
        view_sample_data_list(image_ids, points_list, num_sample=5)

    data = {'image_files': image_ids, 'points_list': points_list, "labels": labels}
    if file_out is not None:
        pickle.dump(data, open(file_out, 'wb'))
        if verbose:
            print("Data is dumped to file: ", file_out)
    else:
        print("Output file is not specified, annotated data was not dumped.")

    return image_ids, points_list, labels


def split_multiple_data_lists(lists, ratio=None, debug=False):
    if ratio is None:
        ratio = [0.8, 0.1, 0.1]

    if len(lists) == 0:
        raise Exception("List is empty")

    num_data = len(lists[0])
    for data_list in lists:
        if len(data_list) != num_data:
            raise Exception("Capacity is not matched.")

    random_ids = list(range(num_data))
    shuffle(random_ids)

    # TODO: Find a way to not duplicate list
    shuffled_lists = list()
    for data_list in lists:
        data_list = [data_list[idx] for idx in random_ids]
        shuffled_lists.append(data_list)

    train_ratio, val_ratio, _ = ratio

    train_data = list()
    val_data = list()
    test_data = list()

    for data_list in shuffled_lists:
        train_data.append(data_list[0:int(train_ratio * num_data)])
        val_data.append(data_list[int(train_ratio * num_data): int((train_ratio + val_ratio) * num_data)])
        test_data.append(data_list[int((train_ratio + val_ratio) * num_data):])

    if debug:
        print("Train shape: ", len(train_data[0]))
        print("Val shape: ", len(val_data[0]))
        print("Test shape: ", len(test_data[0]))

    return train_data, val_data, test_data


def split_data_list(X, y, ratio=None, debug=False):
    if ratio is None:
        ratio = [0.8, 0.1, 0.1]
    if len(X) != len(y):
        raise Exception("Capacity is not matched.")
    num_data = len(X)
    random_ids = list(range(num_data))
    shuffle(random_ids)
    X = [X[idx] for idx in random_ids]
    y = [y[idx] for idx in random_ids]

    if debug:
        print("Show some data after shuffled:")
        view_sample_data_list(X, y, num_sample=5)

    train_ratio, val_ratio, _ = ratio

    X_train = X[0:int(train_ratio * num_data)]
    y_train = y[0:int(train_ratio * num_data)]

    X_val = X[int(train_ratio * num_data): int((train_ratio + val_ratio) * num_data)]
    y_val = y[int(train_ratio * num_data): int((train_ratio + val_ratio) * num_data)]

    X_test = X[int((train_ratio + val_ratio) * num_data):]
    y_test = y[int((train_ratio + val_ratio) * num_data):]

    train_data = {"X": X_train, "y": y_train}
    val_data = {"X": X_val, "y": y_val}
    test_data = {"X": X_test, "y": y_test}

    if debug:
        print("Train shape: ", len(X_train), len(y_train))
        print("Val shape: ", len(X_val), len(y_val))
        print("Test shape: ", len(X_test), len(y_test))

        print("Show some data after splitted:")
        view_sample_data_list(X_train, y_train)

    data = {"train": train_data, "val": val_data, "test": test_data}
    return data


def generate_data(file_in, file_out, debug=False):
    """
    :param file_in:
    :param file_out:
    :param debug:
    :return:
    """

    with open(file_in, 'rb') as f:
        dictionary = pickle.load(f)

    X = dictionary['X']
    y_skeleton = dictionary['y_skeleton']
    y_clf = np.subtract(dictionary['y_clf'], np.asarray([1]))

    have_hard_images = 'X_hard' in dictionary
    X_hard = None
    y_skeleton_hard = None
    y_clf_hard = None

    if have_hard_images:
        X_hard = dictionary['X_hard']
        y_skeleton_hard = dictionary['y_skeleton_hard']
        y_clf_hard = dictionary['y_clf_hard']

    if debug:
        print("Images shape: ", X.shape)
        if have_hard_images:
            print("Hard images shape: ", X_hard.shape)
        print("Skeletons shape: ", y_skeleton.shape)
        print("Classes shape: ", y_clf.shape)

    if have_hard_images:
        new_X_hard, new_y_skeleton_hard, new_y_clf_hard = learning_utils.data_augmentation(X_hard,
                                                                                           y_skeleton_hard,
                                                                                           y_clf_hard,
                                                                                           rotation_angle=60,
                                                                                           horizontal_shift=10,
                                                                                           vertical_shift=10,
                                                                                           iteration=2)

        X = np.concatenate((X, new_X_hard))
        y_skeleton = np.concatenate((y_skeleton, new_y_skeleton_hard))
        y_clf = np.concatenate((y_clf, new_y_clf_hard))

    if debug:
        print("Final images shape: ", X.shape)
        print("Final skeletons shape: ", y_skeleton.shape)
        print("Final classes shape: ", y_clf.shape)

    train_data, val_data, test_data = learning_utils.data_split_and_generation(X, y_skeleton, y_clf, file_out,
                                                                               white_object=False)
    return train_data, val_data, test_data


def generate_image(image, points=None, rotation_angle=60, horizontal_shift=10, vertical_shift=10):
    r = np.random.randint(rotation_angle) + 1
    h = np.random.randint(horizontal_shift) + 1
    v = np.random.randint(vertical_shift) + 1
    new_image = image_utils.translate_image(image, h, v)
    new_image = image_utils.rotate_image(new_image, r)

    if points is None:
        return new_image
    else:
        new_points = learning_utils.points_augmentation(points, -r, h, v, image.shape)
        return new_image, new_points


def process_karyotyping_images(directory, chromosome_type="xx", debug=False, load_karyotype_info=False, max_p=None,
                               need_confirm=False):
    last_chromosome = chromosome_type[1]
    chromosome_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "x", "y"]
    save_directory = directory + "/" + chromosome_type + "/chromosome"
    karyotype_directory = directory + "/" + chromosome_type

    if debug:
        print("Save directory: " + save_directory)
        print("Karyotype directory: " + karyotype_directory)

    general_utils.create_directory(save_directory)

    if not load_karyotype_info:
        # Create directories to store output images
        for idx in chromosome_ids:
            general_utils.create_directory(save_directory + "/" + str(idx))

        karyotypes = dict()
        image_files = general_utils.get_all_files(karyotype_directory)

        for image_file in image_files:
            if not image_file.endswith(".bmp"):
                continue
            if debug:
                print(image_file)

            image = cv2.imread(karyotype_directory + "/" + image_file, 0)
            # ret, thresh = threshold.partial_otsu_threshold(image, minval=0, maxval=255, dark_background=False)
            ret, thresh = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY_INV)
            # if debug:
            #     image_utils.show_image(thresh)

            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            num_contour = len(contours)

            # Find area threshold - taking the 46-th largest area
            areas = list()
            for idx in range(num_contour):
                is_outer_contour = hierarchy[0][idx][3] == -1
                if is_outer_contour:
                    area = cv2.contourArea(contours[idx])
                    areas.append(area)
            areas.sort(reverse=True)
            if len(areas) < 46:
                print("Wrong at " + image_file)
                continue
            area_threshold = areas[45] - 0.01

            chosen_contours = list()
            for idx in range(num_contour):
                contour = contours[idx]
                if cv2.contourArea(contour) < area_threshold:
                    continue
                chosen_contours.append(contour)

            if len(chosen_contours) < 46:
                print("Wrong at " + image_file)
                continue

            contour_info = list()
            for contour in chosen_contours:
                x, y, w, h = cv2.boundingRect(contour)
                contour_info.append([x, y, w, h, contour])

            def sorted_by(a, b):
                x_a, y_a, w_a, h_a, _ = a
                x_b, y_b, w_b, h_b, _ = b
                if (y_a + h_a) < y_b:
                    return -1
                if x_a < x_b:
                    return -1
                return 1

            cmp = functools.cmp_to_key(sorted_by)
            contour_info.sort(key=cmp)

            if need_confirm:
                rgb_image = np.stack((image,) * 3, axis=-1)
                image_utils.show_image(image_utils.get_image_with_contours(rgb_image, chosen_contours, thickness=-1),
                                       cmap=None)
                user_input = input()
                if "yes".startswith(user_input):
                    karyotypes[image_file] = contour_info
                else:
                    print("Skipping: " + image_file)
            else:
                rgb_image = np.stack((image,) * 3, axis=-1)
                image_utils.show_image(image_utils.get_image_with_contours(rgb_image, chosen_contours, thickness=-1),
                                       cmap=None)
                karyotypes[image_file] = contour_info

        # Save data in case of bugs
        pickle.dump(karyotypes, open(directory + "/" + chromosome_type + "_karyotype_info.data", 'wb'))
    else:
        with open(directory + "/" + chromosome_type + "_karyotype_info.data", 'rb') as f:
            karyotypes = pickle.load(f)

    # Find maximum perimeter
    if max_p is None:
        max_p = -1
        for idx in karyotypes.keys():
            contour_info = karyotypes[idx]

            if len(contour_info) != 46:
                print("Skipping:" + idx)
                continue

            con_1 = contour_info[0][4]
            con_2 = contour_info[1][4]
            p_1 = cv2.arcLength(con_1, True)
            p_2 = cv2.arcLength(con_2, True)
            max_p = max(p_1, max_p)
            max_p = max(p_2, max_p)

    if debug:
        print("Max p: " + str(max_p))

    # Resize all images according to max perimeter
    for image_file in karyotypes.keys():
        image = cv2.imread(karyotype_directory + "/" + image_file, 0)
        contour_info = karyotypes[image_file]
        if debug:
            print(image_file)

        if len(contour_info) != 46:
            print("Skipping:" + image_file)
            continue

        # Get max perimeter of two chromosome 1
        con_1 = contour_info[0][4]
        con_2 = contour_info[1][4]
        p_1 = cv2.arcLength(con_1, True)
        p_2 = cv2.arcLength(con_2, True)
        local_max_p = max(p_1, p_2)

        # Get scale according to above local max
        scale = max_p * 1.0 / local_max_p
        print(scale)

        component_size = 512
        counter = 1
        pixel_open = 0

        for idx in range(len(contour_info)):
            x, y, w, h, contour = contour_info[idx]

            # open the bounding box a little bit
            x -= pixel_open
            y -= pixel_open
            w += pixel_open
            h += pixel_open

            # create a white image of size 512
            white_image = 255 - np.zeros(shape=(component_size, component_size))

            # calculate corresponding top-left point in white image
            new_x = int((component_size - w) / 2)
            new_y = int((component_size - h) / 2)

            # copy bounding box patch from karyotype image into white image
            white_image[new_y: (new_y + h), new_x: (new_x + w)] = image[y: (y + h), x: (x + w)]

            # rescale chromosome image
            white_image = cv2.resize(white_image, (int(component_size * scale), int(component_size * scale)))
            white_image = white_image.astype('uint8')

            # get the center patch of size 512
            white_image = image_utils.get_center_sub_image(white_image, size=component_size)

            # get current chromosome name (1, 2, ..., x, y)
            chromosome_name = str(chromosome_ids[int(idx / 2)])
            # the last chromosome name depends on what karyotype image (xx or xy)
            if idx == 45:
                chromosome_name = last_chromosome

            # save the patch contain only 1 chromosome
            cv2.imwrite(
                save_directory + "/" + chromosome_name + "/" + chromosome_type + "_" + image_file + "_" + str(idx),
                white_image)
            counter += 1


def gray_data_to_rgb(X):
    X["train"] = np.stack((X["train"],) * 3, axis=-1)
    X["val"] = np.stack((X["val"],) * 3, axis=-1)
    X["test"] = np.stack((X["test"],) * 3, axis=-1)

    return {"train": X["train"], "val": X["val"], "test": X["test"]}


def expand_data_dims(X):
    X["train"] = X["train"].reshape((X["train"].shape[0], X["train"].shape[1], X["train"].shape[2], 1))
    X["val"] = X["val"].reshape((X["val"].shape[0], X["val"].shape[1], X["val"].shape[2], 1))
    X["test"] = X["test"].reshape((X["test"].shape[0], X["test"].shape[1], X["test"].shape[2], 1))

    return {"train": X["train"], "val": X["val"], "test": X["test"]}


def expand_image_dims(image):
    image = image.reshape((*image.shape, 1))
    return image


def normalize_data(X, image_type='gray'):
    if image_type != "gray":
        raise Exception("Khang don't support RGB images yet.")

    norm_params = {"mean_image": None, "scale_value": None, "is_reversed": None}
    X["train"], norm_params = partially_normalize_data(X["train"], norm_params, is_train=True)
    X["val"], norm_params = partially_normalize_data(X["val"], norm_params, is_train=False)
    X["test"], norm_params = partially_normalize_data(X["test"], norm_params, is_train=False)

    return X, norm_params


def partially_normalize_data(X, norm_params, is_train):
    X = np.subtract(255, X)
    X = X.astype('float64')
    if is_train:
        norm_params["mean_image"] = np.mean(X, axis=0)
        norm_params["scale_value"] = 128.
        norm_params["is_reversed"] = True

    X = np.subtract(X, norm_params["mean_image"])
    X = np.divide(X, norm_params["scale_value"])

    return X, norm_params


def normalize_image(image, norm_params=None):
    image = np.subtract(255, image)
    image = image.astype('float64')
    if norm_params is None:
        norm_params["mean_image"] = np.mean(image, axis=0)
        norm_params["scale_value"] = 128.
        norm_params["is_reversed"] = True

    image = np.subtract(image, norm_params["mean_image"])
    image = np.divide(image, norm_params["scale_value"])

    return image, norm_params


def denormalize_image(image, norm_params):
    if len(image.shape) > 2:
        raise Exception("Image is not in gray-scale.")

    image = np.multiply(image, norm_params["scale_value"])
    image = np.add(image, norm_params["mean_image"])
    image = image.astype('uint8')
    if norm_params["is_reversed"]:
        image = np.subtract(255, image)

    return image


def calculate_sample_mean_image(image_files, num_iter=10):
    mean_image = None
    for i in range(num_iter):
        for image_file in image_files:
            image = cv2.imread(image_file, 0)
            new_image = generate_image(image)
            if mean_image is None:
                mean_image = new_image
            else:
                mean_image = np.abs(mean_image, new_image)

    num_sample = num_iter * len(image_files) * 1.0
    mean_image = np.divide(mean_image, num_sample)
    return mean_image


def view_sample_data(data, num_sample):
    X = data["X"]
    y_skeleton = data["y_skel"]
    y_clf = data["y_clf"]

    for i in range(num_sample):
        idx = np.random.randint(X.shape[0])
        image = X[idx].copy()
        image = np.stack((image,) * 3, axis=-1)
        print(y_skeleton[idx])
        image_with_points = image_utils.get_image_with_points(image, y_skeleton[idx])

        print("Class: " + str(y_clf[idx] + 1))
        image_utils.show_image(image_with_points, cmap=None)


def view_sample_data_list(X, y, num_sample=5):
    for i in range(num_sample):
        idx = np.random.randint(len(X))
        image = cv2.imread(X[idx], 0)
        points = y[idx]
        image_utils.show_image(image_utils.get_image_with_points(image, points))


def annotated_data_to_csv(file_in, csv_dir, image_dir, prefix="", window_radius=1, split_ratio=None, verbose=False,
                          only_center=False):
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    image_files, points_list = annotated_data_to_image_ids_and_points(file_in=file_in, file_out=None,
                                                                      image_dir=image_dir)

    data = split_data_list(image_files, points_list, ratio=split_ratio)

    image_files_train, points_list_train = data["train"]["X"], data["train"]["y"]
    annotations_train_file = csv_dir + "/" + prefix + "annotations_train.csv"
    class_mapping_train_file = csv_dir + "/" + prefix + "class_mapping_train.csv"

    if only_center:
        images_and_centers_to_csv(image_files_train, points_list_train, annotations_train_file,
                                  class_mapping_train_file,
                                  window_radius)
    else:
        images_and_points_to_csv(image_files_train, points_list_train, annotations_train_file, class_mapping_train_file,
                                 window_radius)

    if verbose:
        print("Train files are stored at:")
        print(annotations_train_file)
        print(class_mapping_train_file)

    if len(data["val"]["X"]) > 0:
        image_files_val, points_list_val = data["val"]["X"], data["val"]["y"]
        annotations_val_file = csv_dir + "/" + prefix + "annotations_val.csv"
        class_mapping_val_file = csv_dir + "/" + prefix + "class_mapping_val.csv"

        if only_center:
            images_and_centers_to_csv(image_files_val, points_list_val, annotations_val_file, class_mapping_val_file,
                                      window_radius)
        else:
            images_and_points_to_csv(image_files_val, points_list_val, annotations_val_file, class_mapping_val_file,
                                     window_radius)

        if verbose:
            print("Validation files are stored at:")
            print(annotations_val_file)
            print(class_mapping_val_file)
    else:
        if verbose:
            print("No validation data.")

    image_files_test, points_list_test = data["test"]["X"], data["test"]["y"]
    annotations_test_file = csv_dir + "/" + prefix + "annotations_test.csv"
    class_mapping_test_file = csv_dir + "/" + prefix + "class_mapping_test.csv"

    if only_center:
        images_and_centers_to_csv(image_files_test, points_list_test, annotations_test_file, class_mapping_test_file,
                                  window_radius)
    else:
        images_and_points_to_csv(image_files_test, points_list_test, annotations_test_file, class_mapping_test_file,
                                 window_radius)

    if verbose:
        print("Test files are stored at:")
        print(annotations_test_file)
        print(class_mapping_test_file)


def images_and_centers_to_csv(image_files, points_list, annotations_file, class_mapping_file, window_radius=1):
    temp_image = image_utils.read_image(image_files[0], 0)
    image_shape = temp_image.shape[0]

    center_points = [[int(points[0][0] * image_shape), int(points[0][1] * image_shape)] for points in points_list]
    rect_center_points = [[x - window_radius, y - window_radius, x + window_radius, y + window_radius] for x, y in
                          center_points]

    annotations = [[image_files[idx],
                    rect_center_points[idx][0],
                    rect_center_points[idx][1],
                    rect_center_points[idx][2],
                    rect_center_points[idx][3],
                    "center"] for idx in range(len(image_files))]

    class_mapping = [["center", 0]]

    general_utils.write_list_to_csv(annotations, annotations_file)
    general_utils.write_list_to_csv(class_mapping, class_mapping_file)


def images_and_points_to_csv(image_files, points_list, annotations_file, class_mapping_file, window_radius=1):
    num_point = 6
    temp_image = image_utils.read_image(image_files[0], 0)
    image_shape = temp_image.shape[0]

    points = [list() for _ in range(num_point)]
    rect_points = [list() for _ in range(num_point)]
    for i in range(num_point):
        points[i] = [[int(points[i][0] * image_shape), int(points[i][1] * image_shape)] for points in points_list]
        rect_points[i] = [[x - window_radius, y - window_radius, x + window_radius, y + window_radius] for x, y in
                          points[i]]

    annotations = list()
    for image_id in range(len(image_files)):
        for point_id in range(6):
            annotations.append([image_files[image_id],
                                rect_points[point_id][image_id][0],
                                rect_points[point_id][image_id][1],
                                rect_points[point_id][image_id][2],
                                rect_points[point_id][image_id][3],
                                str(point_id)])

    class_mapping = [[str(i), i] for i in range(num_point)]

    general_utils.write_list_to_csv(annotations, annotations_file)
    general_utils.write_list_to_csv(class_mapping, class_mapping_file)


def process_karyotyping_image(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    # if debug:
    #     image_utils.show_image(thresh)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_contour = len(contours)

    # Find area threshold - taking the 46-th largest area
    areas = list()
    for idx in range(num_contour):
        is_outer_contour = hierarchy[0][idx][3] == -1
        if is_outer_contour:
            area = cv2.contourArea(contours[idx])
            areas.append(area)
    areas.sort(reverse=True)

    if len(areas) < 46 and verbose:
        print("There are less than 46 chromosomes in the image.")

    area_threshold = areas[45] - 0.00001

    chosen_contours = list()
    for idx in range(num_contour):
        contour = contours[idx]
        if cv2.contourArea(contour) < area_threshold:
            continue
        chosen_contours.append(contour)

    chosen_bounding_box = list()
    component_size = 0
    # TODO: Fix this contour selection (so that we can positively select all chosen_contours, not just first 46)
    for contour in chosen_contours[:46]:
        x, y, w, h = cv2.boundingRect(contour)
        component_size = max(w, component_size)
        component_size = max(h, component_size)

        chosen_bounding_box.append([x, y, w, h])

    component_size = component_size + 50
    chromosomes = list()
    for x, y, w, h in chosen_bounding_box:
        # create a white image of size 512
        white_image = 255 - np.zeros(shape=(component_size, component_size, 3))

        # calculate corresponding top-left point in white image
        new_x = int((component_size - w) / 2)
        new_y = int((component_size - h) / 2)

        # copy bounding box patch from karyotype image into white image
        white_image[new_y: (new_y + h), new_x: (new_x + w)] = image[y: (y + h), x: (x + w)]

        white_image = white_image.astype('uint8')
        chromosomes.append(white_image)

    return chromosomes


def remove_chromosomes(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    # if debug:
    #     image_utils.show_image(thresh)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_contour = len(contours)

    # Find area threshold - taking the 46-th largest area
    areas = list()
    for idx in range(num_contour):
        is_outer_contour = hierarchy[0][idx][3] == -1
        if is_outer_contour:
            area = cv2.contourArea(contours[idx])
            areas.append(area)
    areas.sort(reverse=True)

    if len(areas) < 46 and verbose:
        print("There are less than 46 chromosomes in the image.")

    area_threshold = areas[45] - 0.01

    chosen_contours = list()
    for idx in range(num_contour):
        contour = contours[idx]
        if cv2.contourArea(contour) < area_threshold:
            continue
        chosen_contours.append(contour)

    chosen_bounding_box = list()
    component_size = 0
    for contour in chosen_contours:
        x, y, w, h = cv2.boundingRect(contour)
        component_size = max(w, component_size)
        component_size = max(h, component_size)

        chosen_bounding_box.append([x, y, w, h])

    chromosomes = list()
    for x, y, w, h in chosen_bounding_box:
        image[y: (y + h), x: (x + w)] = 255 - np.zeros(shape=image[y: (y + h), x: (x + w)].shape)

    return image


def directory_to_images_files_and_labels(directory, verbose=False):
    if verbose:
        print("Processing " + directory)

    relative_label_dirs = [x for x in next(os.walk(directory))[1]]  # get label names, such as "1", "2", ....

    label_to_id = {key: value for (value, key) in enumerate(relative_label_dirs)}  # mapping label to id (0, 1, ...)
    id_to_label = {key: value for (key, value) in enumerate(relative_label_dirs)}

    if verbose:
        print(label_to_id)

    image_ids, label_ids = list(), list()

    for label in relative_label_dirs:
        label_id = int(label_to_id[label])
        label_dir = directory + "/" + label

        if verbose:
            print("Processing " + label_dir)

        image_files = general_utils.get_all_files(label_dir)

        for image_file in image_files:
            image_file = label_dir + "/" + image_file
            image_ids.append(image_file)
            label_ids.append(label_id)

        if verbose:
            print("#Images so far: ", len(image_ids))

    if verbose:
        print("10 first image ids:", image_ids[:10])
        print("10 first label ids:", label_ids[:10])
        print("10 first label:", [id_to_label[idx] for idx in label_ids[:10]])

    return image_ids, label_ids, id_to_label, label_to_id
