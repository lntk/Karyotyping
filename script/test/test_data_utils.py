import unittest
from script.util import data_utils, general_utils, learning_utils, image_utils
from os.path import dirname, abspath
import pickle
import numpy as np
from script.test import test_utils

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test_process_annotated_data(self):
        file_in = data_dir + "/20190101/20190101.data"
        file_out = data_dir + "/20190101/20190101_processed.data"
        # file_hard = data_dir + "/20190101/20190101_difficult.txt"
        image_dir = data_dir + "/chromosome"
        data_utils.process_annotated_data(file_in, file_out, image_dir, verbose=True)

    def test_preprocess_hard_file(self):
        file_hard = data_dir + "/20190101/20190101_difficult.txt"
        lines = general_utils.read_lines(file_hard)
        lines = [line[7:] for line in lines]
        general_utils.write_lines(lines, file_hard)

    def test_generate_data(self):
        file_in = data_dir + "/test/test_processed.data"
        file_out = data_dir + "/test/test_augmented.data"
        data_utils.generate_data(file_in, file_out, debug=True)

    def test_excerpt_tuan_data(self):
        file_in = data_dir + "/20190101/20190101.data"
        file_out = data_dir + "/test/test.data"

        with open(file_in, 'rb') as handle:
            dictionary = pickle.load(handle)

        X = dictionary["fns"]
        y_skeleton = dictionary["skeleton"]
        y_clf = dictionary["class"]

        data = {"fns": X[:10], "skeleton": y_skeleton[:10], "class": y_clf[:10]}
        pickle.dump(data, open(file_out, 'wb'))

    def test_excerpt_annotated_data(self):
        file_in = data_dir + "/20190101/20190101_processed.data"
        file_out = data_dir + "/test/test_processed.data"

        with open(file_in, 'rb') as handle:
            dictionary = pickle.load(handle)

        X = dictionary["X"]
        y_skeleton = dictionary["y_skeleton"]
        y_clf = dictionary["y_clf"]

        data = {'X': X[:10], 'y_skeleton': y_skeleton[:10], 'y_clf': y_clf[:10]}
        pickle.dump(data, open(file_out, 'wb'))

    def test_data_augmentation(self):
        file_in = data_dir + "/20190101/20190101_processed.data"
        with open(file_in, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            dictionary = u.load()

        X = dictionary['X']
        y_skeleton = dictionary['y_skeleton']
        y_clf = dictionary['y_clf']

        X = X[:10]
        y_skeleton = y_skeleton[:10]
        y_clf = y_clf[:10]

        new_X, new_y_skeleton, new_y_clf = learning_utils.data_augmentation(X, y_skeleton, y_clf, 60, 10, 10,
                                                                            iteration=3)

        data = {'X': new_X, 'y_skel': new_y_skeleton, 'y_clf': new_y_clf}
        data_utils.view_sample_data(data, num_sample=10)

    def test_denormalize_image(self):
        X_train = [test_utils.get_test_image() for _ in range(10)]
        X_val = [test_utils.get_test_image() for _ in range(1)]
        X_test = [test_utils.get_test_image() for _ in range(1)]

        image_utils.show_image(X_test[0], name="Before")

        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        X_test = np.asarray(X_test)

        X = {"train": X_train, "val": X_val, "test": X_test}

        X, norm_params = data_utils.normalize_data(X)
        image = X["test"][0].copy()
        image = data_utils.denormalize_image(image, norm_params)

        image_utils.show_image(image, name="After")

    def test_read_annotated_file(self):
        file_in = data_dir + "/20190101/20190101.data"
        image_dir = data_dir + "/chromosome"

        data_utils.read_annotated_file(file_in, image_dir, verbose=True)

    def test_directory_to_images_files_and_labels(self):
        image_dir = data_dir + "/chromosome"
        data_utils.directory_to_images_files_and_labels(image_dir, verbose=True)
