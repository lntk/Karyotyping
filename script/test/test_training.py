import unittest
from training import Experiment, DataGenerator
from os.path import dirname, abspath

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test_training(self):
        experiment = Experiment()
        experiment.prepare_data()
        experiment.load_model_from_file(working_dir + "/model/20190102_exp1.h5")
        # experiment.visualize_test_result()

    def test_data_generator(self):
        experiment = Experiment("20190107")
        experiment.tuan_data = data_dir + "/20190101/20190101.data"
        params = {
            "input_shape": (512, 512, 1),
            "batch_size": 1,
            "num_output": 12,
            "shuffle": True,
            "norm_params": None
        }
        data_generator = DataGenerator(**params)
