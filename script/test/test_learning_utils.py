import unittest
from script.util import image_utils, learning_utils
from os.path import dirname, abspath
import pickle
from test import test_utils
import numpy as np

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test(self):
        return
