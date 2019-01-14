"""
    @author: lntk
"""

from os.path import dirname, abspath
import sys

working_dir = dirname(dirname(abspath("X")))
data_dir = working_dir + "/data"
model_dir = working_dir + "/model"
script_dir = working_dir + "/script"

sys.path.append(script_dir)
sys.path.append(script_dir + "/model/keras-retinanet")

from pipeline.pipeline import Pipeline


def main():
    save_dir = "../data"
    image_file = "../data/test/karyotype.bmp"
    model_path = "../model/default_inference.h5"
    Pipeline.run(image_file=image_file, save_dir=save_dir, model_path=model_path)


if __name__ == '__main__':
    main()
