import unittest
from pipeline.pipeline import Pipeline
from util import image_utils
from os.path import dirname, abspath

working_dir = dirname(dirname(dirname(abspath("X"))))
data_dir = working_dir + "/data"


class Test(unittest.TestCase):
    def test_generate_chromosome_cluster(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = image_utils.read_image(image_file)
        image_utils.show_image(image)
        chromosome_cluster = Pipeline.generate_chromosome_cluster(image)
        image_utils.show_image(chromosome_cluster, cmap=None)

    def test_read_image(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = Pipeline.read_image(image_file)
        image_utils.show_image(image)

    def test_extract_chromosomes(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = image_utils.read_image(image_file)
        chromosomes = Pipeline.extract_chromosomes(image)
        for chromosome in chromosomes:
            image_utils.show_image(chromosome, cmap=None)

    def test_straighten_chromosomes(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = image_utils.read_image(image_file)
        chromosomes = Pipeline.extract_chromosomes(image)
        _ = Pipeline.straighten_chromosomes(chromosomes, debug=True)

    def test_detect_interesting_points(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = image_utils.read_image(image_file)
        chromosomes = Pipeline.extract_chromosomes(image)
        straightened_chromosomes = Pipeline.straighten_chromosomes(chromosomes)
        _ = Pipeline.detect_interesting_points(straightened_chromosomes, verbose=True)

    def test_organize_chromosomes(self):
        image_file = data_dir + "/test/karyotype.bmp"
        image = Pipeline.read_image(image_file)
        chromosomes = Pipeline.extract_chromosomes(image)
        straightened_chromosomes = Pipeline.straighten_chromosomes(chromosomes)
        # interesting_points = Pipeline.detect_interesting_points(straightened_chromosomes)
        interesting_points = None
        classified_chromosomes = Pipeline.classify_chromosomes(straightened_chromosomes, interesting_points)
        karyotyping_image = Pipeline.organize_chromosomes(classified_chromosomes)
        image_utils.show_image(karyotyping_image)


if __name__ == '__main__':
    unittest.main()
