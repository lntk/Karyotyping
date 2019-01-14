"""
    @author: lntk
"""
import cv2
import numpy as np
from script.util import image_utils, math_utils, general_utils, data_utils


class Pipeline:
    @staticmethod
    def generate_chromosome_cluster(karyotyping_image, save_dir=None, popup=None):
        chromosome_size = 256
        chromosomes = data_utils.process_karyotyping_image(karyotyping_image, verbose=True)
        chromosome_cluster = 255 - np.zeros(shape=(chromosome_size * 7, chromosome_size * 7, 3), dtype='uint8')

        i, j = 0, 0
        for chromosome in chromosomes:
            chromosome = cv2.resize(chromosome, (256, 256))
            chromosome_cluster[i * chromosome_size: (i + 1) * chromosome_size, j * chromosome_size: (j + 1) * chromosome_size] = \
                chromosome[0:chromosome_size, 0:chromosome_size]

            # this is equivalent to 2 for-loops
            j += 1
            i += int(j / 7)
            j = j % 7

        if save_dir is not None:
            general_utils.create_directory(save_dir)
            cv2.imwrite(save_dir + "/chromosome_cluster.bmp", chromosome_cluster)

        return chromosome_cluster

    @staticmethod
    def read_image(image_file):
        return image_utils.read_image(image_file)

    @staticmethod
    def extract_chromosomes(image, save_dir=None, popup=None):
        chromosomes = data_utils.process_karyotyping_image(image)

        if save_dir is not None:
            general_utils.create_directory(save_dir)
            for idx in range(len(chromosomes)):
                cv2.imwrite(save_dir + "/" + str(idx + 1) + ".bmp", chromosomes[idx])

        return chromosomes

    @staticmethod
    def straighten_chromosomes(chromosomes, save_dir=None, popup=None):
        num_chromosome = len(chromosomes)

        """ Handle popup """
        popup_counter = 1
        if popup is not None:
            popup.set_text("0/" + str(num_chromosome) + " chromosomes processed.")

        straightened_chromosomes = list()
        for idx in range(num_chromosome):
            chromosome = chromosomes[idx].copy()
            gray = cv2.cvtColor(chromosome, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

            # get 2 end points of the best line
            x1, y1, x2, y2 = image_utils.find_best_line_hough_transform(binary)

            # make the first point higher than the second point
            if y1 > y2:
                x1, x2 = math_utils.swap(x1, x2)
                y1, y2 = math_utils.swap(y1, y2)

            angle = math_utils.get_angle_between_two_points([x1, y1], [x2, y2])

            straightened_chromosome = image_utils.rotate_image(chromosome, angle + 90)

            if save_dir is not None:
                general_utils.create_directory(save_dir)

                chromosome_with_line = cv2.line(chromosome.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
                straightened_line = image_utils.rotate_points([[x1, y1], [x2, y2]], angle + 90, shape=chromosome.shape)
                rotated_x1, rotated_y1 = straightened_line[0]
                rotated_x2, rotated_y2 = straightened_line[1]
                straightened_chromosome_with_line = cv2.line(straightened_chromosome.copy(), (rotated_x1, rotated_y1), (rotated_x2, rotated_y2), (0, 255, 0), 2)

                """ Save images """
                cv2.imwrite(save_dir + "/" + str(idx) + ".bmp", chromosome_with_line)
                cv2.imwrite(save_dir + "/" + str(idx) + "_straightened.bmp", straightened_chromosome_with_line)

            straightened_chromosomes.append(straightened_chromosome)

            """ Handle popup """
            if popup is not None:
                popup.set_text(str(popup_counter) + "/" + str(num_chromosome) + " chromosomes processed.")
                popup_counter += 1

        return straightened_chromosomes

    @staticmethod
    def not_detect_interesting_points(chromosomes, model_path="default", verbose=False, save_dir=None, popup=None):
        return

    @staticmethod
    def detect_interesting_points(chromosomes, model_path="default", verbose=False, save_dir=None, popup=None):
        """ Local import """
        from script.pipeline.detection import detect_interesting_points, load_model

        interesting_points = list()
        model = load_model(model_path=model_path)
        for idx in range(len(chromosomes)):
            chromosome = chromosomes[idx]
            points, draw, image_with_points = detect_interesting_points(chromosome, model, verbose=verbose)

            if save_dir is not None:
                general_utils.create_directory(save_dir)
                cv2.imwrite(save_dir + "/" + str(idx + 1) + "_draw.bmp", draw)
                cv2.imwrite(save_dir + "/" + str(idx + 1) + "_points.bmp", image_with_points)

            interesting_points.append(points)

        return interesting_points

    @staticmethod
    def classify_chromosomes(chromosomes, points, save_dir=None, popup=None):
        """
        This function randomly classifies chromosomes.

        :param chromosomes:
        :param points:
        :param save_dir:
        :param popup:
        :return:
        """

        # TODO: Remove this in the future
        if chromosomes is None:
            print("No chromosomes. Return None.")
            return None

        """ Default chromosome ids """
        chromosome_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "x", "y"]

        """ Duplicate chromosomes ids (because of having 2 chromosomes/id, 'x' and 'y' excepted) and shuffle all """
        all_chromosome_ids = chromosome_ids + chromosome_ids
        all_chromosome_ids.remove("x")
        all_chromosome_ids.remove("y")

        from random import shuffle
        shuffle(all_chromosome_ids)

        """ Put chromosomes to ids """
        classified_chromosomes = dict()
        for i in range(len(chromosomes)):
            idx = all_chromosome_ids[i]
            if idx not in classified_chromosomes:
                classified_chromosomes[idx] = [chromosomes[i]]
            else:
                classified_chromosomes[idx].append(chromosomes[i])

        if save_dir is not None:
            counter = 0
            general_utils.create_directory(save_dir)
            for idx in chromosome_ids:
                for chromosome in classified_chromosomes[idx]:
                    cv2.imwrite(save_dir + "/" + str(counter) + ".bmp", chromosome)
                    counter += 1

        return classified_chromosomes

    @staticmethod
    def organize_chromosomes(chromosomes, debug=False, save_dir=None):
        template = 255 - np.zeros(shape=(800, 2048, 3), dtype='uint8')
        blocks = {
            "1": [0, 0, 256, 200],
            "2": [256, 0, 512, 200],
            "3": [512, 0, 768, 200],
            "4": [1536, 0, 1792, 200],
            "5": [1792, 0, 2048, 200],
            "6": [0, 200, 256, 400],
            "7": [256, 200, 512, 400],
            "8": [512, 200, 768, 400],
            "9": [768, 200, 1024, 400],
            "10": [1024, 200, 1280, 400],
            "11": [1280, 200, 1536, 400],
            "12": [1536, 200, 1792, 400],
            "13": [0, 400, 256, 600],
            "14": [256, 400, 512, 600],
            "15": [512, 400, 768, 600],
            "16": [1280, 400, 1536, 600],
            "17": [1536, 400, 1792, 600],
            "18": [1792, 400, 2048, 600],
            "19": [0, 600, 256, 800],
            "20": [256, 600, 512, 800],
            "21": [768, 600, 1024, 800],
            "22": [1024, 600, 1280, 800],
            "x": [1536, 600, 1792, 800],
            "y": [1792, 600, 2048, 800],
        }

        chromosome_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "x", "y"]

        for idx in chromosome_ids:
            x1, y1, x2, y2 = blocks[idx]
            block_image = image_utils.get_block_image(chromosomes[idx], idx, shape=(200, 256, 3))
            template[y1: y2, x1: x2] = block_image
            if debug:
                image_utils.get_block_image(chromosomes[idx], idx, shape=(200, 256, 3))

        if save_dir is not None:
            general_utils.create_directory(save_dir)
            cv2.imwrite(save_dir + "/karyotyping.bmp", template)

        return template

    @staticmethod
    def run(image_file, save_dir, model_path):
        general_utils.create_directory(save_dir + "/pipeline")

        image = Pipeline.read_image(image_file)

        image = Pipeline.generate_chromosome_cluster(image, save_dir=save_dir + "/pipeline/1_generate_chromosome_cluster")

        chromosomes = Pipeline.extract_chromosomes(image, save_dir=save_dir + "/pipeline/2_extract_chromosomes")

        straightened_chromosomes = Pipeline.straighten_chromosomes(chromosomes, save_dir=save_dir + "/pipeline/3_straighten_chromosomes")

        interesting_points = Pipeline.detect_interesting_points(straightened_chromosomes, save_dir=save_dir + "/pipeline/4_detect_interesting_points",
                                                                model_path=model_path)

        classified_chromosomes = Pipeline.classify_chromosomes(straightened_chromosomes, interesting_points)

        karyotyping_image = Pipeline.organize_chromosomes(classified_chromosomes, save_dir=save_dir + "/pipeline/5_organize_chromosomes")

        return karyotyping_image
