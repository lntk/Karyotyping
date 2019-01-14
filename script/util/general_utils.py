import os
import json
from shutil import copyfile
import csv


def make_a_copy(src, dst):
    copyfile(src, dst)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def write_list(l, file_name):
    with open(file_name, 'w') as file_handle:
        json.dump(l, file_handle)


def read_list(file_name):
    with open(file_name, 'r') as file_handle:
        l = json.load(file_handle)
        return l


def get_all_files(directory):
    """
    :param directory: A directory
    :return: List of file names in the directory
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files


def rename_files_in_directory(directory, prefix, suffix):
    """
    This function renames (by enumerating) all files in a directory
    E.g.:
    If:
    - prefix = 'karyotype'
    - suffix = '.bmp'
    then:
        '123132', '12312', '2132' --> karyotype_1.bmp, karyotype_2.bmp, karyotype_3.bmp

    :param directory:
    :param prefix:
    :param suffix:
    :return:
    """
    files = get_all_files(directory)
    for i in range(len(files)):
        file_name = directory + "/" + files[i]
        new_file_name = directory + "/" + prefix + "_" + str('{:03}'.format(i + 1)) + suffix
        os.rename(file_name, new_file_name)


def delete_file(filename):
    os.remove(filename)


def read_lines(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def write_lines(l, file_name):
    with open(file_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)


def write_list_to_csv(csvData, file_name):
    with open(file_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

    csvFile.close()
