import os
import sys
from os import listdir
from os.path import isfile
from os.path import join

from svm.convex_hull import cal_lmbda_max
from svm.kernel import polynomial_kernel
from svm.sk_algorithm import SKAlg
from zener_generator import FILE_NAME_LETTER_MAP
import ntpath

def get_file_name(filepath):
    return os.path.splitext(ntpath.basename(filepath))[0]

def get_label(filename):
    name_splits = filename.split("_")
    return name_splits[1].split(".")[0]

def is_valid_format(filename):
    try:
        name_splits = filename.split("_")
        label = get_label(filename)
        return len(name_splits) == 2 and name_splits[0].isdigit() and label in FILE_NAME_LETTER_MAP.values()
    except Exception:
        return False


def read_input_files(folder_name):
    files = []
    folder_path = join(os.getcwd(), folder_name)
    for filename in listdir(folder_path):
        filepath = join(folder_path, filename)
        if isfile(filepath) and is_valid_format(filename):
            files.append(filepath)
    return files

def get_labeled_input(folder_name, class_letter):
    X = read_input_files(folder_name)
    I_positive = []
    I_negative = []
    for i in xrange(len(X)):
        x = X[i]
        label = get_label(get_file_name(x))
        if label == class_letter:
            I_positive.append(i)
        else:
            I_negative.append(i)
    return X, I_positive, I_negative

if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise Exception("Invalid number of arguments")
    epsilon = float(sys.argv[1])
    max_updates = int(sys.argv[2])
    class_letter = sys.argv[3]
    model_file_name = sys.argv[4]
    train_folder_name = sys.argv[5]
    X, I_positive, I_negative = get_labeled_input(train_folder_name, class_letter)
    if len(X) == 0:
        print("NO DATA")
    alg = SKAlg(polynomial_kernel)
    lmbda = cal_lmbda_max(X, I_positive, I_negative)
    alg.train(X, I_positive, I_negative, epsilon, max_updates, lmbda)
    alg.serialize(model_file_name)