import os


def load_python_files(folder_root, recursive=True):
    """
    Returns a list of python file present in a folder
    :param folder_root: path of the folder containig the files
    :param recursive: flag to decide if the subdirectories should be included in the search
    :return:
    """

    folder_root = os.path.abspath(folder_root)

    path = [os.path.join(folder_root, f) for f in os.listdir(folder_root) if
            os.path.isfile(os.path.join(folder_root, f)) and f.endswith('.py')]

    if recursive:
        for dir in  [os.path.join(folder_root, f) for f in os.listdir(folder_root) if
            os.path.isdir(os.path.join(folder_root, f))]:

            path += load_python_files(dir)

    return path