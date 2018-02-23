import os

class EmptyObject:
    pass


def get_newest_submission_number(path):
    """
    Automagically detect the the current submission number by looking for a list of the .final files and finding the
    highest submission #
    """
    max_id = -1
    for f in os.listdir(path):
        filename, extension = os.path.splitext(os.path.basename(f))
        if extension == ".final" and int(filename.split("_")[1]) > max_id:
            max_id = int(filename.split("_")[1])

    max_id = max_id + 1 if max_id != 0 else 1
    return "submit_{0}".format(max_id)
