import pickle
import os


def save_as_pickle_file(directory: str, filename: str, data: object) -> None:
    """
    Saves the given data as a pickle file in the specified directory.

    Args:
        directory (str): The directory where the pickle file will be saved.
        filename (str): The name of the pickle file.
        data (object): The data to be saved as a pickle file.

    Returns:
        None
    """
    full_path = os.path.join(directory, filename)
    with open(full_path, "wb") as file:
        pickle.dump(data, file)


def import_pkl_file(file_path):
    """
    Imports a pickle file and returns the data.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    data: The data loaded from the pickle file.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data
