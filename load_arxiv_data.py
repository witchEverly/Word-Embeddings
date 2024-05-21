from os.path import expanduser
import pandas as pd


def load_arxiv_data(sample=True, columns=None):
    """
    Loads the arXiv dataset from a local file system without headers.
    The dataset can be loaded in either sample or full size.
    :param sample: bool - Whether to load the sample dataset or the full dataset.
    :param columns: list - The columns to load from the dataset.
    :return: pd.DataFrame - The arXiv dataset.
    """
    # default_columns = ['title', 'abstract', 'update_date', 'authors', 'id']
    file_path = '~/data/arxiv_data_sample.csv' if sample else '~/data/arxiv_data.csv'

    try:
        # Load the dataset without headers
        arxiv = pd.read_csv(expanduser(file_path), usecols=['title'])

        # Select only the specified columns
        if columns is not None:
            arxiv = arxiv[columns]

        return arxiv
    except FileNotFoundError as e:
        print(f'File not found: {e}')
    except KeyError as e:
        print(f'Column names: {columns} not found in the dataset: {e}')


def stream_arxiv_data(sample=True, columns=None):
    """
    Streams the arXiv dataset from a local file system without headers.
    The dataset can be streamed in either sample or full size.
    :param sample: bool - Whether to stream the sample dataset or the full dataset.
    :param columns: list - The columns to load from the dataset.
    :return: generator of pd.DataFrame - The arXiv dataset.
    """
    default_columns = ['title', 'abstract', 'update_date', 'authors', 'id']
    file_path = '~/data/arxiv_data_sample.csv' if sample else '~/data/arxiv_data.csv'

    try:
        # Stream the dataset without headers
        for chunk in pd.read_csv(expanduser(file_path), chunksize=1000, header=None, names=default_columns):
            # Select only the specified columns
            if columns is not None:
                chunk = chunk[columns]
            yield chunk
    except FileNotFoundError as e:
        print(f'File not found: {e}')
    except KeyError as e:
        print(f'Column names: {columns} not found in the dataset: {e}')
