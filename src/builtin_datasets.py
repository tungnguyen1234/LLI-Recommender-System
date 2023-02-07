'''This feature was deployed directly from SurPRISE package'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import zipfile
import os

from os.path import join
from collections import namedtuple
from urllib.request import urlretrieve


def get_dataset_dir():
    """Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.data/ and path is 'PWD'
    """

    folder = os.environ.get('PWD') + '/.data'
    if not os.path.exists(folder):
        os.makedirs(folder) 
    return folder


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-10m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-10m.zip',
            path=join(get_dataset_dir(), 'ml-10m/ml-10M100K/ratings.dat'),
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=join(get_dataset_dir(), 'ml-1m/ml-1m/ratings.dat'),
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep="::")
        ),
    'ml-100k': 
        BuiltinDataset(
            url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            path=join(get_dataset_dir(), 'ml-100k/ml-100k/u.data'),
            reader_params=dict(
                line_format="user item rating timestamp", 
                rating_scale=(1, 5), 
                sep="\t")
        ),
    'jester':
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip',
            path=join(get_dataset_dir(), 'jester/jester_ratings.dat'),
            reader_params=dict(line_format='user item rating',
                               rating_scale=(-10, 10))
        )
}


def download_builtin_dataset(name):

    dataset = BUILTIN_DATASETS[name]

    print('Trying to download dataset from ' + dataset.url + '...')
    tmp_file_path = join(get_dataset_dir(), 'tmp.zip')
    urlretrieve(dataset.url, tmp_file_path)

    with zipfile.ZipFile(tmp_file_path, 'r') as tmp_zip:
        tmp_zip.extractall(join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print('Done! Dataset', name, 'has been saved to',
          join(get_dataset_dir(), name))