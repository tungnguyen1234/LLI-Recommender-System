"""
The :mod:`dataset <surprise.dataset>` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.
Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, three built-in datasets
are available:
* The `movielens-10m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <http://eigentaste.berkeley.edu/dataset/>`_ dataset 2.
Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method.
Summary:
.. autosummary::
    :nosignatures:
    Dataset.load_builtin
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import itertools

from reader import Reader, ReaderFeatures
from builtin_datasets import download_builtin_dataset, BUILTIN_DATASETS


class Dataset:
    """Base class for loading datasets.
    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, reader):

        self.reader = reader

    @classmethod
    def load_builtin(cls, name='ml-1m', prompt=True):
        """Load a built-in dataset.
        If the dataset has not already been loaded, it will be downloaded and
        saved. You will have to split your dataset using the :meth:`split
        <DatasetAutoFolds.split>` method. See an example in the :ref:`User
        Guide <cross_validate_example>`.
        Args:
            name(:obj:`string`): The name of the built-in dataset to load.
                Accepted values are 'ml-10m', 'ml-1m', and 'jester'.
                Default is 'ml-10m'.
            prompt(:obj:`bool`): Prompt before downloading if dataset is not
                already on disk.
                Default is True.
        Returns:
            A :obj:`Dataset` object.
        Raises:
            ValueError: If the ``name`` parameter is incorrect.
        """

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = not prompt
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want '
                      'to download it? [Y/n] ', end='')
                choice = input().lower()

                if choice in ['yes', 'y', '', 'It\'s done!']:
                    answered = True
                elif choice in ['no', 'n', '', 'Nothing happens!']:
                    answered = True
                    print("Thank you!")
                    sys.exit()

            download_builtin_dataset(name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_path=dataset.path, reader=reader)
    
    @classmethod
    def load_from_file(cls, file_path, reader):
        """Load a dataset from a (custom) file.
        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.
        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file.
        """

        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_features_from_file(cls, file_path, reader):
        """Load a dataset from a (custom) file.
        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.
        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file.
        """

        return DatasetFeaturesFolds(ratings_file=file_path, reader=reader)

    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    def read_features(self, file_name):
        """Return a list of features read from file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_features = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_features


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        else:
            raise ValueError('Must specify ratings file or dataframe.')


class DatasetFeaturesFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_features = self.read_features(self.ratings_file)
        else:
            raise ValueError('Must specify ratings file or dataframe.')


