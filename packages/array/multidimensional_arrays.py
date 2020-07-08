#
# Copyright 2020 Miguel Angel Abella Gonzalez <miguel.abella@udc.es>
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module offers multiple array implementations."""
import collections
from abc import abstractmethod

import numpy


def _create_array_rec(dimensions: int, sizes: list, initialization_value: int = 0) -> list:
    """Auxiliary recursive function for creating a new matrix consisting on lists of lists.

    This method assumes that the parameters were previously validated.

    :param int dimensions: the number of dimensions to create. One dimension creates a list, two a matrix (list
        of list) and so on.
    :param list[int] sizes: a list of integers, each one representing the size of a dimension. The first element
        of the list represents the size of the first dimension, the second element the size of the second
        dimension and so on. If this list is smaller than the actual number of dimensions then the last size
        read is used for the remaining dimensions.
    :param int initialization_value: (optional; default = 0) the value to use for initializing the arrays during
        their creation.
    :return: a list representing an array of N dimensions.
    :rtype list:
    """
    if dimensions == 1:
        # Just create a list with as many zeros as specified in sizes[0]
        return [initialization_value for x in range(sizes[0])]

    if len(sizes) == 1:
        # Generate lists of the same size per dimension
        return [_create_array_rec(dimensions - 1, sizes, initialization_value) for x in range(sizes[0])]
    else:
        # Generate lists with unique sizes per dimension
        return [_create_array_rec(dimensions - 1, sizes[1:], initialization_value) for x in
                range(sizes[0])]


class MultidimensionalArray:

    @abstractmethod
    def __init__(self, shape: tuple, dtype, offset: int = 0):
        self.shape = shape  # quick reference
        self.dtype = dtype
        self.offset = offset


class MultidimensionalArrayListBase(collections.UserList, MultidimensionalArray):
    """Defines a multidimensional array using a Python's list as the base class.

    This class offers common functionality for list-based arrays such as memory initialization."""

    def __init__(self, shape: tuple, dtype, offset: int = 0):
        MultidimensionalArray.__init__(self, shape, dtype, offset)
        # list.__init__(self, self.__list_init__(shape, dtype, offset))
        collections.UserList.__init__(self, self.__list_init__(shape, dtype, offset))

    @abstractmethod
    def __list_init__(self, shape: tuple, dtype, offset: int) -> list: ...


class MultidimensionalArrayList(MultidimensionalArrayListBase):

    def __init__(self, shape: tuple, dtype, offset: int = 0):
        super(MultidimensionalArrayList, self).__init__(shape, dtype, offset)

        def get_item_1d(self, item):
            if type(item) == tuple:
                return super(MultidimensionalArrayList, self).__getitem__(item[0])
            else:
                return super(MultidimensionalArrayList, self).__getitem__(item)

        def get_item_2d(self, item):
            if type(item) == tuple:
                return super(MultidimensionalArrayList, self).__getitem__(item[0])[item[1]]
            else:
                return super(MultidimensionalArrayList, self).__getitem__(item)

        def get_item_3d(self, item):
            if type(item) == tuple:
                return super(MultidimensionalArrayList, self).__getitem__(item[0])[item[1]][item[2]]
            else:
                return super(MultidimensionalArrayList, self).__getitem__(item)

        def set_item_1d(self, key, value):
            if type(key) == tuple:
                super(MultidimensionalArrayList, self).__setitem__(key[0], value)
            else:
                super(MultidimensionalArrayList, self).__setitem__(key, value)

        def set_item_2d(self, key, value):
            if type(key) == tuple:
                super(MultidimensionalArrayList, self).__getitem__(key[0]).__setitem__(key[1], value)
            else:
                super(MultidimensionalArrayList, self).__setitem__(key, value)

        def set_item_3d(self, key, value):
            if type(key) == tuple:
                super(MultidimensionalArrayList, self).__getitem__(key[0]).__getitem__(key[1]).__setitem__(key[2], value)
            else:
                super(MultidimensionalArrayList, self).__setitem__(key, value)

        if len(shape) == 1:
            self.__getitem2__ = get_item_1d
            self.__setitem2__ = set_item_1d
        elif len(shape) == 2:
            self.__getitem2__ = get_item_2d
            self.__setitem2__ = set_item_2d
        elif len(shape) == 3:
            self.__getitem2__ = get_item_3d
            self.__setitem2__ = set_item_3d

        MultidimensionalArrayList.__getitem__ = self.__getitem2__
        MultidimensionalArrayList.__setitem__ = self.__setitem2__

    def __list_init__(self, shape: tuple, dtype, offset: int) -> list:
        return _create_array_rec(len(shape), shape, dtype(0))


class MultidimensionalArrayListFlattened(MultidimensionalArrayListBase):

    def __init__(self, shape: tuple, dtype, offset: int = 0):
        super(MultidimensionalArrayListFlattened, self).__init__(shape, dtype, offset)

        # Generate convenient quick references for each dimension size
        i = 1
        for size in shape:
            if i == 1:
                self.DIM1_SIZE = size
            elif i == 2:
                self.DIM2_SIZE = size
            elif i == 3:
                self.DIM3_SIZE = size
            i += 1

        def get_item_1d(self, item):
            # if type(item) is tuple:
            if isinstance(item, tuple):
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item[0])
            else:
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item)

        def get_item_2d(self, item):
            # if type(item) is tuple:
            if isinstance(item, tuple):
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item[0] * self.DIM2_SIZE + item[1])
            else:
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item)

        def get_item_3d(self, item):
            # if type(item) is tuple:
            if isinstance(item, tuple):
                return super(MultidimensionalArrayListFlattened, self).__getitem__((item[0] * self.DIM2_SIZE + item[1])
                                                                                   * self.DIM3_SIZE + item[2])
            else:
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item)

        def set_item_1d(self, key, value):
            # if type(key) is tuple:
            if isinstance(key, tuple):
                super(MultidimensionalArrayListFlattened, self).__setitem__(key[0], value)
            else:
                super(MultidimensionalArrayListFlattened, self).__setitem__(key, value)

        def set_item_2d(self, key, value):
            # if type(key) is tuple:
            if isinstance(key, tuple):
                super(MultidimensionalArrayListFlattened, self).__setitem__(key[0] * self.DIM2_SIZE + key[1], value)
            else:
                super(MultidimensionalArrayListFlattened, self).__setitem__(key, value)

        def set_item_3d(self, key, value):
            # if type(key) is tuple:
            if isinstance(key, tuple):
                super(MultidimensionalArrayListFlattened, self).__setitem__((key[0] * self.DIM2_SIZE + key[1])
                                                                            * self.DIM3_SIZE + key[2], value)
            else:
                super(MultidimensionalArrayListFlattened, self).__setitem__(key, value)

        if len(shape) == 1:
            MultidimensionalArrayListFlattened.__getitem__ = get_item_1d
            MultidimensionalArrayListFlattened.__setitem__ = set_item_1d
        elif len(shape) == 2:
            MultidimensionalArrayListFlattened.__getitem__ = get_item_2d
            MultidimensionalArrayListFlattened.__setitem__ = set_item_2d
        elif len(shape) == 3:
            MultidimensionalArrayListFlattened.__getitem__ = get_item_3d
            MultidimensionalArrayListFlattened.__setitem__ = set_item_3d

    def __list_init__(self, shape: tuple, dtype, offset: int) -> list:
        # Flatten the list by converting the multidimensional array into a single dimension array
        self.size = 1  # Store the number of total elements the list will have
        for size in shape:
            self.size *= size

        return [dtype(0) for x in range(self.size)]


class MultidimensionalArrayNumPy(numpy.ndarray, MultidimensionalArray):

    def __init__(self, shape: tuple, dtype, offset: int = 0, order: str = 'C'):
        super(MultidimensionalArrayNumPy, self).__init__(shape, dtype, offset)
        self.fill(dtype(0))
