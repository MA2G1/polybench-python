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


class MultidimensionalArrayListBase(list, MultidimensionalArray):
    """Defines a multidimensional array using a Python's list as the base class.

    This class offers common functionality for list-based arrays such as memory initialization."""

    def __init__(self, shape: tuple, dtype, offset: int = 0):
        MultidimensionalArray.__init__(self, shape, dtype, offset)
        list.__init__(self, self.__list_init__(shape, dtype, offset))

    @abstractmethod
    def __list_init__(self, shape: tuple, dtype, offset: int) -> list: ...


class MultidimensionalArrayList(MultidimensionalArrayListBase):

    def __init__(self, shape: tuple, dtype, offset: int = 0):
        super(MultidimensionalArrayList, self).__init__(shape, dtype, offset)

    def __list_init__(self, shape: tuple, dtype, offset: int) -> list:
        return _create_array_rec(len(shape), shape, dtype(0))

    def __getitem__(self, item: tuple):
        if type(item) is tuple:
            if len(item) == 2:
                return super(MultidimensionalArrayList, self).__getitem__(item[0])[item[1]]
            elif len(item) == 3:
                return super(MultidimensionalArrayList, self).__getitem__(item[0])[item[1]][item[2]]
        else:
            return super(MultidimensionalArrayList, self).__getitem__(item)

    def __setitem__(self, key, value):
        if type(key) is tuple:
            if len(key) == 2:
                super(MultidimensionalArrayList, self).__getitem__(key[0]).__setitem__(key[1], value)
            elif len(key) == 3:
                super(MultidimensionalArrayList, self).__getitem__(key[0]).__getitem__(key[1]).__setitem__(key[2], value)
        else:
            super(MultidimensionalArrayList, self).__setitem__(key, value)


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

    def __list_init__(self, shape: tuple, dtype, offset: int) -> list:
        # Flatten the list by converting the multidimensional array into a single dimension array
        self.size = 1  # Store the number of total elements the list will have
        for size in shape:
            self.size *= size

        return [dtype(0) for x in range(self.size)]

    def __getitem__(self, item: tuple):
        """Custom __getitem__ supporting multiple indexes via tuple for multidimensional lists"""
        if type(item) is tuple:
            if len(item) == 2:
                # item is a tuple, (i, j)
                return super(MultidimensionalArrayListFlattened, self).__getitem__(item[0] * self.DIM2_SIZE + item[1])
            elif len(item) == 3:
                # item is a tuple, (i, j, k)
                return super(MultidimensionalArrayListFlattened, self).__getitem__((item[0] * self.DIM2_SIZE + item[1])
                                                                                   * self.DIM3_SIZE + item[2])
        else:
            return super(MultidimensionalArrayListFlattened, self).__getitem__(item)

    def __setitem__(self, key, value):
        if type(key) is tuple:
            if len(key) == 2:
                super(MultidimensionalArrayListFlattened, self).__setitem__(key[0] * self.DIM2_SIZE + key[1], value)
            elif len(key) == 3:
                super(MultidimensionalArrayListFlattened, self).__setitem__((key[0] * self.DIM2_SIZE + key[1])
                                                                            * self.DIM3_SIZE + key[2], value)
        else:
            super(MultidimensionalArrayListFlattened, self).__setitem__(key, value)


class MultidimensionalArrayNumPy(numpy.ndarray, MultidimensionalArray):

    def __init__(self, shape: tuple, dtype, offset: int = 0, order: str = 'C'):
        super(MultidimensionalArrayNumPy, self).__init__(shape, dtype, offset)
        self.fill(dtype(0))
