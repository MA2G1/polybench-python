# Copyright 2019 Miguel Angel Abella Gonzalez <miguel.abella@udc.es>
#
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

"""This module offers the base Polybench class to be used by kernel implementations."""
from enum import Enum, auto
from sys import stderr


class DatasetSize(Enum):
    """Define the possible values for selecting dataset sizes.

    Instead of manually managing the values of this enumeration we let the python interpreter initialize them.
    """
    MINI = auto()
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()
    EXTRA_LARGE = auto()


class Polybench:
    """This class offers common methods for making and evaluating different kernels.

    This class is not meant to be instantiated (abstract class) but rather each kernel must inherit from it and override
    the following methods:
        - __init__()
        - kernel()
        - print_array_custom()
    """

    _POLYBENCH_DUMP_TARGET = stderr  # Dump user messages into stderr, as in Polybench/C

    DATA_TYPE = int  # The data type used for the current benchmark (used for conversions and formatting)
    DATA_PRINT_MODIFIER = '{:d} '  # A default print modifier. Should be set up in run()

    def __init__(self):
        """Class constructor.

        Since this is an abstract class, this method prevents its instantiation by throwing a RuntimeError.
        This method **MUST** be overridden by subclasses.
        """
        raise RuntimeError('Abstract classes cannot be instantiated.')

    def _create_array_rec(self, dimensions: int, sizes: list, initialization_value: int = 0) -> list:
        """Auxiliary recursive method for creating a new array.

        This method assumes that the parameters were previously validated (in the create_array method).

        :param int dimensions: the number of dimensions to create. One dimension creates a list, two a matrix and so on.
        :param list[int] sizes: a list of integers, each one representing the size of a dimension. The first element of the
            list represents the size of the first dimension, the second element the size of the second dimension and so
            on. If this list is smaller than the actual number of dimensions then the last size read is used for the
            remaining dimensions.
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
            return [self._create_array_rec(dimensions - 1, sizes, initialization_value) for x in range(sizes[0])]
        else:
            # Generate lists with unique sizes per dimension
            return [self._create_array_rec(dimensions - 1, sizes[1:], initialization_value) for x in range(sizes[0])]

    def create_array(self, dimensions: int, sizes: list, initialization_value: int = 0) -> list:
        """
        Create a new array with a specified size.

        :param int dimensions: specifies the number of dimensions of the array.
        :param list[int] sizes: allows to specify the number of elements for each dimension. If this parameter is a list
            with one element, this element represents the size of all dimensions, otherwise the list must have as many
            elements as specified by the dimensions parameter and each element represents the size of a dimension.
            The size of the first dimension is specified by the first element on the list, the size of the second
            dimension is represented by the second element of the list and so on.
        :param int initialization_value: (optional; default = 0) the value at which all array elements are set.
        :return: a list representing an array of M dimensions.
        :rtype list:
        """
        # Sanity check: "dimensions" must be of type integer.
        if not isinstance(dimensions, int):
            raise AssertionError('Invalid type for parameter "dimensions". '
                                 f'Expected "{type(0)}"; received "{type(dimensions)}"')

        # Sanity check: "sizes" must be a list (of integers).
        if not isinstance(sizes, list):
            raise AssertionError('Invalid type for parameter "sizes". '
                                 f'Expected "{type([])}"; received "{type(sizes)}"')

        # NOTE: type checking for "initialization_value" may not be interesting at all...
        # Sanity check: "initialization_value" must be of type integer.
        if not isinstance(initialization_value, (int, float, complex)):
            raise AssertionError('Invalid type for parameter "initialization_value". '
                                 f'Expected one of "[{type(0)}, {type(0.0)}, {type(0j)}]"; '
                                 f'received "{type(initialization_value)}"')

        # Sanity check: "dimensions" must be a non-zero positive integer.
        if dimensions < 1:
            raise AssertionError('Invalid value for parameter "dimensions". '
                                 f'Expected "dimensions > 0"; received "{dimensions}"')

        # Sanity check: "sizes" must not be an empty list.
        if not sizes:
            raise AssertionError('Invalid value for parameter "sizes". '
                                 f'Expected "non-empty list"; received "{sizes}"')

        # The following sanity checks will use Python's list comprehension for checking conditions over the "sizes" list
        # and returning non-empty lists on error with the offending elements.

        # Sanity check: "sizes" must only contain "int" type values.
        not_ints = [val for val in sizes if not isinstance(val, int)]
        if not_ints:
            raise AssertionError('Invalid value for parameter "sizes". '
                                 f'Expected "list of integer"; received "{not_ints}"')

        # Sanity check: "sizes" must only contain positive integer numbers, excluding zero.
        not_positives = [val for val in sizes if val < 1]
        if not_positives:
            raise AssertionError('Invalid value for parameter "sizes". '
                                 f'Expected "list of positive integer"; received "{not_positives}"')

        # At this point it is safe to say that both dimensions and sizes are valid.
        return self._create_array_rec(dimensions, sizes, initialization_value)

    def initialize_array(self, array: list):
        """Implements the array initialization.

        Implement this method when requiring a special array initialization.
        If an array is to be initialized with the same value for all of its elements, use the "initialization_value"
        parameter of the create_array() method instead.
        """
        raise NotImplementedError('Initialize array not implemented')

    def print_array_custom(self, array: list):
        """Prints the benchmark array using the same format as in Polybench/C.

        If a different format is to be used, this method can be overridden.
        """
        raise NotImplementedError('Custom array print not implemented')

    def print_array(self, array: list, native_style: bool = True, dump_message: str = ''):
        """
        Prints the benchmarked array.

        :param list array: the array to be printed.
        :param bool native_style: (optional; default = True) allows to switch between native Python list printing
        :param string dump_message: (optional; default = '') allows to set a custom message on begin and end dump
        (default) or a custom format defined in the print_array_custom() method.
        """
        self.print_message('==BEGIN DUMP_ARRAYS==\n')
        self.print_message(f'begin dump: {dump_message}')
        if native_style:
            print(array)
        else:
            self.print_array_custom(array)
        self.print_message(f'\nend   dump: {dump_message}\n')
        self.print_message('==END   DUMP_ARRAYS==\n')

    def print_message(self, *args, **kwargs):
        """
        Prints a user message into the configured output.
        This method also removes the newline from vanilla print(), so the user must use them manually.

        This method is inspired by: https://stackoverflow.com/a/14981125
        """
        print(*args, file=self._POLYBENCH_DUMP_TARGET, end='', **kwargs)

    def print_value(self, value: DATA_TYPE):
        """
        Prints a data value using the configured data formatter.

        :param value: the value to be printed.
        """
        self.print_message(self.DATA_PRINT_MODIFIER.format(value))

    def run(self, output=stderr):
        """Prepares the environment for running a kernel, executes it and shows the result.

        **DO NOT OVERRIDE THIS METHOD UNLESS YOU KNOWN WHAT YOU ARE DOING!**

        This method is akin to the main() function found in Polybench/C.
        Common tasks for the benchmarks are performed by this method such as:
            - Preparing instruments (control process priority, CPU scheduler [when available], etc.)
            - Performing timing operations
            - Print the benchmark's output (timing, kernel, etc.)
        """
        self._POLYBENCH_DUMP_TARGET = output
        self.run_benchmark()

    def run_benchmark(self):
        """Implements the kernel to be benchmarked.

        This method **MUST** be overridden by subclasses.

        This method should declare the data structures required for running the kernel in a similar manner as done in
        the main() function of Polybench/C.
        """
        raise NotImplementedError('Kernel not implemented')
