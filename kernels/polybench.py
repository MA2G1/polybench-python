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
        __init__()
        kernel()
        print_array_custom()
    """

    def __init__(self):
        """Class constructor.

        Since this is an abstract class, this method prevents its instantiation by throwing a RuntimeError.
        This method MUST be overridden by subclasses.
        """
        raise RuntimeError('Abstract classes cannot be instantiated.')

    def _create_array_rec(self, dimensions: int, sizes: list, initialization_value: int = 0) -> list:
        """Auxiliary recursive method for creating a new array.

        This method assumes that the parameters were previously validated (in the create_array method)
        """
        if dimensions == 1:
            # Just create a list with as many zeros as specified in sizes[0]
            return [initialization_value for x in range(sizes[0])]

        if len(sizes) == 1:
            # Generate lists of the same size per dimension
            return [self.create_array(dimensions - 1, sizes, initialization_value) for x in range(sizes[0])]
        else:
            # Generate lists with unique sizes per dimension
            return [self.create_array(dimensions - 1, sizes[1:], initialization_value) for x in range(sizes[0])]

    def create_array(self, dimensions: int, sizes: list, initialization_value: int = 0) -> list:
        """Create a new array with a specified size.

        Parameters:
            dimensions: specifies the number of dimensions of the array.
            sizes: allows to specify the number of elements for each dimension. If this parameter is a list with one
                element, this element represents the size of all dimensions, otherwise the list must have as many
                elements as specified by the dimensions parameter and each element represents the size of a dimension.
                The size of the first dimension is specified by the first element on the list, the size of the second
                dimension is represented by the second element of the list and so on.
            initialization_value: the value at which all array elements are set. Defaults to zero (0).
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
        parameter of the create_array method instead.
        """
        raise NotImplementedError('Initialize array not implemented')

    def print_array_custom(self, array: list):
        """Prints the benchmark array using the same format as in Polybench/C.

        If a different format is to be used, this method can be overridden.
        """
        raise NotImplementedError('Custom array print not implemented')

    def print_array(self, array: list, native_style: bool = True):
        """Prints the benchmark array.

        This method allows to print the array (actually a Python list) either as a Python array when native_style is
         True (default) or using a format common with Polybench/C.
         The alternative format can be overridden by reimplementing the method print_array_custom().
        """
        print('===BEGIN DUMP_ARRAYS===')
        if native_style:
            print(array)
        else:
            self.print_array_custom(array)
        print('===END   DUMP_ARRAYS===')

    def run(self):
        """Prepares the environment for running a kernel, executes it and shows the result.

        DO NOT OVERRIDE THIS METHOD UNLESS YOU KNOWN WHAT YOU ARE DOING!

        This method is akin to the main() function found in Polybench/C.
        Common tasks for the benchmarks are performed by this method such as:
            - Preparing instruments (control process priority, CPU scheduler [when available], etc.)
            - Performing timing operations
            - Print the benchmark's output (timing, kernel, etc.)
        """
        self.run_benchmark()

    def run_benchmark(self):
        """Implements the kernel to be benchmarked.

        This method MUST be overridden by subclasses.

        This method should declare the data structures required for running the kernel in a similar manner as done in
        the main() function of Polybench/C.
        """
        raise NotImplementedError('Kernel not implemented')
