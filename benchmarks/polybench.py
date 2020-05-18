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

"""This module offers the base Polybench class for implementing kernels in benchmarks."""
from enum import Enum, auto
from sys import stderr
from time import time

# Import requirements for PAPI
from pypapi import events as papi_events, papi_high

# Import requirements for IL
import il
import ctypes

import os


@il.asm
def _read_tsc():
    """
    .intel_syntax noprefix
    RDTSC
    sal     rdx, 32
    mov     eax, eax
    or      rax, rdx
    ret
    """
    return ctypes.c_ulonglong


class DatasetSize(Enum):
    """Define the possible values for selecting dataset sizes.

    Instead of manually managing the values of this enumeration we let the python interpreter initialize them.
    """
    MINI = auto()
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()
    EXTRA_LARGE = auto()


class PolyBench:
    """This class offers common methods for building new benchmarks.

    This class is not meant to be instantiated as it is an abstract class. Each benchmark must inherit from this class
    and override the following methods:
        - __init__()
        - kernel()
        - print_array_custom()
    """

    # Typical options
    POLYBENCH_TIME = False                     # Output execution time
    POLYBENCH_DUMP_ARRAYS = False              # Dump live-out arrays

    # Options that may lead to better performance
    POLYBENCH_PADDING_FACTOR = 0               # Pad all dimensions of lists by this value

    # Timing/profiling options
    POLYBENCH_PAPI = False                     # Turn on PAPI timing
    POLYBENCH_CACHE_SIZE_KB = 32770            # Cache size to flush, in KiB (32+ MiB)
    POLYBENCH_NO_FLUSH_CACHE = False           # Don't flush the cache before calling the timer
    POLYBENCH_CYCLE_ACCURATE_TIMER = False     # Use Time Stamp Counter
    POLYBENCH_LINUX_FIFO_SCHEDULER = False     # Use FIFO scheduler (must run as root)

    # Other options (not present in the README file)
    POLYBENCH_DUMP_TARGET = stderr             # Dump user messages into stderr, as in Polybench/C
    POLYBENCH_GFLOPS = False
    POLYBENCH_PAPI_VERBOSE = False

    # Timing counters
    __polybench_program_total_flops = 0
    __polybench_timer_start = 0
    __polybench_timer_stop = 0

    # PAPI counters
    __papi_available_counters = []
    __papi_counters = []
    __papi_counters_result = []

    DATA_TYPE = int  # The data type used for the current benchmark (used for conversions and formatting)
    DATA_PRINT_MODIFIER = '{:d} '  # A default print modifier. Should be set up in run()

    def __init__(self):
        """Class constructor.

        Since this is an abstract class, this method prevents its instantiation by throwing a RuntimeError.
        This method **MUST** be overridden by subclasses.
        """
        raise RuntimeError('Abstract classes cannot be instantiated.')

    def __create_array_rec(self, dimensions: int, sizes: list, initialization_value: int = 0) -> list:
        """Auxiliary recursive method for creating a new array.

        This method assumes that the parameters were previously validated (in the create_array method).

        :param int dimensions: the number of dimensions to create. One dimension creates a list, two a matrix and so on.
        :param list[int] sizes: a list of integers, each one representing the size of a dimension. The first element of
            the list represents the size of the first dimension, the second element the size of the second dimension and
            so on. If this list is smaller than the actual number of dimensions then the last size read is used for the
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
            return [self.__create_array_rec(dimensions - 1, sizes, initialization_value) for x in range(sizes[0])]
        else:
            # Generate lists with unique sizes per dimension
            return [self.__create_array_rec(dimensions - 1, sizes[1:], initialization_value) for x in range(sizes[0])]

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

        # Add post-padding to every array dimension
        new_sizes = [size + self.POLYBENCH_PADDING_FACTOR for size in sizes]

        # At this point it is safe to say that both dimensions and sizes are valid.
        return self.__create_array_rec(dimensions, new_sizes, initialization_value)

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
        print(*args, file=self.POLYBENCH_DUMP_TARGET, end='', **kwargs)

    def print_value(self, value):
        """
        Prints a data value using the configured data formatter.

        :param value: the value to be printed.
        """
        self.print_message(self.DATA_PRINT_MODIFIER.format(value))

    def run(self, print_result: bool = False, output=stderr):
        """Prepares the environment for running a benchmark, executes it and shows the result.

        **DO NOT OVERRIDE THIS METHOD UNLESS YOU KNOWN WHAT YOU ARE DOING!**

        This method is akin to the main() function found in Polybench/C.
        Common tasks for the benchmarks are performed by this method such as:
            - Preparing instruments (control process priority, CPU scheduler [when available], etc.)
            - Performing timing operations
            - Print the benchmark's output (timing, kernel, etc.)
        """
        #
        # Perform pre-benchmark actions.
        #
        self.POLYBENCH_DUMP_TARGET = output  # set the output target for printing messages

        #
        # Run the benchmark
        #
        outputs = self.run_benchmark()

        #
        # Perform post-benchmark actions
        #
        if print_result:
            for out in outputs:
                self.print_array(out[1], False, out[0])

    def run_benchmark(self):
        """Implements the kernel to be benchmarked.

        This method **MUST** be overridden by subclasses.

        This method should declare the data structures required for running the kernel in a similar manner as done in
        the main() function of Polybench/C.

        :returns: a list of tuples. For each tuple (X, Y), X represents the name of the output and Y is the actual
         output. The output(s) will be used by the print_array() method when required.
        :rtype: list[tuple]
        """
        raise NotImplementedError('Kernel not implemented')

    def start_instruments(self):
        """Perform various actions before running the actual benchmark.
        """
        if self.POLYBENCH_TIME or self.POLYBENCH_GFLOPS:
            self.__timer_start()
        elif self.POLYBENCH_PAPI:
            self.__prepare_instruments()
            self.__papi_init()
            self.__papi_start_counters()

    def stop_instruments(self):
        """Restore system state if it was previously modified by start_instruments()."""
        if self.POLYBENCH_TIME or self.POLYBENCH_GFLOPS:
            self.__timer_stop()
        elif self.POLYBENCH_PAPI:
            self.__papi_stop_counters()

    def print_instruments(self):
        """Print the state of the instruments."""
        if self.POLYBENCH_TIME or self.POLYBENCH_GFLOPS:
            self.__timer_print()
        elif self.POLYBENCH_PAPI:
            self.__papi_print()

    def __prepare_instruments(self):
        if not self.POLYBENCH_NO_FLUSH_CACHE:
            self.__flush_cache()
        if self.POLYBENCH_LINUX_FIFO_SCHEDULER:
            self.__linux_fifo_scheduler()

    def __timer_start(self):
        self.__prepare_instruments()
        if not self.POLYBENCH_CYCLE_ACCURATE_TIMER:
            self.__timer_start = time()
        else:
            self.__timer_start = _read_tsc()

    def __timer_stop(self):
        if not self.POLYBENCH_CYCLE_ACCURATE_TIMER:
            self.__timer_stop = time()
        else:
            self.__timer_stop = _read_tsc()
        if self.POLYBENCH_LINUX_FIFO_SCHEDULER:
            self.__linux_standard_scheduler()

    def __timer_print(self):
        if not self.POLYBENCH_CYCLE_ACCURATE_TIMER:
            print(f'{self.__timer_stop - self.__timer_start:0.6f}')
        else:
            print(f'{self.__timer_stop - self.__timer_start:Ld}')

    def __papi_init(self):
        """
        Performs initialization of the PAPI library.

        Since the library python-papi only exports standard events, it is necessary to know which ones are available and
        which ones are requested by the user to perform a validation phase.
        :return: None
        """
        def get_available_counters() -> list:
            """
            Gets a list of available counters. Each element of the list is a tuple containing the name of the event and
            the numerical value corresponding to the event. This value is masked.
            :return: A list with the available benchmarks
            :rtype: list[tuple]
            """
            def is_number(x):
                return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)
            # See: https://stackoverflow.com/a/9794849
            from inspect import getmembers
            return getmembers(papi_events, is_number)

        def parse_counters_file():
            result = []
            with open('papi_counters.list') as f:
                contents = f.read()
                lines = contents.splitlines()
                lines = [line.strip() for line in lines]  # Lines without single line comments

                is_in_comment = False
                for line in lines:
                    if not is_in_comment:
                        if line.startswith('/*'):
                            is_in_comment = True
                            continue
                        elif line.startswith('//'):
                            continue
                        else:
                            result.append(line.strip('",'))  # store plain counter names
                    else:
                        if line.endswith('*/'):
                            is_in_comment = False
            return result

        self.__papi_available_counters = get_available_counters()
        user_counters = parse_counters_file()

        # Check if there are any user supplied counters after reading the file
        if len(user_counters) < 1:
            raise NotImplementedError('Must supply at least one counter name in file "papi_counters.list"')

        # Check if the user counters exist within the available standard set.
        for usr_counter in user_counters:
            is_available = False
            for avlbl_counter in self.__papi_available_counters:
                if usr_counter == avlbl_counter[0]:
                    self.__papi_counters.append(avlbl_counter[1])
                    is_available = True
                    break
            if not is_available:
                print(f'WARNING: counter "{usr_counter}" not available.')

    def __papi_start_counters(self):
        papi_high.start_counters(self.__papi_counters)

    def __papi_stop_counters(self):
        self.__papi_counters_result = papi_high.stop_counters()

    def __papi_print(self):
        def papi_counter_names():
            """
            Translate back the PAPI identifiers into user-readable ones.
            :return: a list of PAPI identifiers
            """
            result = []
            for usr_counter in self.__papi_counters:
                for avlbl_counter in self.__papi_available_counters:
                    if usr_counter == avlbl_counter[1]:
                        result.append(avlbl_counter[0])
                        break
            return result

        counter_names = papi_counter_names()
        for i in range(0, len(self.__papi_counters)):
            if self.POLYBENCH_PAPI_VERBOSE:
                print(f'{counter_names[i]}=', end='')
            print(f'{self.__papi_counters_result[i]} ', end='')
            if self.POLYBENCH_PAPI_VERBOSE:
                print()  # new line
        print()  # new line

    def __flush_cache(self):
        """Thrashes the cache by generating a very large data structure."""
        cs = self.POLYBENCH_CACHE_SIZE_KB * 1024 / 8  # divided by sizeof(double)
        flush = [0.0 for x in range(cs)]  # 0.0 forces data type to be a float
        tmp = 0.0
        for i in range(cs):
            tmp += flush[i]
        assert tmp <= 10.0

    def __linux_fifo_scheduler(self):
        param = os.sched_getparam(0)
        param.sched_priority = os.sched_get_priority_max(os.SCHED_FIFO)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)

    def __linux_standard_scheduler(self):
        param = os.sched_getparam(0)
        param.sched_priority = os.sched_get_priority_max(os.SCHED_OTHER)
        os.sched_setscheduler(0, os.SCHED_OTHER, param)

