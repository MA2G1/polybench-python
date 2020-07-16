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

"""<replace_with_module_description>"""

from benchmarks.polybench import PolyBench
from benchmarks.polybench_classes import PolyBenchParameters
from benchmarks.polybench_options import ArrayImplementation
from numpy.core.multiarray import ndarray


class TemplateClass(PolyBench):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        implementation = options['array_implementation']
        if implementation == ArrayImplementation.LIST:
            return _TemplateClassList.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.LIST_FLATTENED:
            return _TemplateClassListFlattened.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.NUMPY:
            return _TemplateClassNumPy.__new__(cls, options, parameters)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options)

        # Validate inputs
        if not isinstance(parameters, PolyBenchParameters):
            raise AssertionError(f'Invalid parameter "parameters": "{parameters}"')

        # The parameters hold the necessary information obtained from "polybench.spec" file
        params = parameters.DataSets.get(self.DATASET_SIZE)
        if not isinstance(params, dict):
            raise NotImplementedError(f'Dataset size "{self.DATASET_SIZE.name}" not implemented '
                                      f'for {parameters.Category}/{parameters.Name}.')

        # Adjust the data type and print modifier according to the data type
        self.DATA_TYPE = parameters.DataType
        self.set_print_modifier(parameters.DataType)

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.M = params.get('M')
        self.N = params.get('N')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        data = self.create_array(2, [self.N, self.M], self.DATA_TYPE(0))  # custom initialization value
        output = self.create_array(2, [self.M, self.M])

        # Initialize data structures
        self.initialize_array(data)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(data, output)

        # Stop and print instruments
        self.stop_instruments()

        # Return printable data as a list of tuples ('name', value).
        # Each tuple element must have the following format:
        #   (A: str, B: matrix)
        #     - A: a representative name for the data (this string will be printed out)
        #     - B: the actual data structure holding the computed result
        #
        # The syntax for the return statement would then be:
        #   - For single data structure results:
        #     return [('data_name', data)]
        #   - For multiple data structure results:
        #     return [('matrix1', m1), ('matrix2', m2), ... ]
        return [('results', output)]


class _TemplateClassList(TemplateClass):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TemplateClassList)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, array: list):
        for i in range(0, self.N):
            for j in range(0, self.M):
                array[i][j] = 42

    def print_array_custom(self, array: list, name: str):
        if name == 'array1':
            loop_bound = self.M
        else:
            loop_bound = self.N

        for i in range(0, loop_bound):
            for j in range(0, self.M):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[i][j])

    def kernel(self, input_data: list, output_data: list):
        """The actual kernel implementation.

        Modify this method's signature according to the kernel's needs.
        """
        print(f'NOT IMPLEMENTED: Template kernel for {self.__module__}')

class _TemplateClassListFlattened(TemplateClass):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TemplateClassListFlattened)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, array: list):
        for i in range(0, self.N):
            for j in range(0, self.M):
                array[self.M * i + j] = 42

    def print_array_custom(self, array: list, name: str):
        if name == 'array1':
            loop_bound = self.M
        else:
            loop_bound = self.N

        for i in range(0, loop_bound):
            for j in range(0, self.M):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[self.M * i + j])

    def kernel(self, input_data: list, output_data: list):
        """The actual kernel implementation.

        Modify this method's signature according to the kernel's needs.
        """
        print(f'NOT IMPLEMENTED: Template kernel for {self.__module__}')

class _TemplateClassNumPy(TemplateClass):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TemplateClassNumPy)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, array: list):
        for i in range(0, self.N):
            for j in range(0, self.M):
                array[i, j] = 42

    def print_array_custom(self, array: list, name: str):
        if name == 'array1':
            loop_bound = self.M
        else:
            loop_bound = self.N

        for i in range(0, loop_bound):
            for j in range(0, self.M):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[i, j])

    def kernel(self, input_data: ndarray, output_data: ndarray):
        """The actual kernel implementation.

        Modify this method's signature according to the kernel's needs.
        """
        print(f'NOT IMPLEMENTED: Template kernel for {self.__module__}')
