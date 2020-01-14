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

"""Implements the correlation kernel in a Polybench class."""

from kernels.polybench import Polybench
from kernels.polybench import DatasetSize


class Correlation(Polybench):

    def __init__(self, dataset_size: DatasetSize = DatasetSize.LARGE):
        if not isinstance(dataset_size, DatasetSize):
            raise AssertionError(f'Invalid parameter "dataset_size": "{dataset_size}"')

        values = {
            DatasetSize.MINI:           {'M': 28,   'N': 32},
            DatasetSize.SMALL:          {'M': 80,   'N': 100},
            DatasetSize.MEDIUM:         {'M': 240,  'N': 260},
            DatasetSize.LARGE:          {'M': 1200, 'N': 1400},
            DatasetSize.EXTRA_LARGE:    {'M': 2600, 'N': 3000}
        }
        parameters = values.get(dataset_size)
        if not isinstance(parameters, dict):
            # Could not find a valid dataset size
            raise NotImplementedError(f'Dataset size "{dataset_size.name}" not implemented.')

        # Set up problem size
        self.M = parameters.get('M')
        self.N = parameters.get('N')

    def initialize_array(self, array: list):
        pass

    def print_array_custom(self, array: list):
        pass

    def run_benchmark(self):
        print('Correlation')
