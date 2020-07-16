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

from benchmarks.polybench_options import DataSetSize


class PolyBenchParameters:
    """Stores all parameters required for a benchmark, obtained from a spec file.
    """

    def __init__(self, parameters: dict):
        """Process the parameters dictionary and store its values on public class fields."""
        self.Name = parameters['kernel']
        self.Category = parameters['category']

        if parameters['datatype'] == 'float' or parameters['datatype'] == 'double':
            self.DataType = float
        else:
            self.DataType = int

        mini_dict = {}
        small_dict = {}
        medium_dict = {}
        large_dict = {}
        extra_large_dict = {}
        for i in range(0, len(parameters['params'])):
            mini_dict[parameters['params'][i]] = parameters['MINI'][i]
            small_dict[parameters['params'][i]] = parameters['SMALL'][i]
            medium_dict[parameters['params'][i]] = parameters['MEDIUM'][i]
            large_dict[parameters['params'][i]] = parameters['LARGE'][i]
            extra_large_dict[parameters['params'][i]] = parameters['EXTRALARGE'][i]

        self.DataSets = {
            DataSetSize.MINI: mini_dict,
            DataSetSize.SMALL: small_dict,
            DataSetSize.MEDIUM: medium_dict,
            DataSetSize.LARGE: large_dict,
            DataSetSize.EXTRA_LARGE: extra_large_dict
        }

