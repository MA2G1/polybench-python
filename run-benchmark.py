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

"""Polybench/Python is the reimplementation of the polyhedral benchmark Polybench/C in the Python programming language.

This module implements a main program which allows the user to run benchmarks easily without the burden of creating
makefiles or using configuration scripts for generating and using those.
This program allows to use only the Python runtime for everything the user should need when evaluating or implementing
the different kernels."""

from kernels import kernel_classes
from kernels.polybench import Polybench
import importlib
import sys

if __name__ == '__main__':
    print(kernel_classes)

    #for impl in kernel_classes:
    #    print(Polybench(impl).run())
    #    importlib.reload(impl)
