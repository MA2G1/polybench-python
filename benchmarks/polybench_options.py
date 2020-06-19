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
"""This module defines all of the available options for PolyBench."""

from sys import stderr


# Typical options
POLYBENCH_TIME = 'POLYBENCH_TIME'                # Output execution time
POLYBENCH_DUMP_ARRAYS = 'POLYBENCH_DUMP_ARRAYS'  # Dump live-out arrays

# Options that may lead to better performance
POLYBENCH_PADDING_FACTOR = 'POLYBENCH_PADDING_FACTOR'  # Pad all dimensions of lists by this value

# Timing/profiling options
POLYBENCH_PAPI = 'POLYBENCH_PAPI'                                  # Turn on PAPI timing
POLYBENCH_CACHE_SIZE_KB = 'POLYBENCH_CACHE_SIZE_KB'                # Cache size to flush, in KiB (32+ MiB)
POLYBENCH_NO_FLUSH_CACHE = 'POLYBENCH_NO_FLUSH_CACHE'              # Don't flush the cache before calling the timer
POLYBENCH_CYCLE_ACCURATE_TIMER = 'POLYBENCH_CYCLE_ACCURATE_TIMER'  # Use Time Stamp Counter
POLYBENCH_LINUX_FIFO_SCHEDULER = 'POLYBENCH_LINUX_FIFO_SCHEDULER'  # Use FIFO scheduler (must run as root)

# Other options (not present in the README file)
POLYBENCH_DUMP_TARGET = 'POLYBENCH_DUMP_TARGET'  # Dump user messages into stderr, as in Polybench/C
POLYBENCH_GFLOPS = 'POLYBENCH_GFLOPS'
POLYBENCH_PAPI_VERBOSE = 'POLYBENCH_PAPI_VERBOSE'


polybench_default_options = {
    POLYBENCH_TIME: False,
    POLYBENCH_DUMP_ARRAYS: False,
    # Options that may lead to better performance
    POLYBENCH_PADDING_FACTOR: 0,
    # Timing/profiling options
    POLYBENCH_PAPI: False,
    POLYBENCH_CACHE_SIZE_KB: 32770,
    POLYBENCH_NO_FLUSH_CACHE: False,
    POLYBENCH_CYCLE_ACCURATE_TIMER: False,
    POLYBENCH_LINUX_FIFO_SCHEDULER: False,
    # Other options (not present in the README file)
    POLYBENCH_DUMP_TARGET: stderr,
    POLYBENCH_GFLOPS: False,
    POLYBENCH_PAPI_VERBOSE: False,
}
