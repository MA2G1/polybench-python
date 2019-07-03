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


from setuptools import find_packages

# Find packages relative to this file's location in the hierarchy.
_available_packages = find_packages(__name__)

# Separate intermediate packages (categories) from kernel implementations.
# Note: we can consider all available packages as categories and then the innermost ones are the actual implementations.
kernel_categories = set()
kernel_implementations = set()

# Compare the permutations of the package names for searching unique packages
for candidate in _available_packages:
    # Perform a small optimization: skip checking candidates which are already categorized
    if candidate in kernel_categories:
        continue

    is_category = False
    for package in _available_packages:
        # Test whether a package name is a category or not. Since we are iterating over the full set, it will happen
        # that a package will be compared against itself.
        # This implementation is not very efficient, but works.
        if candidate != package and candidate in package:
            # We possibly found a category...
            # At this point, the two strings match something like ["cat1.subcat1" in "cat1.subcat1xx"]. The problem is
            # that it may not be a category, but rather a similar name. Check if it has a dot on its name or not.
            candidate_len = len(candidate)
            package_len = len(package)
            if package_len > candidate_len:
                # When we have a dot (.) as the next character in the package name, we know the candidate is part of the
                # subpackage's name, effectively making a category.
                if package[candidate_len] == '.':
                    is_category = True
            else:
                is_category = True

    if is_category:
        kernel_categories.add(candidate)
    else:
        kernel_implementations.add(candidate)


# Now we should have the categories and implementations.
# Perform a sort operation for better user readability.
kernel_categories = sorted(kernel_categories)
kernel_implementations = sorted(kernel_implementations)
