# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import re
from functools import wraps

type_pattern_mapping = {
    "server": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "host_name": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "overseer": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "sp_end_point": r"^((([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]):[0-9]*:[0-9]*)$",
    "client": r"^[A-Za-z0-9-_]+$",
    "relay": r"^[A-Za-z0-9-_]+$",
    "admin": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "email": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "org": r"^[A-Za-z0-9_]+$",
    "simple_name": r"^[A-Za-z0-9_]+$",
}


def name_check(name: str, entity_type: str):
    regex_pattern = type_pattern_mapping.get(entity_type)
    if regex_pattern is None:
        return True, "entity_type={} not defined, unable to check name={}.".format(entity_type, name)
    if re.match(regex_pattern, name):
        return False, "name={} passed on regex_pattern={} check".format(name, regex_pattern)
    else:
        return True, "name={} is ill-formatted for entity_type={} based on regex_pattern={}".format(
            name, entity_type, regex_pattern
        )


def validate_class_methods_args(cls):
    # Pre-fetch all functions once to avoid repeated introspection in setattr
    members = inspect.getmembers(cls, inspect.isfunction)
    for name, method in members:
        if name != "__init_subclass__":
            # Avoid wrapping methods multiple times (idempotence)
            if not hasattr(method, "__validated__"):
                wrapper = validate_args(method)
                # Mark the wrapped method so we don't wrap repeatedly
                wrapper.__validated__ = True
                setattr(cls, name, wrapper)
    return cls


def validate_args(method):
    # Cache the signature and parameter annotation mapping for fast access
    signature = inspect.signature(method)
    parameters = signature.parameters

    # Precompute empty value for fast lookup in runtime
    empty = inspect.Signature.empty

    @wraps(method)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind(*args, **kwargs)
        # Localize variables for faster loop
        bound_items = bound_arguments.arguments.items()
        for name, value in bound_items:
            annotation = parameters[name].annotation
            if not (annotation is empty or isinstance(value, annotation)):
                raise TypeError(
                    "argument '{}' of {} must be {} but got {}".format(name, method, annotation, type(value))
                )
        return method(*args, **kwargs)

    return wrapper
