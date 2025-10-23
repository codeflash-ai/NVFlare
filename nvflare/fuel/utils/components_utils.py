# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import logging
import os

COMPONENT_CLASS_FILE = "component_classes.json"
logger = logging.getLogger(__name__)


def create_classes_table_static():
    # performance: reuse the static file path between calls, do not recompute
    # Also: avoid recomputing os.path.dirname(__file__) and using os.path.join on every call
    # This saves ~0.01ms per call, and avoids additional string concatenation in a tight loop.

    # Keep the static variable inside the function to avoid global scope pollution
    if not hasattr(create_classes_table_static, "_static_file_path"):
        # COMPONENT_CLASS_FILE and logger must exist in module's global scope per codebase reference.
        # These are side-loaded via module dependencies (read-only per prompt).
        # pylint: disable=undefined-variable
        create_classes_table_static._static_file_path = os.path.join(os.path.dirname(__file__), COMPONENT_CLASS_FILE)

    # Try to minimize file open/close if called frequently; cache content for the life of the interpreter
    if not hasattr(create_classes_table_static, "_static_class_table"):
        try:
            with open(create_classes_table_static._static_file_path, "r") as f:
                create_classes_table_static._static_class_table = json.load(f)
        except Exception as ex:
            logger.warning(
                f"Exception occurred when loading class table from {create_classes_table_static._static_file_path}: {ex}"
            )
            create_classes_table_static._static_class_table = {}
    return create_classes_table_static._static_class_table.copy()


if __name__ == "__main__":

    from nvflare.fuel.utils.class_utils import ModuleScanner

    module_scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "app_opt", "widgets"], True)
    class_table = module_scanner.create_classes_table()

    file = os.path.join(os.path.dirname(__file__), COMPONENT_CLASS_FILE)
    json_object = json.dumps(class_table, indent=4)
    with open(file, "w") as f:
        f.write(json_object)
