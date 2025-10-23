# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.edge.web.models.base_model import BaseModel, EdgeProtoKey
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class TaskRequest(BaseModel):
    def __init__(self, device_info: DeviceInfo, user_info: UserInfo, job_id: str, cookie: dict, **kwargs):
        super().__init__()
        self.device_info = device_info
        self.user_info = user_info
        self.job_id = job_id
        self.cookie = cookie

        if kwargs:
            self.update(kwargs)

    @classmethod
    def validate(cls, d: dict) -> str:
        return cls.check_keys(d, [EdgeProtoKey.JOB_ID, EdgeProtoKey.DEVICE_INFO])

    @classmethod
    def from_dict(cls, d: dict):
        error = cls.validate(d)
        if error:
            return error, None

        # Use local variables to temporarily cache popped values
        device_info_dict = d.pop(EdgeProtoKey.DEVICE_INFO, None)
        if not device_info_dict:
            return "missing device_info", None

        device_id = device_info_dict.pop(EdgeProtoKey.DEVICE_ID, None)
        if not device_id:
            return "missing device_id", None

        device_info = DeviceInfo(device_id)
        device_info.update(device_info_dict)

        user_info_dict = d.pop(EdgeProtoKey.USER_INFO, None)
        if user_info_dict:
            user_info = UserInfo()
            user_info.update(user_info_dict)
        else:
            user_info = None

        job_id = d.pop(EdgeProtoKey.JOB_ID)
        cookie = d.pop(EdgeProtoKey.COOKIE, {})

        # Update all remaining keys in d to the task_req (only once, after pops)
        task_req = TaskRequest(device_info, user_info, job_id, cookie)
        if d:
            task_req.update(d)
        return "", task_req
