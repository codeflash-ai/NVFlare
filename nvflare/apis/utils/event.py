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
import uuid
from typing import List

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.security.logging import secure_format_exception

# do not use underscore as key name; otherwise it cannot be removed from ctx
_KEY_EVENT_DEPTH = "###event_depth"
_MAX_EVENT_DEPTH = 20


def fire_event_to_components(event: str, components: List[FLComponent], ctx: FLContext):
    """Fires the specified event and invokes the list of handlers.

    Args:
        event: the event to be fired
        components: components to be invoked
        ctx: context for cross-component data sharing

    Returns: N/A

    """
    if not components:
        return

    event_id = str(uuid.uuid4())
    event_data = ctx.get_prop(FLContextKey.EVENT_DATA, None)
    event_origin = ctx.get_prop(FLContextKey.EVENT_ORIGIN, None)
    event_scope = ctx.get_prop(FLContextKey.EVENT_SCOPE, EventScope.LOCAL)

    depth = ctx.get_prop(_KEY_EVENT_DEPTH, 0)
    if depth > _MAX_EVENT_DEPTH:
        # too many recursive event calls
        raise RuntimeError("Recursive event calls too deep (>{})".format(_MAX_EVENT_DEPTH))

    ctx.set_prop(key=_KEY_EVENT_DEPTH, value=depth + 1, private=True, sticky=False)

    set_prop_args = [
        (FLContextKey.EVENT_ID, event_id),
        (FLContextKey.EVENT_DATA, event_data),
        (FLContextKey.EVENT_ORIGIN, event_origin),
        (FLContextKey.EVENT_SCOPE, event_scope)
    ]

    for h in components:
        if not isinstance(h, FLComponent):
            raise TypeError(f"handler must be FLComponent but got {type(h)}")
        try:
            # Only set these props once per-handler (as in original logic)
            for k, v in set_prop_args:
                ctx.set_prop(key=k, value=v, private=True, sticky=False)

            event_table = h.get_event_handlers()
            if event_table:
                entries = event_table.get(event)
                if entries:
                    for cb, kwargs in entries:
                        cb(event, ctx, **kwargs)
                else:
                    # no CB explicitly for this event - call the default handler.
                    h.handle_event(event, ctx)
            else:
                # no explicitly defined CBs - call the default handler.
                h.handle_event(event, ctx)
        except Exception as e:
            h.log_exception(
                ctx, f'Exception when handling event "{event}": {secure_format_exception(e)}', fire_event=False
            )
            exceptions = ctx.get_prop(FLContextKey.EXCEPTIONS)
            if exceptions is None:
                exceptions = {}
                ctx.set_prop(FLContextKey.EXCEPTIONS, exceptions, sticky=False, private=True)
            exceptions[h.name] = e

    ctx.set_prop(key=_KEY_EVENT_DEPTH, value=depth, private=True, sticky=False)
