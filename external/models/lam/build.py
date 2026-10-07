#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

try:
    from fvcore.common.registry import Registry
except ModuleNotFoundError:
    class Registry:
        """Small bundled fallback for the only fvcore API used by LAM."""

        def __init__(self, name):
            self._name = name
            self._obj_map = {}

        def _do_register(self, name, obj):
            if name in self._obj_map:
                raise KeyError("An object named '{}' is already registered in {}".format(name, self._name))
            self._obj_map[name] = obj

        def register(self, obj=None, name=None):
            if obj is not None:
                self._do_register(name or obj.__name__, obj)
                return obj

            def decorator(item):
                self._do_register(name or item.__name__, item)
                return item

            return decorator

        def get(self, name):
            return self._obj_map[name]

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

def build_model(args):
    model_name = args.model
    model = MODEL_REGISTRY.get(model_name)(args)
    return model
