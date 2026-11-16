"""Base configuration class with recursive member-class instantiation (mirrors HIMLoco legacy)."""

import inspect


class BaseConfig:
    def __init__(self) -> None:
        """Initialize all nested config classes recursively."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj) -> None:
        for key in dir(obj):
            if key == "__class__":
                continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)
