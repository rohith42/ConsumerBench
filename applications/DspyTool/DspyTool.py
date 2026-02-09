import importlib
import inspect
import os
import sys
from typing import Any, Callable, Dict, List

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application


class DspyTool(Application):
    """
    Generic application to invoke a tool by import string.
    The tool can be a function, a callable object, or an object
    that exposes a `forward(...)` method (common in DSPy tools).
    """

    def __init__(self):
        super().__init__()
        self.tool = None

    def _resolve_tool(self, tool_string: str, init_args: List[Any], init_kwargs: Dict[str, Any]):
        if not tool_string:
            raise ValueError("tool_string must be set to import a tool.")

        if ":" in tool_string:
            module_path, attr_path = tool_string.split(":", 1)
        else:
            parts = tool_string.split(".")
            if len(parts) < 2:
                raise ValueError(
                    "tool_string must be in the form 'module.attr' or 'module:attr'."
                )
            module_path, attr_path = ".".join(parts[:-1]), parts[-1]

        module = importlib.import_module(module_path)
        tool = module
        for attr in attr_path.split("."):
            tool = getattr(tool, attr)

        if inspect.isclass(tool):
            tool = tool(*init_args, **init_kwargs)
        elif init_args or init_kwargs:
            raise ValueError("tool_init_args/kwargs are only valid for class tools.")

        return tool

    def _call_tool(self, tool: Any, tool_args: List[Any], tool_kwargs: Dict[str, Any]):
        if callable(tool):
            return tool(*tool_args, **tool_kwargs)
        if hasattr(tool, "forward") and callable(tool.forward):
            return tool.forward(*tool_args, **tool_kwargs)
        raise TypeError("Resolved tool is not callable and has no forward(...) method.")

    def run_setup(self, *args, **kwargs):
        print("DspyTool setup")
        tool_string = kwargs.get("tool_string", self.config["tool_string"])
        init_args = kwargs.get("tool_init_args", self.config["tool_init_args"])
        init_kwargs = kwargs.get("tool_init_kwargs", self.config["tool_init_kwargs"])

        self.tool = self._resolve_tool(tool_string, init_args, init_kwargs)
        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("DspyTool cleanup")
        self.tool = None
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print("DspyTool application")
        if self.tool is None:
            raise RuntimeError("DspyTool is not set up.")

        tool_args = kwargs.get("tool_args", self.config["tool_args"])
        tool_kwargs = kwargs.get("tool_kwargs", self.config["tool_kwargs"])
        result = self._call_tool(self.tool, tool_args, tool_kwargs)
        return {"status": "tool_complete", "result": result}

    def load_dataset(self, *args, **kwargs):
        print("DspyTool loading dataset")
        return {"status": "dataset_loaded"}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "tool_string": "applications.DspyTool.example_tools.add",
            "tool_init_args": [],
            "tool_init_kwargs": {},
            "tool_args": [],
            "tool_kwargs": {"a": 1, "b": 2},
        }
