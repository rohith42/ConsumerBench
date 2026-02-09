from typing import Any


def add(a: int, b: int) -> int:
    return a + b


def concat(a: str, b: str, sep: str = " ") -> str:
    return f"{a}{sep}{b}"


class EchoTool:
    def __call__(self, text: str) -> str:
        return text


class UpperTool:
    def forward(self, text: str) -> str:
        return text.upper()


def passthrough(value: Any) -> Any:
    return value
