from pathlib import Path
import os
import shutil
import sys


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("python3.12 evaluator runtime is required")
    os.execv(python, [python, __file__])

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import click  # noqa: E402


ctx = click.Context(click.Command("cli"))

optional_choice = click.Argument(
    ["method"],
    type=click.Choice(["foo", "bar", "baz"]),
    required=False,
)
assert optional_choice.make_metavar(ctx) == "[foo|bar|baz]", (
    "optional choice metavar must not be double bracketed"
)

optional_datetime = click.Argument(
    ["when"],
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=False,
)
assert optional_datetime.make_metavar(ctx) == "[%Y-%m-%d]", (
    "optional datetime metavar must not be double bracketed"
)

required_choice = click.Argument(
    ["method"],
    type=click.Choice(["foo", "bar"]),
    required=True,
)
assert required_choice.make_metavar(ctx) == "{foo|bar}", (
    "required choice metavar semantics changed"
)

deprecated_repeated_choice = click.Argument(
    ["method"],
    type=click.Choice(["foo", "bar"]),
    required=False,
    deprecated=True,
    nargs=2,
)
assert deprecated_repeated_choice.make_metavar(ctx) == "[foo|bar]!...", (
    "deprecated or repeated metavar semantics changed"
)
