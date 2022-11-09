import sys

import click

from click.testing import CliRunner

from bob.io.base import testing_utils


@click.command("dummy")
def dummy_command_0():
    sys.exit(0)


@click.command("dummy_exit_1")
def dummy_command_1():
    sys.exit(1)


@click.command("dummy_exit_raise")
def dummy_command_raise():
    raise RuntimeError("Expected exception")


def test_assert_dummy():
    result = CliRunner().invoke(dummy_command_0)
    assert result.exit_code == 0
    testing_utils.assert_click_runner_result(result)

    result = CliRunner().invoke(dummy_command_1)
    assert result.exit_code == 1
    testing_utils.assert_click_runner_result(result, exit_code=1)

    result = CliRunner().invoke(dummy_command_raise)
    assert result.exit_code == 1
    testing_utils.assert_click_runner_result(
        result, exit_code=1, exception_type=RuntimeError
    )
