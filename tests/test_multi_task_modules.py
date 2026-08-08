# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task module may define more than one Task, and every one of them registers.

Discovery used to key an environment off the FILE and take whichever Task class came first, so a
module defining several tasks registered one and dropped the rest -- with no error, since get_envs()
simply returned a shorter list. These cover the selection rule directly, and the registration path
end to end.
"""

import textwrap

import pytest

from robolab.core.task.task_utils import clear_task_cache, load_task_from_file

_TWO_TASKS = '''
from robolab.core.task.task import Task


class AlphaTask(Task):
    pass


class BetaTask(Task):
    pass
'''


@pytest.fixture
def two_task_module(tmp_path):
    path = tmp_path / "two_tasks.py"
    path.write_text(textwrap.dedent(_TWO_TASKS))
    clear_task_cache()
    yield str(path)
    clear_task_cache()


def test_every_task_in_a_module_is_returned(two_task_module):
    names = {cls.__name__ for cls in load_task_from_file(two_task_module, allow_multiple=True)}
    assert names == {"AlphaTask", "BetaTask"}


def test_a_task_can_be_selected_by_name(two_task_module):
    for name in ("AlphaTask", "BetaTask"):
        assert load_task_from_file(two_task_module, task_class_name=name).__name__ == name


def test_selecting_by_name_survives_the_module_cache(two_task_module):
    # The cache is keyed by path alone. Before the by-name selector it returned whatever the first
    # caller had asked for, so the second task in a module was unreachable once the first was loaded.
    assert load_task_from_file(two_task_module, task_class_name="AlphaTask").__name__ == "AlphaTask"
    assert load_task_from_file(two_task_module, task_class_name="BetaTask").__name__ == "BetaTask"


def test_an_unknown_name_names_what_the_module_does_define(two_task_module):
    with pytest.raises(ValueError, match="AlphaTask"):
        load_task_from_file(two_task_module, task_class_name="GammaTask")


def test_single_task_module_is_unchanged(tmp_path):
    # The behaviour 133 of the 140 shipped task files rely on: no name, one class, returned bare.
    path = tmp_path / "one_task.py"
    path.write_text(textwrap.dedent(_TWO_TASKS).replace("class BetaTask(Task):\n    pass\n", ""))
    clear_task_cache()
    assert load_task_from_file(str(path)).__name__ == "AlphaTask"
    assert [c.__name__ for c in load_task_from_file(str(path), allow_multiple=True)] == ["AlphaTask"]
