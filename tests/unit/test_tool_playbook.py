"""Registry alignment and size cap for ``tools.tool_playbook``."""

from tools.tool_playbook import (
    playbook_line,
    render_compact_playbook,
    validate_playbook_complete,
)
from tools.tools_executor import get_valid_actions


def test_playbook_covers_registry():
    validate_playbook_complete()
    for name in get_valid_actions():
        line = playbook_line(name)
        assert isinstance(line, str) and len(line) >= 8


def test_render_within_cap():
    text = render_compact_playbook(4000)
    assert len(text) <= 4000
    assert "TOOL PLAYBOOK" in text
