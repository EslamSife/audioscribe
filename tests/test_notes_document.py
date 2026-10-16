import re
from datetime import date

from transcribe.notes import document

TODAY = date(2026, 10, 16)
NOTE = "# Release Planning: Version 2\n\n## Summary\nWe pick a date — Friday."
VAULT_FILENAME = re.compile(r"^[a-z0-9][a-z0-9-]*\.md$")


def test_render_adds_all_seven_frontmatter_keys_and_related_section():
    content = document.render(NOTE, sources=["/tmp/standup.md"], today=TODAY)

    assert content.startswith(
        "---\n"
        "created: 2026-10-16\n"
        "updated: 2026-10-16\n"
        "type: distilled\n"
        "status: inbox\n"
        "tags: [meeting, transcript]\n"
        'sources: ["/tmp/standup.md"]\n'
        "related: []\n"
        "---\n"
        "# Release Planning"
    )
    assert content.endswith("\n\n## Related\n")


def test_render_replaces_em_dashes():
    content = document.render(NOTE, sources=[], today=TODAY)
    assert "—" not in content
    assert "We pick a date, Friday." in content


def test_filename_uses_title_and_date_and_matches_vault_rule():
    name = document.filename(NOTE, today=TODAY, fallback="live_session")
    assert name == "release-planning-version-2-2026-10-16.md"
    assert VAULT_FILENAME.match(name)


def test_filename_falls_back_for_arabic_titles():
    arabic = "# مراجعة الإصدار\n\nbody"
    assert document.filename(arabic, today=TODAY, fallback="Team_Sync") == "team-sync-2026-10-16.md"
    assert document.filename(arabic, today=TODAY, fallback="___") == "meeting-notes-2026-10-16.md"


def test_filename_keeps_long_titles_short_and_whole_words():
    long_title = "# " + " ".join(["integration"] * 20)
    name = document.filename(long_title, today=TODAY, fallback="x")
    slug = name.removesuffix("-2026-10-16.md")
    assert len(slug) <= document.MAX_SLUG_LENGTH
    assert slug.endswith("integration")


def test_render_tags_work_templates_but_not_the_general_one():
    assert "tags: [meeting, transcript, incident]\n" in document.render(
        NOTE, sources=[], today=TODAY, template="incident"
    )
    assert "tags: [meeting, transcript]\n" in document.render(NOTE, sources=[], today=TODAY)
