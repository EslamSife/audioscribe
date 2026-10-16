"""Wrap the model's note in the Obsidian vault format: frontmatter, file name, clean body.

The vault rules: lowercase kebab-case ASCII file names, seven frontmatter keys,
no em dashes, and a closing "## Related" section that link tools append to.
"""

import json
import re
import unicodedata
from datetime import date

# A note rebuilt from a source text (here: the transcript), in the vault's type list
NOTE_TYPE = "distilled"
TAGS = ("meeting", "transcript")
MAX_SLUG_LENGTH = 60
EM_DASH = "—"


def render(note: str, *, sources: list[str], today: date, template: str = "general") -> str:
    """Return the full file content: frontmatter, the note, and an empty Related section.

    A work template (e.g. 'standup') is added as a tag, so the vault can list those notes.
    """
    tags = TAGS if template == "general" else (*TAGS, template)
    frontmatter = [
        "---",
        f"created: {today.isoformat()}",
        f"updated: {today.isoformat()}",
        f"type: {NOTE_TYPE}",
        "status: inbox",
        f"tags: [{', '.join(tags)}]",
        f"sources: [{', '.join(json.dumps(s, ensure_ascii=False) for s in sources)}]",
        "related: []",
        "---",
        "",
    ]
    return "\n".join(frontmatter) + _clean(note) + "\n\n## Related\n"


def filename(note: str, *, today: date, fallback: str) -> str:
    """Name the file after the note's title, e.g. 'release-planning-2026-10-16.md'.

    Non-Latin titles (e.g. Arabic) have no ASCII slug, so the recording name is used instead.
    """
    slug = _slugify(_title(note)) or _slugify(fallback) or "meeting-notes"
    return f"{slug}-{today.isoformat()}.md"


def _title(note: str) -> str:
    for line in note.splitlines():
        if line.startswith("# "):
            return line[2:]
    return ""


def _slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > MAX_SLUG_LENGTH:
        slug = slug[:MAX_SLUG_LENGTH].rsplit("-", 1)[0]
    return slug


def _clean(note: str) -> str:
    # Em dashes are banned in the vault; a comma keeps the sentence readable
    return re.sub(rf"\s*{EM_DASH}\s*", ", ", note).strip()
