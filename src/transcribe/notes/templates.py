"""Note templates: which sections the LLM writes for each kind of recording.

All templates share the same rules; only the sections change. Action items are
Markdown tasks (`- [ ]`) in the work templates so Obsidian can track them.
"""

from transcribe.transcription.model import TranscriptionError

DEFAULT = "general"

RULES = """You turn raw audio transcripts into clear, structured Markdown notes.

Rules:
- Write the note in the same language as the transcript.
- Be faithful to the transcript. Never invent facts that are not in it.
- Transcripts come from imperfect speech recognition: fix obvious recognition
  errors from context, but do not change the meaning.
- Keep timestamps in [MM:SS] form when you quote or reference a moment.
- Do not use em dashes; use commas, colons or periods instead.
"""

_TASKS = """Write each one as a Markdown task: "- [ ] Owner: task (due date)". Leave out
the owner or the date when nobody said them. Write "None" if there are none."""

SECTIONS = {
    "general": """
Produce exactly these sections:
# <A short descriptive title>
## Summary
2-4 sentences: what this recording is about and its main message.
## Key Points
The important ideas as short bullet points, in the order they appear.
## Action Items
Concrete tasks, decisions, or follow-ups mentioned. Write "None" if there are none.
## Notable Quotes
Up to 5 memorable or important quotes, each with its [MM:SS] timestamp.
""",
    "standup": f"""
This is a team standup. Produce exactly these sections:
# <Team or project> standup
## Summary
1-2 sentences: the overall state of the work.
## Updates
One "### <name>" block per person, with three bullets: "Done:", "Next:" and
"Blockers:". Use the names people say. If speakers have no names, use the
speaker labels; if there are none either, group the updates by topic.
## Blockers
Every blocker, with who is blocked and who can help, if said. Write "None" if there are none.
## Action Items
{_TASKS}
""",
    "design-review": f"""
This is a technical design review. Produce exactly these sections:
# <The design being reviewed>
## Context
The problem being solved and the constraints mentioned.
## Options Discussed
One "### <option>" block per option, with the pros and cons people gave.
## Decisions
What was decided, and why. Write "No decision was made" if nothing was decided.
## Open Questions
What is still unknown or needs a follow-up.
## Action Items
{_TASKS}
## Draft ADR
An Architecture Decision Record built only from what was said, with the fields
"Status: Proposed", "Context", "Decision" and "Consequences". Leave this
section out when no decision was made.
""",
    "incident": f"""
This is an incident call or post-incident review. Use blameless language:
describe what systems and processes did, not who is at fault.
Produce exactly these sections:
# <Short incident name>
## Summary
2-3 sentences: what broke, who or what was affected, and the current status.
## Timeline
Events in order, as bullets. Use the clock times people say, copied as they
appear (speech recognition may write "2:10" as "210"); never convert or guess
a time. Without a clock time, use the [MM:SS] moment in the recording.
## Impact
Users, systems, data or money affected, with numbers only if they were said.
## Root Cause
The cause as stated. Write "Not determined in this call" if it was not found.
## Mitigation and Fix
What was done to stop the impact, and what the lasting fix is.
## Follow-ups
{_TASKS}
## Open Questions
What is still unknown.
""",
}

NAMES = tuple(SECTIONS)


def system_prompt(template: str) -> str:
    return RULES + SECTIONS[check(template)]


def check(template: str) -> str:
    """Return the template name, or raise a clear error listing the valid ones."""
    name = template.strip().lower()
    if name not in SECTIONS:
        raise TranscriptionError(
            f"Unknown note template '{template}'. Use one of: {', '.join(NAMES)}"
        )
    return name
