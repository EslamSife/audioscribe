# Security policy

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting instead: open the repository's **Security** tab and choose **Report a vulnerability**.
You should get an answer within a week.

## What this tool stores and sends

- `transcribe config set-key` stores the AssemblyAI key in `~/.config/transcribe/config.json`,
  readable only by your user (file mode 0600).
- Audio is uploaded to AssemblyAI only with the default `assemblyai` provider. Transcripts are
  sent to an LLM only with `--notes`, and the tool prints which one.
- Recordings (`.wav`) and transcripts are written as normal files in the folder you choose.
