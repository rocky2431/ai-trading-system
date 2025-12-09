# Context Archive

This directory stores compressed context from long sessions.

## Purpose

- Prevent context overflow (200K token limit)
- Enable 20-30 tasks per session
- Maintain session continuity

## Files

- `session-index.json` - Quick session recovery index
- `session-{timestamp}.md` - Archived session context

## Compression Triggers

- 5+ tasks completed
- Token usage > 120K (Yellow zone)
- Before /ultra-test or /ultra-deliver

## Usage

Context is automatically compressed by the `compressing-context` skill.
