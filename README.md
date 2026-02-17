# Skills

Claude Code agent skills collection. Each skill can be installed independently or all at once.

## Install All

```bash
git clone --recursive https://github.com/tivojn/skills.git
cp -r skills/*-agent skills/gogcli skills/google-maps-api skills/photo-dedup skills/video-processor skills/voicebox ~/.claude/skills/
```

## Install One Skill

```bash
git clone https://github.com/tivojn/xlsx-design-agent.git ~/.claude/skills/xlsx-design-agent
```

## Skills

| Skill | Description |
|-------|-------------|
| [xlsx-design-agent](https://github.com/tivojn/xlsx-design-agent) | Excel workbook design with openpyxl + AppleScript |
| [docx-design-agent](https://github.com/tivojn/docx-design-agent) | Word document design with python-docx + AppleScript |
| [pptx-design-agent](https://github.com/tivojn/pptx-design-agent) | PowerPoint design with python-pptx + AppleScript |
| [gogcli](https://github.com/tivojn/gogcli-skill) | Google Workspace CLI (Gmail, Calendar, Drive, Sheets, Docs) |
| [google-maps-api](https://github.com/tivojn/google-maps-api-skill) | Google Maps Platform API client (20+ APIs) |
| [photo-dedup](https://github.com/tivojn/photo-dedup) | Photo deduplication with perceptual hashing |
| [video-processor](https://github.com/tivojn/video-processor) | Video/audio transcription, translation, dubbing |
| [voicebox](https://github.com/tivojn/voicebox) | Voice toolkit: TTS, voice cloning, multi-speaker audio |

## Requirements

- macOS (AppleScript skills require Mac apps)
- Python 3
- Claude Code CLI

## License

MIT
