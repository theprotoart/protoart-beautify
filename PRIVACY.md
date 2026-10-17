# Privacy

ProtoArt Beautify runs entirely on your Mac. Your pictures and settings never leave it.
There is no account, no telemetry, no analytics and no crash reporting.

## When the app goes online

| Where | When | What is sent |
| --- | --- | --- |
| huggingface.co (and its file storage) | First launch only, to download the AI models (~6 GB) | Standard file requests for pinned files. No account, token or analytics. |

After that the app works offline.

## What is stored on your Mac

| Location | Contents |
| --- | --- |
| `~/Library/Application Support/ProtoArt Beautify/` | The downloaded models (~3 GB after one-time conversion) |
| `~/Library/Logs/ProtoArt Beautify.log` | The engine log (errors and progress, no picture data) |
| `~/Library/Preferences/com.protoart.beautify.plist` | Your settings: presets, knobs, save folder, filters |
| Your save folder (default `~/Pictures/ProtoArt Beautify`) | Pictures you save |
| The system temporary folder | Copies of the picture being processed; deleted when the app opens and quits |

## Removing everything

Quit the app, move it to the Trash, and delete the folders and files listed above.
