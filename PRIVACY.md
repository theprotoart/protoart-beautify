# Privacy

ProtoArt Beautify runs entirely on your Mac. Your pictures and settings never leave it.
There is no account, no telemetry, no analytics and no crash reporting.

## When the app goes online

| Where | When | What is sent |
| --- | --- | --- |
| huggingface.co (and its file storage) | First launch only, to download the AI models (~6 GB) | Standard file requests for pinned files. No account, token or analytics. |
| huggingface.co or github.com (release files) | The first time you use an add-on: Face Fix, Likeness, an upscaler other than 4x-UltraSharp, or Remove | The same: a request for one pinned file, checked against its SHA-256. |
| api.github.com and github.com | Only when you choose Check for Updates…, or once a day if you said yes when asked | A request for this project's latest release (with the app's version in the User-Agent); the installer download if you press Download. |

After that the app works offline.

## What is stored on your Mac

| Location | Contents |
| --- | --- |
| `~/Library/Application Support/ProtoArt Beautify/` | The downloaded models (~3 GB after one-time conversion) |
| `~/Library/Logs/ProtoArt Beautify.log` | The engine log (errors and progress, no picture data) |
| `~/Library/Preferences/com.protoart.beautify.plist` | Your settings: your own presets, knobs, save folder, filters |
| Your save folder (default: your Desktop) | Pictures you save (Auto-save puts every result here) |
| The system temporary folder | Copies of the picture being processed, and photos with spots removed; deleted when the app opens and quits |
| `~/Library/Application Support/ProtoArt Beautify/updates/` | A downloaded update installer (only if you downloaded one) |

## Removing everything

Quit the app, move it to the Trash, and delete the folders and files listed above.
