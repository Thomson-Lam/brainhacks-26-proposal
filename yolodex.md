
We built an end-to-end “video → dataset → agent” pipeline as a Codex Skill plugin that can turn any YouTube video into a training-ready vision dataset. You paste a URL, and the skill runs locally to download the video, extract frames, and spin up as many parallel Codex labeling agents as needed to generate bounding boxes for the objects of interest. It exports a full YOLO dataset on-device, which can immediately train a detector. That detector becomes a game-state extractor from pixels, enabling bots to play any game without emulator hooks or engine access, essentially compressing dataset creation + model bootstrapping into one repeatable command.

Check inside yolodex/ for more details about how the skill works. The current yolodex/ directory works for Codex only as of now.

[Github](https://github.com/qtzx06/yolodex)


