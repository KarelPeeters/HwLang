Create lyrics, audio and a music video about the KHDL language, whose compiler is implemented in his repository. Do all your work in a video/ subdirectory.

First start subagent to generate a full summary of the language, the compiler, the goals, the future, ... Also look at the files in `ignored/feedback`, everything in `design/` and `docs/` for more background information. Assume all planned features and bugfixes are actually implemented.

Then propose lyrics. Make them clever, targeting people with deep digital hardware design knowledge and frustrations. Use classic verse/chorus structure, I especially like the P(doom) lyrics linked below, but we want something in hardware design, not at all safety research related.

After asking the user for feedback on the lyrics, you'll also generate the audio and a video.

Sources of inspiration for the video:
* https://www.youtube.com/watch?v=8j-hR4fJywU
* https://github.com/JohnHeibel/PDoomVideo
* https://www.youtube.com/watch?v=u1kxLBF3Im4
* This comment:
    > I generated the lyrics with gemini-3.8-flash, fed them into Suno to create the song, then dropped the mp3 of the track and a text file of the lyrics into a local directory and pointed Claude Code at it using Opus 5.5 (high). I literally did nothing else, Claude did the design, the edit, and the final output to a 1080p60 video without me touching a frame.
    > My exact prompt was:
    > Create a procedurally generated video based on the song "Omakase.mp3" that you will find in this directory. There is also a lyrics file for reference. It's a song about the linux OS distribution Omarchy. I want you to independently do an end-to-end complete pass on making a music video. Use the exact audio track and think and feel very deeply about what is the best way to visually represent all of the lyrics on screen. You can do truly anything that you think might best let you visually express yourself, including abstract motion graphics.
    > You can use the internet freely to pull in references. You can look at motion design. I want you to make a new music video that has beautifully rendered JavaScript animations and push the aesthetics in any direction you want and consider what is part of the modern zeitgeist.
    > You also need to be open to going back and revisiting things in order to be able to reiterate. You're going to want to watch the entire video multiple times, take screenshots at individual parts, and think about if something is really up to the bar of quality that we need here. I trust that you can do this, and I think that it's really important to nail the style of animations.
    > You can also use search abilities and find other references to pull from for motion, for JavaScript, animations, et cetera, and integrate them.