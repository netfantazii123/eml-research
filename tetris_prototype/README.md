# tetris_prototype — early prototype (v1, frame-by-frame)

First version of Tetris with frame-by-frame control: the PPO oracle selected an
action on every frame. Over 4.2M steps the agent did not learn to clear lines
reliably — the line reward was spread across hundreds of frames.

Superseded by [`../tetris`](../tetris/) (placement-based action space: one
action = one piece placement). This directory is kept for reference and is not
developed further.
