"""SoccerTrack v2 v1.2 from Hugging Face for one match, then its players cached. python nb_fetch.py MID ..."""
import os, sys
sys.path[:0] = [os.getcwd(), os.getcwd() + "/external"]
from huggingface_hub import snapshot_download
import touch_spot as T
for mid in sys.argv[1:]:
    snapshot_download("atomscott/soccertrack-v2", repo_type="dataset", revision="v1.2", local_dir=T.EXT,
                      allow_patterns=[f"videos/{mid}/*", f"gsr/{mid}/*", f"bas/{mid}/*", f"ball/{mid}_*"])
    for h in (1, 2):
        T.players(mid, h)
