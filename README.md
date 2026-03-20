# ACV Cswk

deps:
- pretrained cut model
- pretrained dino model - https://github.com/facebookresearch/dinov2/blob/main/README.md

## Dataset

| File | Duration | Size | Resolution | FPS | Frames | Bitrate | Domain |
|------|----------|------|------------|-----|--------|---------|--------|
| `downloaded_data/Train/game/MafiaVideogame.mp4` | 2:21:04 | 484 MB | 1280×720 | 30.00 | 253,944 | ~0.5 Mbps | game |
| `downloaded_data/Train/movie/TheGodfather.mp4` | 0:08:59 | 70 MB | 1280×720 | 23.98 | 12,945 | ~1.1 Mbps | movie |
| `downloaded_data/Train/movie/TheIrishman.mp4` | 0:15:27 | 114 MB | 1280×720 | 25.06 | 23,236 | ~1.0 Mbps | movie |
| `downloaded_data/Train/movie/TheSopranos.mp4` | 0:28:43 | 121 MB | 1280×720 | 30.00 | 51,714 | ~0.6 Mbps | movie |
| `downloaded_data/Test/Test.mp4` | 0:01:10 | 17 MB | 1280×720 | 30.00 | 2,114 | ~2.0 Mbps | game |

**Total training data:** ~53 min movie + ~141 min game  
**Total test data:** 70 seconds, 2,114 frames @ 30 fps

## Paper requirements
- Please be explicit in your reports about the hardware you used to train your model, and the time it took
- Can use others' code but have to do novel adaptiation; bad results but high orginality >> good results with no originality
- can use external datasets / pretrained models if you want - provided you cite them!

## Links
- 1st claude chat - https://claude.ai/chat/04bd8392-d930-4c2b-9eb0-013b1368697e
- pytorch cyclegan (orig authors) - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?tab=readme-ov-file