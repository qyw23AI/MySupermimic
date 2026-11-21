# WBLWBL flat walk + commanded turn AMP pool v3

This pool is dedicated to the PhysX open-flat experiment. It contains two Kine2Go walk clips re-retargeted against the current deployment MJCF (SHA256 `a6ae3a95...`) plus three WBLWBL-specific four-beat IK walk clips at 0.22/0.28/0.32 m/s.

The command sampler is restricted to `forward` and `turn`. There is intentionally no AMP turn reference: the current retargeted/atomic/procedural turn candidates have poor yaw replay in the current robot, while the 10k PhysX policy already passes commanded +/-0.6 rad/s turns. The yaw behavior is therefore trained by command tracking reward, with AMP preserving a walk-like leg style.

`train.json` is the only file loaded by AMP. `manifest.json` records source and quality provenance.
