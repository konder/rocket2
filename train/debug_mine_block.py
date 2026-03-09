"""
Debug script: verify mine_block events fire and check info structure.

Usage:
    python debug_mine_block.py
"""
import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import PrevActionCallback
from minestudio.simulator.callbacks.voxels import VoxelsCallback


def main():
    print("=== Debug: mine_block event + info structure ===\n")

    callbacks = [
        PrevActionCallback(),
        VoxelsCallback([-5, 5, -5, 5, -5, 5]),
    ]

    # mine_coal spawn: seed=19961103, pos=[-923.5, 78, 766.5]
    sim = MinecraftSim(
        action_type='env',
        callbacks=callbacks,
        spawn_at_pos=[-923.5, 78, 766.5],
        seed=19961103,
        initial_inventory=[{"slot": 0, "type": "diamond_pickaxe", "quantity": 1}],
    )

    print("Resetting env...")
    obs, info = sim.reset()

    print(f"\nobs keys: {list(obs.keys())}")
    print(f"info keys: {list(info.keys())}")
    print(f"obs['image'].shape: {obs['image'].shape}")

    mine_block_val = info.get("mine_block", "KEY_MISSING")
    print(f"\ninfo['mine_block'] = {mine_block_val}")
    print(f"  type: {type(mine_block_val)}")

    player_pos = info.get("player_pos", {})
    print(f"\ninfo['player_pos'] = {player_pos}")

    voxels = info.get("voxels", "KEY_MISSING")
    print(f"\ninfo['voxels'] type: {type(voxels)}")
    if isinstance(voxels, list):
        print(f"  len: {len(voxels)}")
        if voxels:
            print(f"  first item: {voxels[0]}")
    elif isinstance(voxels, dict):
        print(f"  keys: {list(voxels.keys())[:5]}")

    print("\nWarming up (40 steps)...")
    for i in range(40):
        obs, _, _, _, info = sim.step(sim.noop_action())
    print("Warmup done.")

    print("\nnoop_action() format:")
    noop = sim.noop_action()
    for k, v in noop.items():
        print(f"  {k}: {v!r} (type={type(v).__name__})")

    print("\n--- Running 200 attack steps ---")
    prev_mine = dict(info.get("mine_block") or {})
    total_mined = 0

    for step_i in range(200):
        action = sim.noop_action()
        action["attack"] = 1

        try:
            obs, _, _, _, info = sim.step(action)
        except Exception as e:
            print(f"Step {step_i} FAILED: {e}")
            traceback.print_exc()
            break

        cur_mine = info.get("mine_block") or {}

        # Print structure once
        if step_i == 0:
            print(f"\nAfter step 0, mine_block = {cur_mine}")
            print(f"  type: {type(cur_mine)}")
            if isinstance(cur_mine, dict):
                for k, v in list(cur_mine.items())[:3]:
                    print(f"  [{k!r}] = {v!r} (type={type(v).__name__})")

        # Detect changes
        changed = False
        if isinstance(cur_mine, dict):
            for ns, blocks in cur_mine.items():
                if not isinstance(blocks, dict):
                    # flat format: ns is block name, blocks is count
                    cnt = int(blocks) if blocks is not None else 0
                    prev_cnt = int(prev_mine.get(ns, 0))
                    if cnt > prev_cnt:
                        print(f"  [MINED flat] step={step_i} key={ns!r} {prev_cnt}->{cnt}")
                        total_mined += 1
                        changed = True
                else:
                    for blk, cnt in blocks.items():
                        cnt = int(cnt) if cnt is not None else 0
                        prev_cnt = int((prev_mine.get(ns) or {}).get(blk, 0))
                        if cnt > prev_cnt:
                            print(f"  [MINED nested] step={step_i} {ns}:{blk} {prev_cnt}->{cnt}")
                            total_mined += 1
                            changed = True

        prev_mine = {k: dict(v) if isinstance(v, dict) else v
                     for k, v in cur_mine.items()} if isinstance(cur_mine, dict) else {}

    print(f"\nTotal mine events detected: {total_mined}")
    sim.close()
    print("Done.")


if __name__ == "__main__":
    main()
