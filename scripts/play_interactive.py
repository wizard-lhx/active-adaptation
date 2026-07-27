"""Record effective impedance with interactive push and sling controls."""

from __future__ import annotations

import sys


FIXED_OVERRIDES = (
    "backend=isaac",
    "headless=false",
    "device=cpu",
    "task.num_envs=1",
    "task.command._target_=InteractiveTwist",
    "++task.command.teleop=true",
)


def main() -> None:
    sys.argv.extend(FIXED_OVERRIDES)
    print(
        "Interactive controls:\n"
        "  H                 toggle the vertical support sling\n"
        "  UP / DOWN         raise / lower the sling while enabled\n"
        "  W / A / S / D     locomotion command\n"
        "  LEFT / RIGHT      yaw command\n"
        "  X / Y / Z         hold to apply +body-axis force\n"
        "  SHIFT + left drag spring-grab any rigid body (CPU PhysX)\n"
        "  SHIFT + double click push the pointed rigid body\n"
        "Effective-damping clamp (CLI):\n"
        "  off       algo.eff_impedance.clamp.enabled=false (default)\n"
        "  baseline  algo.eff_impedance.clamp.enabled=true "
        "algo.eff_impedance.clamp.baseline_mode=true\n"
        "  active    algo.eff_impedance.clamp.enabled=true "
        "algo.eff_impedance.clamp.baseline_mode=false"
    )

    from play_eff import main as play_main

    play_main()


if __name__ == "__main__":
    main()
