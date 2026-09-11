"""
PURPOSE: Cambiar la máquina del Lightning Studio actual (CPU <-> GPU) desde la línea de
         comandos, usando lightning_sdk con las credenciales del entorno del Studio.
         Uso: python tools/lightning/switch_machine.py L4        # o T4, L40S, A100, CPU
              python tools/lightning/switch_machine.py --status  # solo muestra la máquina actual
         El cambio reinicia el Studio (la sesión actual se corta). Gasta créditos: usar solo
         con aprobación de Gabriel y volver a CPU al terminar el trabajo GPU.
CHANGELOG:
  - 2026-09-11: Creación inicial.
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Switch the current Lightning Studio machine.")
    parser.add_argument("machine", nargs="?", help="Machine name from lightning_sdk.Machine (e.g. L4, T4, L40S, A100, CPU)")
    parser.add_argument("--status", action="store_true", help="Print current machine and exit")
    parser.add_argument("--interruptible", action="store_true", help="Use an interruptible (spot) instance")
    parser.add_argument("--cloud-provider", default=None,
                        help="Cloud provider from lightning_sdk.machine.CloudProvider (e.g. AWS, LIGHTNING); "
                             "needed when the target machine only exists on another cloud account")
    args = parser.parse_args()

    from lightning_sdk import Machine, Studio
    from lightning_sdk.machine import CloudProvider

    studio = Studio()
    print(f"[INFO] Studio: {studio.name} | status: {studio.status} | machine: {studio.machine}")
    if args.status or not args.machine:
        return 0

    name = args.machine.upper()
    if not hasattr(Machine, name):
        print(f"[ERROR] Unknown machine {name!r}. Examples: CPU, T4, L4, L40S, A100")
        return 2
    target = getattr(Machine, name)
    if str(studio.machine).upper().endswith(name):
        print(f"[INFO] Already on {name}; nothing to do.")
        return 0

    provider = None
    if args.cloud_provider:
        pname = args.cloud_provider.upper()
        if not hasattr(CloudProvider, pname):
            print(f"[ERROR] Unknown cloud provider {pname!r}. Options: "
                  f"{[c for c in dir(CloudProvider) if not c.startswith('_')]}")
            return 2
        provider = getattr(CloudProvider, pname)

    print(f"[INFO] Switching to {name}{' (interruptible)' if args.interruptible else ''}"
          f"{' on ' + provider.name if provider else ''} ... the Studio will restart and this session will end.")
    studio.switch_machine(target, interruptible=args.interruptible, cloud_provider=provider)
    print(f"[INFO] Switched. Now on: {studio.machine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
