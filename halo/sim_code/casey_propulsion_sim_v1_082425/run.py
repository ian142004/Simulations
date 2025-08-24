# Create a separate runner file that uses the "example wiring" (default_system)
# from rocket_prop_sim_v1.py and executes the simulation. Also adds optional CSV export.
import matplotlib.pyplot as plt
from rocket_prop_sim_v1 import default_system, run_sim
from textwrap import dedent

runner = dedent
# run_simulation.py
# Runner that imports the example wiring from rocket_prop_sim_v1.py and executes the sim.
# Optional: plots and CSV export.
import argparse
import csv
import sys

try:
    import matplotlib.pyplot as plt  # optional
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from rocket_prop_sim_v1 import default_system, run_sim

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run N2O/IPA pressure-fed rocket simulation")
    parser.add_argument("--t_end", type=float, default=None, help="Override sim end time (s)")
    parser.add_argument("--dt", type=float, default=None, help="Override sim timestep (s)")
    parser.add_argument("--csv", type=str, default=None, help="Write outputs to CSV file")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args(argv)

    # Build system config and initial state from example wiring
    cfg, st = default_system()

    # Allow simple overrides from CLI
    if args.t_end is not None:
        cfg = type(cfg)(t_end=args.t_end, dt=cfg.dt,
                        A_ox_pintle_fn=cfg.A_ox_pintle_fn,
                        throttle_ox_fn=cfg.throttle_ox_fn,
                        throttle_fuel_fn=cfg.throttle_fuel_fn)
    if args.dt is not None:
        cfg = type(cfg)(t_end=cfg.t_end, dt=args.dt,
                        A_ox_pintle_fn=cfg.A_ox_pintle_fn,
                        throttle_ox_fn=cfg.throttle_ox_fn,
                        throttle_fuel_fn=cfg.throttle_fuel_fn)

    # Run the simulation
    out = run_sim(cfg, st)

    # Minimal console output
    print("Final t = %.3f s" % out["t"][-1])
    print("Final Pc = %.1f bar" % (out["Pc"][-1] / 1e5))
    print("Final MR (O/F) = %.3f" % out["MR"][-1])
    print("Final thrust = %.1f N" % out["thrust"][-1])
    print("Final P_ipa = %.1f bar, P_n2o = %.1f bar" %
          (out["P_ipa"][-1] / 1e5, out["P_n2o"][-1] / 1e5))

    # Optional CSV write
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s","Pc_Pa","MR","mdot_fuel_kgps","mdot_ox_kgps","thrust_N","P_ipa_Pa","P_n2o_Pa"])
            for i in range(len(out["t"])):
                w.writerow([out["t"][i], out["Pc"][i], out["MR"][i], out["mdot_fuel"][i],
                            out["mdot_ox"][i], out["thrust"][i], out["P_ipa"][i], out["P_n2o"][i]])
        print(f"Wrote CSV to {args.csv}")

    # Optional plotting
    if not args.no_plot and _HAVE_MPL:
        plt.figure()
        plt.plot(out["t"], out["Pc"]/1e5, label="Chamber Pressure [bar]")
        plt.plot(out["t"], out["P_ipa"]/1e5, label="IPA Tank [bar]")
        plt.plot(out["t"], out["P_n2o"]/1e5, label="N2O Tank [bar]")
        plt.xlabel("Time [s]")
        plt.ylabel("Pressure [bar]")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(out["t"], out["mdot_fuel"], label="Fuel mdot [kg/s]")
        plt.plot(out["t"], out["mdot_ox"], label="Ox mdot [kg/s]")
        plt.xlabel("Time [s]")
        plt.ylabel("Mass flow [kg/s]")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.plot(out["t"], out["thrust"])
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [N]")
        plt.title("Thrust vs Time")
        plt.grid(True)

        plt.show()
    elif not _HAVE_MPL and not args.no_plot:
        print("matplotlib not available; skipping plots.", file=sys.stderr)

if __name__ == "__main__":
    main()

path = "/mnt/data/run_simulation.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(runner)

path
