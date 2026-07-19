import importlib.util
from pathlib import Path

import cicada_analysis
import numpy as np
from cicada_analysis.cicada_tools.ephys.psth_utils import compute_psth
from cicada_nwb.nwb_session import NWBSession

# cicada_analysis.cicada_tools.imaging's __init__ chain imports anatomy_utils.py, which
# loads a structures.json file missing from the installed package (upstream packaging bug,
# v0.3.1) -- load event_alignment.py directly to avoid triggering that broken __init__.
_event_alignment_path = (
    Path(cicada_analysis.__file__).parent / "cicada_tools" / "imaging" / "event_alignment.py"
)
_spec = importlib.util.spec_from_file_location("cicada_event_alignment", _event_alignment_path)
_event_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_event_alignment)
align_array_to_timestamps = _event_alignment.align_array_to_timestamps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "nwb"

T_BEFORE = 2.0  # seconds before stim_onset
T_AFTER = 3.0  # seconds after stim_onset
BIN_SIZE = 0.01  # seconds, neural PSTH bin width

DLC_PARTS = ["jaw_angle", "jaw_distance"]


def extract_session(nwb_path: Path) -> dict:
    with NWBSession(str(nwb_path)) as session:
        trial_df = session.behavior.get_trial_table()
        trial_df = trial_df.loc[trial_df["early_lick"] == 0].reset_index(drop=True)
        event_times = trial_df["stim_onset"].to_numpy()

        units_df = session.electrophysiology.get_units_table()
        good_units = units_df.loc[units_df["bc_label"] == "good"]

        # compute_psth's t_before/t_after are literal window bounds relative to the event
        # (so "before" must be negative), unlike align_array_to_timestamps' window_s below
        # which takes positive magnitudes -- confirmed against build_peth_dataframe's usage.
        neural, time_neural = compute_psth(
            spike_times=list(good_units["spike_times"]),
            event_times=event_times,
            t_before=-T_BEFORE,
            t_after=T_AFTER,
            bin_size=BIN_SIZE,
        )

        traces, ts_ref = [], None
        for part in DLC_PARTS:
            data = np.asarray(session.behavior.get_behavioral_time_series(part).data[:])
            ts = session.behavior.get_behavioral_time_series_timestamps(part)
            if len(data) != len(ts):
                raise ValueError(
                    f"{nwb_path.name}: '{part}' data/timestamps length mismatch "
                    f"({len(data)} vs {len(ts)}) -- exclude this session."
                )
            traces.append(data)
            ts_ref = ts
        dlc_stack = np.stack(traces, axis=0)  # (n_bodyparts, n_frames)

        dlc_aligned = align_array_to_timestamps(
            data=dlc_stack,
            event_timestamps=event_times,
            frame_timestamps=ts_ref,
            window_s=(T_BEFORE, T_AFTER),
        )  # (n_bodyparts, n_trials, n_t_dlc)
        dlc_aligned = dlc_aligned.transpose(1, 2, 0)  # (n_trials, n_t_dlc, n_bodyparts)

        # Mirrors align_array_to_timestamps' internal frame-count conversion so the
        # returned time axis lines up exactly with dlc_aligned's bins.
        sampling_rate_dlc = 1.0 / np.median(np.diff(ts_ref))
        n_pre = int(np.ceil(sampling_rate_dlc * T_BEFORE))
        n_post = int(np.floor(sampling_rate_dlc * T_AFTER))
        time_dlc = np.arange(-n_pre, n_post + 1) / sampling_rate_dlc

        return dict(
            neural=neural,
            time_neural=time_neural,
            unit_ids=good_units.index.to_numpy(),
            dlc=dlc_aligned,
            time_dlc=time_dlc,
            bodyparts=np.array(DLC_PARTS),
            trial_type=trial_df["trial_type"].to_numpy(),
            lick_flag=trial_df["lick_flag"].to_numpy(),
            context=trial_df["context"].to_numpy(),
            subject_id=session.subject_id,
            session_id=session.session_id,
        )


def main():
    for nwb_path in sorted(DATA_DIR.glob("*.nwb")):
        result = extract_session(nwb_path)
        out_path = DATA_DIR / f"{result['session_id']}_aligned.npz"
        np.savez(out_path, **result)
        print(f"{nwb_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
