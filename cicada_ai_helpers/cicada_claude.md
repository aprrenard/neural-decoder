# CICADA — Claude Code Context

<!--
  HOW TO LOAD THESE FILES INTO CLAUDE CODE
  ─────────────────────────────────────────
  1. Copy the cicada_ai_helpers/ folder into your project root.
  2. Add these THREE lines to your project's CLAUDE.md (create it if absent):
       @cicada_ai_helpers/cicada_claude.md
       @cicada_ai_helpers/cicada_nwb_api.md
       @cicada_ai_helpers/cicada_tools_api.md
  All three files are loaded from the project root — no nested import issues.
  Order matters: cicada_claude.md first (router and invariants), API files after.
-->

---

## How to use the API reference files

`cicada_nwb_api.md` and `cicada_tools_api.md` are loaded into your context alongside this file. They are the **authoritative source for all method signatures, parameter names, and return shapes**. Before writing any CICADA code:

- Look up the exact signature in the relevant API file — never rely on memory or inference.
- Use the router below to identify which file and which section to consult.
- If a method name or parameter is not found in the API files, say so explicitly rather than guessing.

**Preference rule:** Before writing any custom numeric/statistical/display code:
1. Check the Level 3 table and `cicada_tools_api.md` for a matching function.
2. If not found there, **grep the installed cicada_tools source before concluding nothing exists** — the API doc may be incomplete. Locate the source via the Source code fallback section below, then Grep inside `cicada_tools/` for relevant terms (e.g. `correlation`, `similarity`, `psth`, `detect`, `cluster`, `raster`, `connectivity`).
3. Only proceed with a custom implementation if the source grep also finds nothing.

---

## Source code fallback

If the API files don't cover a method or you need to verify a signature, follow this sequence:

**Step 1 — detect active Python / conda env:**
```bash
where python        # PowerShell / cmd
# or
which python        # bash
```
If the path contains `envs/<name>/` or `envs\<name>\`, the conda env name is `<name>`.

**Step 2 — get the package location:**
- If an env name was found: `conda run -n <env_name> pip show cicada-analysis`
- Otherwise: `pip show cicada-analysis`

Look at the `Location:` line — the source is at `<Location>/cicada_analysis/`. Claude Code can then `Read`, `Grep`, or `Glob` inside that path like any local directory.

**Step 3 — only if Step 2 gives no clear Location:** ask the user: "What is your conda environment name?"

---

## What is CICADA

CICADA is a neuroscience analysis pipeline for NWB files built around two external packages:

- **`cicada-nwb`** — read-only data access. Entry point: `NWBSession`. Never computes, never writes.
- **`cicada-analysis`** — contains **`cicada_tools`** (computation and display utilities) and the **`CicadaAnalysis`** plugin system.

Dependency chain: `cicada-gui → cicada-analysis → cicada-nwb → PyNWB + HDF5`

**Repository layout:**
```
cicada_nwb/
├── nwb_session.py           # NWBSession
├── submodules/              # NWBCalciumImaging, NWBBehavior, NWBElectrophysiology,
│                            #   NWBWidefield, NWBMetadata, NWBOptogenetic
└── lab_submodules/          # NWBCossartLab, NWBPetersenLab (auto-discovered, no registration)

cicada_analysis/
├── base/cicada_analysis.py  # CicadaAnalysis ABC
├── base/argument_schema.py  # AnalysisArgument, AnalysisArgumentsHandler
├── config/runner.py         # run_from_config(), run_from_combined(), generate_config()
├── core_analysis/           # Built-in display/export plugins
├── Cossart_lab_plugins/     # Assemblies, Connectivity, Descriptive_statistics, Epochs, Sequences
├── Petersen_lab_plugins/    # Behavior, Ephys, Widefield
└── cicada_tools/            # Computation + display utilities — see cicada_tools_api.md
```

---

## Critical invariants

1. `NWBSession` must always be closed after use. Choose the pattern based on the requirements:

   **Default — process one session at a time and aggregate** (memory-efficient; use whenever per-session results can be merged after the loop):
   ```python
   from cicada_nwb import NWBSession
   results = []
   for path in nwb_paths:
       with NWBSession(path) as session:
           results.append(session.calcium_imaging.get_roi_response_serie_data(keys=[...]))
   # merge results after the loop
   ```

   **All sessions open simultaneously** — use only when the computation cannot be decomposed per session, e.g. cross-session normalization, computing a common reference across all sessions at once:
   ```python
   from cicada_nwb import NWBSession
   sessions = [NWBSession(path) for path in nwb_paths]
   try:
       # work across all open sessions
       ...
   finally:
       for s in sessions:
           s.close()
   ```
2. **cicada-nwb always returns epochs and events as `(2, N)`**: row 0 = start times, row 1 = stop times. However, many cicada-tools functions expect event timestamps as a flat `(N,)` array. Extract the relevant row before passing:
   ```python
   epochs = session.behavior.get_behavioral_epochs_times("running")  # (2, N)
   events = session.behavior.get_behavioral_events_times("reward")   # (2, N)

   # cicada-tools that take full epoch intervals: pass (2, N) as-is
   # e.g. from_timestamps_to_frame_epochs, filter_epoch_timestamps, filter_events_based_on_epochs

   # cicada-tools that take a list of event times: extract row 0 (start times)
   # e.g. align_array_to_timestamps, compute_psth
   event_times = events[0]   # (N,)
   ```
   Check each cicada-tools function's signature in `cicada_tools_api.md` to confirm expected shape.
3. The `keys` argument for calcium imaging methods is a 3-element list `[module_name, interface_name, rrs_name]`. Always discover valid keys first with `session.calcium_imaging.get_roi_response_series()`.
4. Do **not** store `get_widefield_dff0_mmap()` results outside the `with` block — the HDF5 handle closes.
5. Lab submodules (`session.cossart`, `session.petersen`) are auto-attached to every `NWBSession` — no explicit import needed.

---

## Router — top-down resolution protocol

Always resolve top-down. At each level, if the needed information is missing or ambiguous, ask **one** clarifying question before proceeding. If the request is specific enough that modality and data type are clear, skip directly to the relevant level. **If a clarifying answer changes a previous assumption, restart from Level 1.**

---

### Level 0 — Classify the request by intent

Classify by what the user is trying to accomplish, not by the verb used:

| Request type | Indicators | How to proceed |
|---|---|---|
| **Conceptual / design** | Describes a scientific goal without naming data types or methods | Ask targeted questions to gather the minimum context needed: (1) what type of neural recording? (2) what behavioral or experimental variable? (3) what output is expected? Then resolve each component through Levels 1–3. |
| **Data extraction** | Wants to retrieve specific data from NWB files | Level 1 → Level 2 |
| **Computation / analysis / display** | Wants to process, transform, or visualize data | Level 1 (confirm modality) → Level 3 |
| **Compound** | Involves both extraction and computation, or neural + behavioral data together | Resolve each component independently. For neural + behavioral: establish neural modality and behavioral modality separately at Level 1, extract each at Level 2, then compose at Level 3. Extraction always before computation. |

---

### Level 1 — Establish data modality

**Principle:** a term is modality-ambiguous if it could plausibly apply to more than one row in the table below. When in doubt, treat it as ambiguous. Do not infer modality from surrounding context alone — only a term that directly maps to a unique row below counts as unambiguous.

Ambiguous terms include but are not limited to: `timestamps` · `traces` · `data` · `signal` · `activity` · `recording` · `frames` · `rate` · `events` · `response` · `onset` · `baseline` · `population` · `epoch-aligned` · `trial-averaged`

| If the description unambiguously refers to… | Modality → Submodule |
|---|---|
| Fluorescence, dF/F0, calcium, ROI, optically imaged cells, two-photon, one-photon, GCaMP | Calcium imaging → `session.calcium_imaging` |
| Running, licking, reward, movement, trial, stimulus, locomotion, position, speed, behavioral state | Behavior → `session.behavior` |
| Spikes, units, electrodes, LFP, extracellular, multi-unit, Neuropixels, tetrode | Electrophysiology → `session.electrophysiology` |
| Widefield, mesoscale, dorsal cortical imaging | Widefield → `session.widefield` |
| Animal info, age, genotype, sex, weight, experimenter, lab, session description | Metadata → `session.metadata` |

If still ambiguous: **"Is this data from calcium imaging, electrophysiology, behavior, or widefield imaging?"**

---

### Level 2 — Identify specific data within the modality

When names are unknown, always run discovery first and **show the results to the user to confirm which to use** before calling any data accessor.

**Calcium imaging**

Discovery:
```python
session.calcium_imaging.get_roi_response_series()
# → nested dict {module: {interface: [rrs_name, ...]}}
# If multiple series exist, present the options and ask which to use.
```
Build `keys = [module_name, interface_name, rrs_name]` from the chosen series, then:

| User wants… | Method | Returns |
|---|---|---|
| Fluorescence / activity values | `get_roi_response_serie_data(keys)` | `(n_cells, n_times)` |
| Time axis for traces | `get_roi_response_serie_timestamps(keys)` | `(n_times,)` |
| Sampling rate | `get_rrs_sampling_rate(keys)` | float Hz |
| Cell types or indices per type | `get_cell_indices_by_cell_type(keys)` | `{type: indices}` |
| Spatial footprints / masks | `get_segmentations()` → build `seg_info`, then `get_pixel_mask(seg_info)` or `get_image_mask(seg_info)` | list of arrays |
| Raw movie frames or movie info | `get_ci_movies()`, `get_ci_movie_dimension()`, `get_ci_movie_sampling_rate()` | — |

**Behavior**

Discovery:
```python
session.behavior.get_behavioral_epochs_names()   # on/off state periods with duration
session.behavior.get_behavioral_events_names()   # discrete point timestamps
session.behavior.has_trial_table()               # structured trial table
```
If the user says "events" or "epochs" without specifying which kind, ask: **"Is it a state with duration (epoch = start/stop interval) or a discrete instant (event = single timestamp)?"**

| User wants… | Method | Returns |
|---|---|---|
| On/off state periods (duration) | `get_behavioral_epochs_times(name)` | `(2, N)` |
| Discrete timestamps (no duration) | `get_behavioral_events_times(name)` | `(2, N)` |
| Continuous signal (speed, position) | `get_mouse_speed_info()` / `get_mouse_position_info()` | `(data, timestamps)` |
| Structured trials with metadata | `get_trial_table()` | DataFrame |
| Interval tables | `get_interval_times(name)` | `(2, N)` |

**Electrophysiology**

| User wants… | Method | Returns |
|---|---|---|
| Spike times for one unit | `get_single_unit_spike_times(unit_id)` | `(n_spikes,)` |
| All units + metadata | `get_units_table()` | DataFrame |
| Spike times by brain region or label | `get_spiketimes_group(col_name, target, ...)` | array |
| Brain region per unit | `get_units_location()` | array of strings |
| Electrode geometry | `get_electrode_info()` | DataFrame |

**Widefield**

Discovery:
```python
session.widefield.get_wf_acquisition_dict()
# → {module: [acquisition_name, ...]}
# If multiple acquisitions exist, show options and ask which to use.
```
Build `keys = [module_name, interface_name]`, then:

| User wants… | Method | Returns |
|---|---|---|
| dF/F0 pixel data (slice) | `get_widefield_dff0(keys, start, stop)` | `(n_frames, H, W)` |
| dF/F0 lazy full access | `get_widefield_dff0_mmap(keys)` → inside session context only | H5 dataset |
| Time axis | `get_widefield_timestamps(keys)` | `(n_frames,)` |

---

### Level 3 — Computation and display (cicada-tools)

Carry the modality from Level 1. Full signatures in `cicada_tools_api.md` — section references given below for direct lookup.

**Modality-dependent tools:**

| Goal | Calcium imaging | Electrophysiology | Widefield | Section in cicada_tools_api.md |
|---|---|---|---|---|
| Extract fixed windows around events | events in **seconds** → `align_array_to_timestamps(data, event_ts, frame_ts, window_s)` / events as **frame indices** → `align_array_to_events(data, event_frames, window_frames)` | — | slice `get_widefield_dff0(keys, start, stop)` per event window | §Event Alignment |
| PSTH | `build_psth_dataframe()` or `build_population_psth()` | `compute_psth()` | — | §PSTH — Calcium Imaging / §PSTH — Electrophysiology |
| Detect transients in neural signal | `detect_small_transients()` | — | — | §Signal Processing |
| Detect population synchrony (SCE) | `detect_sce_traces()` (from raw traces) / `detect_sce_raster()` (from raster) | — | — | §Population Burst / SCE Detection |
| Render traces / heatmap | `render_traces()` / `render_traces_heatmap()` | `render_spikes()` | — | §Calcium Imaging — Display / §Ephys Display |
| Load / slice movie frames | `as_sliceable_movie()`, `TiffMovieView` | — | `get_widefield_dff0_mmap()` (NWB) | §Tiff Movie Handling |
| Average frames by epoch | `get_avg_frames_by_epoch()` | — | — | §Tiff Movie Handling |

**Modality-independent tools:**

| Goal | Function | Section in cicada_tools_api.md |
|---|---|---|
| Convert epoch timestamps (s) → frame indices | `from_timestamps_to_frame_epochs(epoch_array_2N, frame_timestamps_1D)` | §Period Utilities |
| Filter epochs by invalid/artifact periods | `filter_epoch_timestamps()` | §Period Utilities |
| Keep only events inside epoch windows | `filter_events_based_on_epochs(events_1D, epochs_2N)` | §Period Utilities |
| Nearest index in array | `find_nearest()` | §Array Utilities |
| Save a figure | `save_figure()` | §Display — Figure |
| Custom colormaps (hotcold, cyanmagenta…) | `get_colormap()` | §Display — Figure |
| Raster visualization | `plot_raster()` | §Display — Raster |
| Box plot / violin / histogram / scatter | `plot_box_plots()` / `plot_violin_distribution()` / `plot_hist_distribution()` / `plot_scatter_family()` | §Display — Distributions |
| PSTH visualization | `plot_one_psth()` / `plot_several_psth()` / `plot_population_psth()` | §Display — PSTH Plots |
| Graph visualization | `plot_graph()` / `plot_connectivity_graphs()` | §Display — Graphs |
| Pairwise similarity / correlation matrix between neurons | `compute_similarity_matrix()` (Pearson/Hamming/Jaccard/Cosine), `build_pearson_adjacency_matrix()`, `plot_similarity_matrix()` | §Pairwise Cell Similarity |
| Spike-based connectivity graph | `build_connectivity_graphs()` | §Graphs & Connectivity |
| Hub cell detection | `detect_hub_cells()` | §Graphs & Connectivity |
| Statistical tests, FDR | `compare_two_distributions()`, `bh_reject()`, `multiple_comparison_one_factor_effect()` | §Stats |
| Behavior trial table utilities | `filter_table_with_dict()`, `build_standard_behavior_table()` | §Behavior Analysis |
| DLC trace preprocessing / PSTH | `preprocess_dlc_trace()`, `build_dlc_psth_df()` | §Behavior Analysis |
| Widefield atlas / average images | `get_allen_ccf()`, `plot_wf_avg()` | §Widefield |
| Cell assembly stability | `get_stability_among_cell_assemblies()` | §Cell Assemblies |

---

## When to offer plugin conversion

If a script or analysis session shows **two or more** of the following signals, proactively offer to convert it into a `CicadaAnalysis` plugin:

| Signal | Examples |
|---|---|
| Multiple cicada-tools calls chained together | alignment → PSTH → display |
| Parameters the user is iterating or tuning | "let me try threshold=0.3 instead", "what if I change the window?" |
| Running the same logic across multiple sessions | loop over NWB files, aggregating results |
| Interest in saving or reproducing the result | "save this", "I want to rerun this later" |
| Exploration of a parameter space | varying bin size, comparing methods |
| Script is growing beyond ~40 lines of analysis logic | |

**How to offer:**

> "This is getting complex enough that a CicadaAnalysis plugin would be a better fit — it gives you GUI parameter tuning, headless batch execution, and automatic reproducibility via combined YAML. Want me to convert the current script into a plugin?"

If the user agrees, use the plugin shell in the section below. Map every tunable value to the appropriate parameter helper (`add_int_values_arg_for_gui`, `add_bool_option_for_gui`, etc.), and wrap the existing analysis logic inside `run_analysis`. Do not change the logic itself — only restructure it into the plugin pattern.

---

## Writing a CicadaAnalysis plugin

**File placement:**
- Core / shared: `cicada_analysis/core_analysis/cicada_{name}_analysis.py`
- Cossart lab: `cicada_analysis/Cossart_lab_plugins/{Category}/cicada_{name}_analysis.py`
- Petersen lab: `cicada_analysis/Petersen_lab_plugins/{Category}/cicada_{name}_analysis.py`

Class name: `Cicada{Name}Analysis` (PascalCase). Auto-discovered — no registration needed.

### Plugin shell

```python
from cicada_analysis import CicadaAnalysis

class CicadaMyAnalysis(CicadaAnalysis):
    def __init__(self, config_handler=None):
        super().__init__(
            name="My Analysis",              # GUI display name
            short_description="Brief description",
            accepted_data_formats=["CI_DATA"],
            family_id="Category",            # GUI grouping; None = standalone
            long_description="<p>HTML for GUI tooltip</p>",
            config_handler=config_handler
        )

    def copy(self):
        # Required: returns a fresh instance (used by the headless runner)
        return CicadaMyAnalysis(config_handler=self.config_handler)

    def check_data(self):
        # Return False + set self.invalid_data_help to grey out in GUI
        for session in self._data_to_analyse:
            if not session.calcium_imaging.get_roi_response_series_list():
                self.invalid_data_help = "Requires ROI response series"
                return False
        return True

    def setup_parameters(self):
        super().setup_parameters()   # REQUIRED — registers results_path parameter

        # Data selectors (auto-populated from loaded sessions)
        self.add_roi_response_series_arg_for_gui(short_description="ROI Series", family_widget="Data")
        self.add_spiketimes_series_arg_for_gui(short_description="Spike times", family_widget="Data")
        self.add_segmentation_arg_for_gui()
        self.add_ci_movie_arg_for_gui()
        self.add_epoch_dict_arg_for_gui(short_description="Epochs", arg_name="epoch_names")
        self.add_trial_dict_arg_for_gui(short_description="Trials", family_widget="Data")
        self.add_intervals_arg_for_gui(short_description="Intervals", arg_name="interval_names")
        self.add_widefield_acquisitions_dict_for_gui(short_description="WF acquisition", family_widget="Data")

        # Scalar parameters
        self.add_int_values_arg_for_gui(
            arg_name="window_size", min_value=1, max_value=1000, default_value=50,
            short_description="Window (frames)", family_widget="Parameters"
        )
        self.add_field_float_option_for_gui(
            arg_name="threshold", default_value=0.5, mandatory=False,
            short_description="Threshold", family_widget="Parameters"
        )
        self.add_bool_option_for_gui(
            arg_name="normalize", true_by_default=True,
            short_description="Normalize", family_widget="Parameters"
        )
        self.add_field_text_option_for_gui(
            arg_name="label", default_value="",
            short_description="Label", family_widget="Parameters"
        )
        self.add_color_arg_for_gui(arg_name="color", default_value=(1, 0, 0),
                                   short_description="Color")

        # Generic (use when no dedicated helper exists)
        self.add_argument_for_gui(
            arg_name="my_param",
            value_type="str",           # 'str' | 'int' | 'float' | 'bool' | 'file' | 'dir'
            default_value=None,
            short_description="My param",
            mandatory=True,
            choices=["a", "b", "c"],    # optional — restricts to dropdown
            multiple_choices=False,
            family_widget="Group"
        )

        # Group/color selection (for trial condition grouping, Petersen lab style)
        self.add_choices_for_groups_for_gui(
            arg_name="groups", choices=["cond_a", "cond_b"],
            with_color=True, short_description="Groups", family_widget="Group"
        )

        # Output helpers
        self.add_save_formats_arg_for_gui(family_widget="Output")
        self.add_dpi_arg_for_gui(family_widget="Output")
        self.add_verbose_arg_for_gui()
        self.add_with_timestamp_in_filename_arg_for_gui()
        self.add_analysis_description_for_gui("Optional annotation saved with results")

    def run_analysis(self, **kwargs):
        super().run_analysis(**kwargs)  # REQUIRED — sets self._results_path

        roi_series   = kwargs.get('roi_response_series', {})
        window_size  = kwargs.get('window_size', 50)
        save_formats = kwargs.get('save_formats', ['pdf'])

        for session in self.get_data_to_analyse():
            sid = session.identifier
            if sid in roi_series:
                keys = roi_series[sid]
                traces = session.calcium_imaging.get_roi_response_serie_data(keys)
                # ... analysis code ...

        # Save figures:
        # from cicada_analysis.cicada_tools.display.core_plots import save_figure
        # save_figure(fig, self._results_path, "figname", save_formats=save_formats)
```

---

## Headless runner

```python
from cicada_analysis.config.runner import run_from_config, run_from_combined, generate_config

# 1. Generate a filled parameter YAML template
generate_config(
    analysis_class=CicadaMyAnalysis,   # subclass, not an instance
    output_yaml_path="params.yaml",
    sessions_yaml="sessions.yaml",
    result_path="/results",
    overrides={"window_size": 100}     # optional: pre-fill specific parameters
)

# 2a. Run from one params file — returns a single CicadaAnalysis instance
result = run_from_config(
    sessions="sessions.yaml",
    params="params.yaml",
    save_combined=True,                # default: True — writes combined_*.yaml
    extra_plugin_dirs=None,            # list of dirs with external plugins
    results_path=None                  # overrides results_path from the YAML if set
)

# 2b. Run from multiple params files — sessions loaded ONCE, analyses run sequentially
#     Returns a list of CicadaAnalysis instances (one per params file)
results = run_from_config(
    sessions="sessions.yaml",
    params=["params_analysis1.yaml", "params_analysis2.yaml"]
)

# 2c. Alternative: pass a combined YAML directly (equivalent to run_from_combined)
result = run_from_config(combined_config="combined_CicadaMyAnalysis_2026-06-14T10-30-00.yaml")

# 3. Replay an exact previous run from its combined YAML
result = run_from_combined(
    "combined_CicadaMyAnalysis_2026-06-14T10-30-00.yaml",
    save_combined=True,                # default: True
    extra_plugin_dirs=None,
    results_path="/new/output/path"    # redirect outputs without editing YAML
)
```

**sessions.yaml:**
```yaml
- identifier: "mouse1_day1"
  path: "/data/mouse1_day1.nwb"
- identifier: "mouse2_day1"
  path: "/data/mouse2_day1.nwb"
```

**params.yaml:**
```yaml
analysis_class: "CicadaMyAnalysis"
analysis_module: "cicada_analysis.Cossart_lab_plugins.Category.cicada_my_analysis"
results_path: "/results"
roi_response_series: {}
window_size: 50
normalize: true
save_formats: ["pdf", "png"]
```

The **combined_*.yaml** is auto-generated after every run (filename: `combined_{ClassName}_{timestamp}.yaml`). It embeds sessions + class + all final parameter values + CICADA version + run duration. Use `run_from_combined()` to replay exactly. `analysis_module` is optional in params.yaml — the runner auto-discovers the class by searching the built-in plugin roots.

---

## Key constraints

- **No parallel execution on the same NWB files** — `NWBSession` holds HDF5 file handles open. Run analyses sequentially.
- **`super().setup_parameters()` must be called first** in `setup_parameters` — it registers the mandatory `results_path` parameter.
- **`super().run_analysis()` must be called first** in `run_analysis` — it resolves `self._results_path` from kwargs.
- **`copy()` must return a fresh independent instance** — the headless runner calls it to create isolated instances per session group.
