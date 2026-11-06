# Faulted Surface Gridding

A QGIS Processing plugin for interpolating scattered horizon points onto a
regular grid while respecting fault discontinuities.

The plugin provides two algorithms:

- **Local Gridding** – fast tanh-distance weighted interpolation onto nodes
  near the input points.
- **Iterative Gridding** – local gridding followed by multi-scale
  second-derivative least-squares interpolation, based on Léger & Clochard
  (2020).

Both write a node-centred GeoTIFF with `NaN` as the NoData value.

## Requirements

- **QGIS 3.40 or newer** (3.40+ and 4.x are supported;
  `qgisMaximumVersion` is 4.99).
- **NumPy** – included with QGIS.
- **SciPy** – required by **Iterative Gridding** only.
  **Local Gridding** works without it.

### Installing SciPy into QGIS's Python

SciPy is not bundled with QGIS. Install it into the Python interpreter QGIS
uses, not into a separate system Python or virtual environment, then restart
QGIS.

**Windows – OSGeo4W install (recommended, cleanest):**

1. Run **OSGeo4W Setup** → *Advanced Install*.
2. Search for **`python3-scipy`** (category *Libs*) and install it.
3. Restart QGIS.

**Windows – standalone QGIS installer:**

Open an elevated Command Prompt and run:

```bat
"C:\Program Files\QGIS 3.40.0\bin\python-qgis-ltr.bat" -m pip install scipy
```

Use `python-qgis.bat` instead of `python-qgis-ltr.bat` for non-LTR builds.
For a per-user install, add `--user`. If pip is missing:

```bat
"C:\Program Files\QGIS 3.40.0\bin\python-qgis-ltr.bat" -m ensurepip --upgrade
```

**Linux:**

Use your distribution package, for example:

```bash
sudo apt install python3-scipy        # Debian/Ubuntu
sudo dnf install python3-scipy        # Fedora
sudo pacman -S python-scipy           # Arch
```

Or install into QGIS's Python with `python3 -m pip install scipy`.
For the Flatpak build, see the QGIS installation guide for the
`flatpak run --devel --command=pip3 org.qgis.qgis install scipy --user`
workaround.

**macOS:**

QGIS ships its own Python. Install into that interpreter rather than a
separate system Python, for example:

```bash
/path/to/QGIS.app/Contents/MacOS/bin/python3 -m pip install scipy
```

See QGIS's installation guide for the exact path for your build.

**Conda/mamba QGIS:**

```bash
micromamba install -c conda-forge scipy
```

Verify the installation:

```bat
"C:\Program Files\QGIS 3.40.0\bin\python-qgis-ltr.bat" -c "import scipy; print(scipy.__version__)"
```

> **Note:** `pip install scipy` can upgrade NumPy and affect other packages in
> the QGIS environment. The OSGeo4W `python3-scipy` package avoids that risk.

## Installation

### From a ZIP file

1. Build or obtain `faulted_surface_gridder-<version>.zip`.
2. In QGIS open **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Select the ZIP and click **Install Plugin**.
4. Enable the plugin if it is not enabled automatically.

The ZIP must contain a single top-level folder with `metadata.txt` and
`__init__.py` inside it:

```text
faulted_surface_gridder/
├── __init__.py
├── metadata.txt
├── plugin.py
├── provider.py
└── algorithms/
    └── ...
```

Do **not** zip the *contents* of the folder directly. QGIS locates the plugin
by finding `metadata.txt` inside the archive and uses its parent directory as
the plugin name.

Build a clean ZIP from the repository root:

```bash
mkdir -p dist
zip -r dist/faulted_surface_gridder-1.0.0.zip faulted_surface_gridder \
  -x '*/__pycache__/*' '*.pyc' '*.pyo' '*/.DS_Store'
```

### From source (development)

Copy or symlink the `faulted_surface_gridder` folder into your QGIS profile's
Python plugin directory:

| Platform | Default plugin directory |
| --- | --- |
| Linux | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |
| Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` |
| macOS | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |

Then restart QGIS and enable **Faulted Surface Gridding** under
**Plugins → Manage and Install Plugins → Installed**.

## Usage

Open **Processing → Toolbox**, expand **Gridding Methods**, and choose either
**Local Gridding** or **Iterative Gridding**.

### Shared parameters

| Parameter | Description |
| --- | --- |
| **Horizon points (XYZ)** | Point layer containing the input samples. |
| **Point value field** | Numeric attribute holding the horizon value (`Z`). |
| **Fault traces** | Optional line layer. Traces become vertical discontinuities. |
| **Fault heave polygons** | Optional polygon layer. Polygon interiors become NoData. |
| **Grid extent** | Extent and CRS of the output grid. |
| **Grid size** | Node spacing in layer units (default `100`). |
| **Clip output to convex hull** | If enabled, nodes outside the samples' convex hull are NoData. |
| **Output Grid** | Output GeoTIFF path. |

Fault traces and heave polygons are optional and may be used together.

### Iterative Gridding parameters

| Parameter | Default | Description |
| --- | --- | --- |
| **Solver iterations at the finest scale** | `100` | Maximum LSMR iterations on the finest grid. |
| **Multi-scale levels** | `0` | Number of scales; `0` selects automatically (up to 6). |
| **Solver tolerance** | `1e-6` | Convergence tolerance for LSMR. |

### Output

The result is a single-band Float32 GeoTIFF:

- Values are samples at the grid nodes.
- The geotransform is shifted by half a cell so that each pixel centre falls
  exactly on a grid node.
- NoData is `NaN`.
- Fault-trace nodes, heave-polygon interiors, and (optionally) nodes outside
  the convex hull are NoData.

## How it works

### Stage 1 – Local Gridding

For each grid node, points within a one-grid-step square window are gathered
with a QGIS KDBush spatial index. Nodes are filled with a tanh-distance
weighted mean:

```text
w(d) = tanh(d) / d,  with w(0) = 1
```

Only samples in the same fault block contribute, so values do not leak across
faults. Nodes on fault traces, inside heave polygons, or outside the convex
hull are masked out.

### Stage 2 – Iterative second-derivative optimisation

The remaining valid nodes are filled by minimising the squared second
derivatives of the surface:

- `d²/dx²` and `d²/dy²` have weight `1`.
- `d²/dxdy` has weight `√2`, following Eq. (6) of the paper.

The system is assembled as a sparse least-squares problem and solved with
`scipy.sparse.linalg.lsmr`. Stencils that cross a fault edge or use an
inactive node are omitted, which decouples the fault blocks exactly as in the
paper. Stage-1 values are hard constraints.

Fault blocks are handled independently:

- A block with **fewer than three independent known nodes** is filled with its
  mean value and reported as a warning.
- A block with **no known nodes** remains NoData and is reported as a warning.

### Multi-scale refinement

The surface is solved coarse-to-fine. Coarse grids use every second (then
every fourth, etc.) node of the fine grid, so their nodes are always a subset
of the fine lattice. Each coarse result is bilinearly upsampled to initialise
the next finer solve.

Iterations are multiplied by the level factor (`iterations * 2**level`), so
coarse levels receive proportionally more work for the large-scale structure.
With `Multi-scale levels = 0`, the algorithm chooses up to six levels based on
the grid size.

## Troubleshooting

**"The Iterative Gridding algorithm requires SciPy..."**

SciPy is not installed in the Python interpreter QGIS is using, or a broken
SciPy installation could not be imported. Follow the
[SciPy installation steps](#installing-scipy-into-qgiss-python) and restart
QGIS. Verify with:

```bat
"...\bin\python-qgis-ltr.bat" -c "import scipy; print(scipy.__version__)"
```

**The plugin does not appear after installing the ZIP**

Check that the ZIP contains `faulted_surface_gridder/metadata.txt` with a
top-level folder, and that QGIS is version 3.40 or newer. Then enable it under
**Plugins → Manage and Install Plugins → Installed**.

**The output is entirely NoData**

- Check that the grid extent overlaps the input points.
- Check that the point value field contains finite numeric values.
- If convex-hull clipping is enabled, points outside the hull are NoData.
- Fault blocks that contain no data remain NoData; the Processing log reports
  a warning for each such block.

**The iterative solve stops at the iteration limit**

Increase **Solver iterations at the finest scale**. The log reports the number
of LSMR iterations and the stop code (`istop`), and warns when the limit is
reached before convergence.

## Development

The repository uses [pixi](https://pixi.sh/) with a conda-forge QGIS
environment:

```bash
pixi install
pixi run pytest
pixi run qgis          # launch QGIS with the environment
```

Run the tests directly with:

```bash
pixi run python -m pytest -q
```

The tests use an offscreen QGIS application and do not require a display.

### Project layout

```text
faulted_surface_gridder/
├── metadata.txt                 Plugin metadata
├── __init__.py                  classFactory entry point
├── plugin.py                    Plugin registration/unregistration
├── provider.py                  Processing provider
└── algorithms/
    ├── local_gridder.py         Local Gridding algorithm
    ├── local_gridder_impl.py    KDBush gather + tanh-IDW
    ├── iterative_gridder.py     Iterative Gridding orchestration
    ├── iterative_gridder_impl.py Sparse LSMR solver and multi-scale helpers
    ├── gridding_context.py      Shared input loading and per-level setup
    ├── fault_blocks.py          Fault trace rasterisation and block labelling
    ├── gdal_helpers.py          Heave rasterisation and GeoTIFF writing
    └── helpers.py               GridSpec, layer access, convex hull mask
```

## Limitations

- Faults are treated as **vertical discontinuities** at the trace location;
  inclined fault geometry is not modelled.
- Fault blocks with insufficient data are either filled with their mean or
  left as NoData, with a warning.
- The iterative solver may need more than the default 100 fine-grid iterations
  for large or strongly faulted grids. Increase **Solver iterations** if the
  log reports that LSMR reached its limit.
- SciPy must be installed separately for Iterative Gridding (see above).

## Citation

If you use the iterative method in published work, please cite:

> Léger, M. & Clochard, V. (2020). Fast interpolation method for surfaces with
> faults by multi-scale second-derivative optimization. *Oil & Gas Science and
> Technology – Rev. IFP Energies nouvelles*, 75, 62.
> <https://doi.org/10.2516/ogst/2020057>

## License

MIT License. See [LICENSE](LICENSE).
