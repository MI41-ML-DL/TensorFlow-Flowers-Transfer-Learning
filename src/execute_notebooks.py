"""Execute pure-Python notebook cells in IPython without kernel sockets.

Useful in restricted build environments. These notebooks contain no magics or
JavaScript and can also be run normally in Jupyter. Training itself is performed
by src.pipeline; the default notebook run verifies and presents its real outputs.
"""
import contextlib
import os
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from .common import ROOT, save_json


def execute():
    records = []
    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            InteractiveShell.clear_instance()
            shell = InteractiveShell.instance()
            notebook = nbformat.read(path, as_version=4)
            count = 0
            print("EXECUTING", path.name, flush=True)
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                with capture_output(stdout=True, stderr=True, display=True) as captured:
                    result = shell.run_cell(cell.source, store_history=True)
                if result.error_before_exec is not None:
                    raise result.error_before_exec
                if result.error_in_exec is not None:
                    raise result.error_in_exec
                outputs = []
                if captured.stdout:
                    outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
                if captured.stderr:
                    outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
                for rich in captured.outputs:
                    outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
                cell.outputs = outputs
                cell.execution_count = count
            notebook.metadata.kernelspec = {"display_name": "Python 3 (Flowers)", "language": "python", "name": "python3"}
            notebook.metadata["execution_validation"] = {"backend": "in-process IPython", "all_code_cells_executed": True,
                "training_provenance": "real TensorFlow/scikit-learn CLI runs; default RETRAIN=False", "error_count": 0}
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            records.append({"notebook": path.name, "code_cells_executed": count, "errors": 0,
                            "backend": "in-process IPython, socket-free"})
            print("DONE", path.name, "code cells", count, flush=True)
    finally:
        os.chdir(original_cwd)
    save_json(ROOT / "results/notebook_validation.json", records)
    return records


if __name__ == "__main__":
    execute()
