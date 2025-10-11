from typing import List
import numpy as np
import os
from icecream import ic
import matplotlib.pyplot as plt


def read_in_data(work_path: str, searched_extension: str) -> List:
    """Returns a list with the file names in a given path

    Args:
        work_path (str): path where to perform the search
        searched_extension (str): file extention to search for

    Returns:
        List: List with the file names
    """
    files = []
    for file in os.listdir(work_path):
        path = os.path.join(work_path, file)
        if os.path.isdir(path):
            continue
        elif file.endswith(".txt.dsc"):
            continue
        elif file.endswith(searched_extension):
            files.append(file)
        else:
            continue
    return files


work_path = "./data/W-Spectra/"
files = np.array(read_in_data(work_path, ".txt"))

first_file_idx = np.where(files == "data_9-10.txt")[0][0]
data = np.loadtxt(os.path.join(work_path, files[first_file_idx]), delimiter=",")
all_data = np.zeros((data.shape[0], 93))
all_data[:, 0:3] = data.copy()
files = np.delete(files, first_file_idx)

data_labels = [f"{i}" for i in range(8, 101)]
data_labels[0] = "Energy[keV]"

i0 = 2
i1 = 2  # Last energy added

for f in files:
    # Adds simulated spectra
    if f.find("data") >= 0:
        data = np.loadtxt(os.path.join(work_path, f), delimiter=",")
        p0 = f.find("_") + 1
        p1 = f.find("-")

        e0 = int(f[p0:p1])
        e1 = int(f[p1 + 1 : -4])
        i0 += e1 - e0
        all_data[:, i0] = data[:, -1].copy()
    # Adds interpolated spectra
    else:
        data = np.loadtxt(os.path.join(work_path, f), delimiter="\t")
        p0 = f.find("-")
        e0 = int(f[2:p0])
        e1 = int(f[p0 + 1 : -4])

        idx0 = i1 + 1
        idx1 = idx0 + data.shape[1]  # End index should be start + number of columns
        ic(
            f"e0={e0}, e1={e1}, data shape={data.shape}, assigning to columns {idx0}:{idx1}"
        )
        i1 = idx1

        new_shape = (int(all_data.shape[0] - data.shape[0]), data.shape[1])

        if new_shape[0] != 0:
            padded_data = np.concatenate((data, np.ones(new_shape) * 1e-35), axis=0)
            if e0 in [15, 20, 25]:
                all_data[:, idx0:idx1] = padded_data.copy()[:, ::-1]
            else:
                all_data[:, idx0:idx1] = padded_data.copy()
        else:
            all_data[:, idx0:idx1] = data.copy()

# Export to CSV
output_file = os.path.join(work_path, "Spectra_9-100.csv")
ic(f"Exporting data to {output_file}")
ic(f"Data shape: {all_data.shape}")
ic(f"Number of labels: {len(data_labels)}")

# Save with header
np.savetxt(
    output_file,
    all_data,
    delimiter=",",
    header=",".join(data_labels),
    comments="",
    fmt="%.6e"
)

ic(f"Export complete!")
