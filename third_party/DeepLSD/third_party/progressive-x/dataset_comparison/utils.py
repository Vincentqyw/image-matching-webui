import os.path
from os import path
from urllib.request import urlopen
from zipfile import ZipFile
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import math

def convert_to_number (s):
    return int.from_bytes(s.encode(), 'little')

def convert_from_number (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

def load_data(dataset):
    files = os.listdir(dataset)
    data = {}

    for f in files:
        try:
            M = np.loadtxt(f"{dataset}/{f}/{f}.txt", delimiter=" ")
            data[f] = {}
            data[f]['corrs'] = np.concatenate((M[:,:2], M[:,3:5]), axis=1)
            data[f]['labels'] = M[:,-1]
        except:
            print(f"Error when loading scene {f}")
    return data

def download_datasets(url_base, datasets):
    # Download the dataset if needed
    for dataset in datasets:
        if not path.exists(dataset):
            url = f'{url_base}{dataset}.zip'
            # Download the file from the URL
            print(f"Beginning file download '{url}'")
            zipresp = urlopen(url)
            # Create a new file on the hard drive
            tempzip = open("/tmp/tempfile.zip", "wb")
             # Write the contents of the downloaded file into the new file
            tempzip.write(zipresp.read())
                # Close the newly-created file
            tempzip.close()
                # Re-open the newly-created file with ZipFile()
            zf = ZipFile("/tmp/tempfile.zip")
            # Extract its contents into <extraction_path>
            # note that extractall will automatically create the path
            zf.extractall(path = '')
            # close the ZipFile instance
            zf.close()

def misclassification(segmentation, ref_segmentation):
    n = int(max(ref_segmentation)) + 1
    indices = np.array(range(n))
    n_labels = len(segmentation)
    miss = []

    for p in multiset_permutations(indices):
        tmp_ref_segmentation = np.zeros((n_labels))

        for i in range(n):
            indices = ref_segmentation == i
            tmp_ref_segmentation[indices] = p[i]

        misclassified_points = np.sum(tmp_ref_segmentation != segmentation)
        miss.append(misclassified_points)
   
    return np.min(miss) / n_labels