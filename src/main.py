import os, urllib.request, sys, pickle, math, textwrap
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# path where data should be placed
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok = True)

# path where images should be placed
img_dir = os.path.join(os.getcwd(), "imgs")
os.makedirs(img_dir, exist_ok = True)


def dl_progress(count, block_size, total_size):
    """Progress bar used during download."""
    length = 50
    current_size = count * block_size
    done = current_size * length // total_size
    togo = length - done
    prog = "[" + done * "=" + togo * "-" + "]"
    sys.stdout.write(prog)
    if(current_size < total_size):
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def download_data():
    """Download required data."""
    data = {"traffic-signs-data.zip": "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip",
            "signnames.csv": "https://raw.githubusercontent.com/udacity/CarND-Traffic-Sign-Classifier-Project/master/signnames.csv"}
    local_data = {}
    for fname, url in data.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print("Downloading {}".format(fname))
            urllib.request.urlretrieve(url, path, reporthook = dl_progress)
        else:
            print("Found {}. Skipping download.".format(fname))
        local_data[fname] = path
    return local_data


def extract_data(path):
    """Extract zip file if not already extracted."""
    with ZipFile(path) as f:
        targets = dict((fname, os.path.join(data_dir, fname)) for fname in f.namelist())
        if not all([os.path.exists(target) for target in targets.values()]):
            print("Extracting {}".format(path))
            f.extractall(data_dir)
        else:
            print("Skipping extraction of {}".format(path))
    return targets


def load_data(paths):
    """Load data from pickled files."""
    result = dict()
    for split in ["train", "valid", "test"]:
        fname = split + ".p"
        path = paths[fname]
        with open(path, "rb") as f:
            data = pickle.load(f)
        result[split] = (data["features"], data["labels"])
    return result


def load_signnames(path):
    """Return mapping from class index to class description."""
    data = np.genfromtxt(path, dtype = None, delimiter = ",", skip_header = 1)
    return dict((i, v.decode("utf-8")) for i, v in data)


def table_format(row, header = False, width = 10):
    """Format row as markdown table."""
    result = "|" + "|".join(str(entry).center(width) for entry in row) + "|"
    if header:
        l = len(result)
        result = result + "\n" + "|" + (l-1) * "-"
    return result


def basic_summary(data):
    """Summarize size of data for each split."""
    headers = ["Split", "Samples", "Height", "Width", "Channels", "Classes"]
    print(table_format(headers, header = True))
    for split in ["train", "valid", "test"]:
        X, y = data[split]
        n, h, w, c = X.shape
        n_classes = np.unique(y).shape[0]
        row = [split, n, h, w, c, n_classes]
        print(table_format(row))


def visualize_class_examples(data, signnames):
    """Visualize examples for each class."""
    n_classes = len(signnames)
    nrows = ncols = round(math.sqrt(n_classes))
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10,10))
    # use examples from the training split
    X, y = data["train"]
    examples = []
    for label in range(n_classes):
        examples.append(X[y == label][0])
    for i in range(nrows):
        for j in range(ncols):
            idx = i*ncols + j
            ax = axes[i, j]
            ax.set_axis_off()
            if idx < n_classes:
                img = examples[i*ncols + j]
                ax.imshow(img)
                # wrapped and center aligned label description
                ax.text(-10.0, -4.0,
                        "\n".join(line.center(16) for line in textwrap.wrap(signnames[idx], 16)),
                        fontsize = 9)
    fig.subplots_adjust(hspace = 1.25)
    fig.savefig(os.path.join(img_dir, "class_examples.png"))


def visualize_label_distributions(data, signnames):
    """Visualize distribution of labels."""
    n_classes = len(signnames)
    n_splits = len(data)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    width = 0.8 / n_splits
    for i, split in enumerate(data.keys()):
        X, y = data[split]
        labels, counts = np.unique(y, return_counts = True)
        fraction = counts / y.shape[0]
        ax.bar(labels + i*width, fraction, width, label = split)
    ax.hlines(1.0/n_classes, *(ax.get_xlim()), label = "uniform distribution")
    ax.set_xlabel("label")
    ax.set_ylabel("fraction")
    ax.set_title("Label Distributions")
    ax.legend()
    fig.savefig(os.path.join(img_dir, "class_distributions.png"))


def most_common_labels(data, signnames):
    """Show which labels are most and least common."""
    n_classes = len(signnames)
    n_splits = len(data)
    X, y = data["train"]
    labels, counts = np.unique(y, return_counts = True)
    fractions = counts / y.shape[0]
    sorted_indices = sorted(np.arange(n_classes), key = lambda i: fractions[i])
    sorted_labels = labels[sorted_indices]
    sorted_fractions = fractions[sorted_indices]

    width = 30
    print(table_format(["Fraction", "Traffic Sign"], header = True, width = width))
    for i in range(3):
        l = signnames[sorted_labels[-1 - i]]
        f = "{:.2f} %".format(100.0*sorted_fractions[-1 - i])
        print(table_format([f, l], width = width))
    for i in range(3):
        l = signnames[sorted_labels[i]]
        f = "{:.2f} %".format(100.0*sorted_fractions[i])
        print(table_format([f, l], width = width))


if __name__ == "__main__":
    files = download_data()
    data = extract_data(files["traffic-signs-data.zip"])
    data = load_data(data)
    signnames = load_signnames(files["signnames.csv"])

    basic_summary(data)
    visualize_class_examples(data, signnames)
    visualize_label_distributions(data, signnames)
    most_common_labels(data, signnames)
