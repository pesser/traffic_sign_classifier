import os, urllib.request, sys, pickle, math, textwrap
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multiprocessing.pool import ThreadPool
import tensorflow as tf
import skimage.transform


# path where data should be placed
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok = True)

# path where images should be placed
img_dir = os.path.join(os.getcwd(), "imgs")
os.makedirs(img_dir, exist_ok = True)

# path to additional test images
test_dir = os.path.join(os.getcwd(), "test_images")


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


def load_test_data():
    """Load additional testing data."""
    X = []
    y = []
    for fname in os.listdir(test_dir):
        label = int(fname.split("_")[0])
        img = plt.imread(os.path.join(test_dir, fname))
        X.append(img)
        y.append(label)
    X = np.stack(X)
    y = np.stack(y)
    return X, y


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


class DataFlow(object):
    """Serve as a batchwise interface to the data. Returns batches of
    preprocessed and augmented data. Furthermore the preprocessing and
    augmentation steps are buffered and run in a new thread. All threads run
    on the same kernel but if we are training on a GPU the preprocessing and
    augmentation will run concurrently with the training process and we thus
    avoid that these operations become the bottleneck of the training
    procedure."""
    
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.img_shape = X.shape[1:]
        self.init_transformations()
        self.pool = ThreadPool(1)
        self.indices = np.arange(self.X.shape[0])
        self.shuffle()
        self._async_next()


    def init_transformations(self):
        """Precompute transformation matrices that are needed for data
        augmentation."""
        shifts = (self.img_shape[0] // 2, self.img_shape[1] // 2)
        unshifts = (-self.img_shape[0] // 2, -self.img_shape[1] // 2)
        self.preshift = skimage.transform.SimilarityTransform(
                translation = shifts)
        self.postshift = skimage.transform.SimilarityTransform(
                translation = unshifts)


    def _async_next(self):
        self.buffer_ = self.pool.apply_async(self._next)


    def _next(self):
        """Get next full batch, handle bookkeeping and shuffling."""
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        if batch_end > self.X.shape[0]:
            self.shuffle()
            return self._next()
        else:
            batch_indices = self.indices[batch_start:batch_end]
            X_batch, y_batch = self.X[batch_indices], self.y[batch_indices]
            X_batch, y_batch = self.process_batch(X_batch, y_batch)
            self.batch_start = batch_end
            return X_batch, y_batch


    def shuffle(self):
        np.random.shuffle(self.indices)
        self.batch_start = 0


    def process_batch(self, X, y):
        """Preprocess and augment batch."""
        # normalize to [-1.0, 1.0]
        X = X / 127.5 - 1.0

        for i in range(X.shape[0]):
            # scaling and bias for contrast and brightness augmentation
            scale = 1.0 + 0.1 * np.random.randn()
            bias = 0.0 + 0.1 * np.random.randn()
            X[i] = np.clip(scale*X[i] + bias, -1.0, 1.0)

            # transformations for geometric augmentations
            angle = 6.0 * np.random.randn()
            zoom = 1 + 0.1 * np.random.randn()
            translation = 2.0 * np.random.randn()
            shear = 0.1 * np.random.randn()

            trafo = skimage.transform.AffineTransform(
                    translation = translation,
                    rotation = np.deg2rad(angle),
                    scale = (zoom, zoom),
                    shear = shear)
            centered_trafo = (self.postshift + (trafo + self.preshift))
            X[i] = skimage.transform.warp(X[i], centered_trafo, mode = "edge", order = 1)
        return X, y


    def get_batch(self):
        """Get a batch and prepare next one asynchronuously."""
        result = self.buffer_.get()
        self._async_next()
        return result


    def batches_per_epoch(self):
        return math.ceil(self.X.shape[0] / self.batch_size)


class DataFlowValid(object):
    """Similiar to DataFlow but for validation/testing data. Does not
    shuffle nor augment."""

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.pool = ThreadPool(1)
        self.batch_start = 0
        self._async_next()


    def _async_next(self):
        self.buffer_ = self.pool.apply_async(self._next)


    def _next(self):
        """Get next batch (possibly not of full batch size) and handle
        bookkeeping."""
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        X_batch, y_batch = self.X[batch_start:batch_end], self.y[batch_start:batch_end]
        X_batch, y_batch = self.process_batch(X_batch, y_batch)
        if batch_end > self.X.shape[0]:
            self.batch_start = 0
        else:
            self.batch_start = batch_end
        return X_batch, y_batch


    def process_batch(self, X, y):
        """Preprocess batch."""
        # normalize to [-1.0, 1.0]
        X = X / 127.5 - 1.0
        return X, y


    def get_batch(self):
        """Get a batch and prepare next one asynchronuously."""
        result = self.buffer_.get()
        self._async_next()
        return result


    def batches_per_epoch(self):
        return math.ceil(self.X.shape[0] / self.batch_size)


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def visualize_augmentation(X, y):
    """Visualize effect of data augmentation."""
    n_samples = 10
    n_augmentations = 9
    Xbatches = DataFlow(X, y, 64)
    indices = Xbatches.indices[:n_samples]
    X_samples = X[indices]
    y_samples = y[indices]
    X_augmented = X_samples / 127.5  - 1.0
    for i in range(n_augmentations):
        x, _ = Xbatches.process_batch(X_samples, y_samples)
        X_augmented = np.concatenate([X_augmented, x])
    X_tiled = tile(X_augmented, n_augmentations + 1, n_samples)
    X_tiled = (X_tiled + 1.0) / 2.0

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    ax.imshow(X_tiled)
    ax.set_axis_off()
    fig.savefig(os.path.join(img_dir, "data_augmentation.png"))


def tf_conv(x, kernel_size, n_features, stride = 1):
    """Pass x through a convolutional layer."""
    # input shape
    x_shape = x.get_shape().as_list()
    assert(len(x_shape) == 4)
    x_features = x_shape[3]

    # weights and bias
    weight_shape = (kernel_size, kernel_size, x_features, n_features)
    weight_stddev = math.sqrt(2.0 / (kernel_size * kernel_size * n_features))
    weight = tf.Variable(
            tf.random_normal(weight_shape, mean = 0.0, stddev = weight_stddev))
    bias = tf.Variable(tf.zeros((n_features,)))

    # operation
    padding = "SAME"
    result = tf.nn.conv2d(
            x, weight,
            strides = (1, stride, stride, 1),
            padding = padding)
    result = tf.nn.bias_add(result, bias)
    return result


def tf_activate(x, keep_prob = 1.0):
    """Activate and drop x."""
    method = "relu"
    x = tf.nn.dropout(x, keep_prob = keep_prob)
    if method == "relu":
        return tf.nn.relu(x)
    elif method == "elu":
        return tf.nn.elu(x)


def tf_downsample(x):
    """Subsample x by a factor of two."""
    method = "stridedconv"
    if method == "maxpool":
        kernel_size = 3
        stride = 2
        padding = "SAME"
        return tf.nn.max_pool(
                x,
                ksize = (1, kernel_size, kernel_size, 1),
                strides = (1, stride, stride, 1),
                padding = padding)
    elif method == "stridedconv":
        x_shape = x.get_shape().as_list()
        assert(len(x_shape) == 4)
        x_features = x_shape[3]
        return tf_conv(x, 3, x_features, stride = 2)


def tf_flatten(x):
    """Flatten x to prepare as input for fc layer."""
    return tf.contrib.layers.flatten(x)


def tf_fc(x, n_features):
    """Pass x through fully connected layer."""
    # input shape
    x_shape = x.get_shape().as_list()
    assert(len(x_shape) == 2)
    x_features = x_shape[1]

    # weights and bias
    weight_shape = (x_features, n_features)
    weight_stddev = math.sqrt(2.0 / n_features)
    weight = tf.Variable(
            tf.random_normal(weight_shape, mean = 0.0, stddev = weight_stddev))
    bias = tf.Variable(tf.zeros((n_features,)))

    # operation
    result = tf.matmul(
            x, weight)
    result = tf.add(result, bias)
    return result


class TSCModel(object):
    """Traffic Sign Classifier. Expects batches with images normalized to
    [-1, 1]. Uses a simple architecture with four blocks of
    convolution-activation-downsampling followed by a two layer, fully
    connected classifier. Cross entropy is minimized by Adam with a learning
    rate decaying linearly from 1e-3 to 1e-8 in the specified number of
    steps."""

    def __init__(self, img_shape, n_labels, n_total_steps):
        self.session = tf.Session()
        self.img_shape = img_shape
        self.n_labels = n_labels
        self.n_total_steps = n_total_steps
        self.initial_learning_rate = 1e-3
        self.end_learning_rate = 0.0
        self.log_frequency = 250
        self.logs = dict()
        self.define_graph()

    
    def define_graph(self):
        """Define tensorflow graph."""
        img_batch_shape = (None,) + self.img_shape
        label_batch_shape = (None,)
        # inputs
        self.x = tf.placeholder(tf.float32, shape = img_batch_shape)
        self.y = tf.placeholder(tf.int32, shape = label_batch_shape)
        self.keep_prob = tf.placeholder(tf.float32)

        kernel_size = 3
        n_channels = 32
        # architecture
        features = self.x
        for i in range(5):
            features = tf_conv(features, kernel_size, (i+1)*n_channels)
            features = tf_activate(features, keep_prob = self.keep_prob)
            features = tf_downsample(features)
        features = tf_flatten(features)
        features = tf_fc(features, 128)
        features = tf_activate(features, keep_prob = self.keep_prob)
        features = tf_fc(features, 64)
        features = tf_activate(features, keep_prob = self.keep_prob)
        self.logits = tf_fc(features, self.n_labels)

        # loss
        self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.y,
                logits = self.logits))
        # accuracy
        correct_logits = tf.equal(
                tf.cast(tf.argmax(self.logits, axis = 1), tf.int32),
                self.y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_logits, tf.float32))
        # categorical probabilities
        self.categorical_probabilities = tf.nn.softmax(self.logits)

        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate = tf.train.polynomial_decay(
                learning_rate = self.initial_learning_rate,
                global_step = self.global_step,
                decay_steps = self.n_total_steps,
                end_learning_rate = self.end_learning_rate,
                power = 1.0)

        optimizer = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate)
        self.train_op = optimizer.minimize(
                self.loss_op,
                global_step = self.global_step)

        # init
        self.session.run(tf.global_variables_initializer())


    def fit(self, batches, batches_valid):
        """Fit model to dataset and evaluate on validation set."""
        self.batches = batches
        self.batches_valid = batches_valid
        for batch in range(self.n_total_steps):
            X_batch, y_batch = batches.get_batch()
            feed_dict = {
                    self.x: X_batch,
                    self.y: y_batch,
                    self.keep_prob: 0.85}
            fetch_dict = {
                    "train": self.train_op,
                    "loss": self.loss_op,
                    "accuracy": self.accuracy_op,
                    "global_step":  self.global_step,
                    "learning_rate": self.learning_rate}
            result = self.session.run(fetch_dict, feed_dict)
            self.log_training(batch, total_batches, result)


    def log_training(self, batch, total_batches, result):
        """Keep average of training metrics and log them together with
        validation metrics."""
        metrics = ["loss", "accuracy"]
        for metric in metrics:
            if metric not in self.logs:
                self.logs[metric] = []
            self.logs[metric].append(result[metric])
        if batch % self.log_frequency == 0 or batch + 1 == total_batches:
            print("Batch {} / {} = {:.2f} %".format(batch, total_batches, 100 * batch / total_batches))
            print("{:20}: {}".format("Global step", result["global_step"]))
            print("{:20}: {:.4e}".format("Learning rate", result["learning_rate"]))
            for metric in metrics:
                metric_logs = self.logs[metric]
                average = sum(metric_logs) / len(metric_logs)
                print("{:20}: {:.4}".format("Training " + metric, average))
                self.logs[metric] = []
            val_metrics = self.evaluate(self.batches_valid)
            for k, v in val_metrics.items():
                print("{:20}: {:.4}".format("Validation " + k, v))


    def evaluate(self, batches):
        """Evaluate metrics averaged over one epoch."""
        total_batches = batches.batches_per_epoch()
        logs = dict()
        for batch in range(total_batches):
            X_batch, y_batch = batches.get_batch()
            feed_dict = {
                    self.x: X_batch,
                    self.y: y_batch,
                    self.keep_prob: 1.0}
            fetch_dict = {
                    "loss": self.loss_op,
                    "accuracy": self.accuracy_op}
            result = self.session.run(fetch_dict, feed_dict)
            for metric in result:
                if not metric in logs:
                    logs[metric] = []
                logs[metric].append(result[metric])
        for metric in logs:
            logs[metric] = sum(logs[metric]) / len(logs[metric])
        return logs


    def evaluate_probabilities(self, batches):
        """Evaluate categorical probabilities for each sample."""
        total_batches = batches.batches_per_epoch()
        catprobs = []
        for batch in range(total_batches):
            X_batch, y_batch = batches.get_batch()
            feed_dict = {
                    self.x: X_batch,
                    self.y: y_batch,
                    self.keep_prob: 1.0}
            fetch_dict = {
                    "catprobs": self.categorical_probabilities}
            result = self.session.run(fetch_dict, feed_dict)
            catprobs.append(result["catprobs"])
        catprobs = np.concatenate(catprobs)
        return catprobs


def visualize_test_results(X, y, pred, signnames):
    """Visualize images and their predicted class probabilities."""
    assert(X.shape[0] == 14)
    nrows = 2
    ncols = 7
    nlabels = 43
    fig, axes = plt.subplots(nrows = 2 * nrows, ncols = ncols, figsize = (10, 10))
    for i in range(nrows):
        for j in range(ncols):
            aximg = axes[2*i, j]
            axprobs = axes[2*i + 1, j]
            idx = i*ncols + j

            img = X[idx]
            aximg.imshow(img)
            aximg.set_axis_off()

            probs = pred[idx]
            label = y[idx]
            colors = probs.shape[0] * ["red"]
            colors[label] = "green"

            n_top = 5
            topindices = sorted(np.arange(probs.shape[0]), key = lambda i: probs[i])[-n_top:]
            topprobs = probs[topindices]
            topcolors = [colors[i] for i in topindices]
            ypos = np.arange(n_top)
            axprobs.barh(ypos, topprobs, color = topcolors)
            axprobs.set_yticks(ypos)
            for ypos, l in zip(ypos, topindices):
                axprobs.text(0.025, ypos, textwrap.fill(signnames[l], 20), fontsize = 6)
            axprobs.set_axis_off()
    fig.savefig(os.path.join(img_dir, "test_results.png"))


if __name__ == "__main__":
    files = download_data()
    data = extract_data(files["traffic-signs-data.zip"])
    data = load_data(data)
    signnames = load_signnames(files["signnames.csv"])

    basic_summary(data)
    visualize_class_examples(data, signnames)
    visualize_label_distributions(data, signnames)
    most_common_labels(data, signnames)
    
    X, y = data["train"]
    visualize_augmentation(X, y)

    batch_size = 64
    batches = DataFlow(X, y, batch_size)
    X_valid, y_valid = data["valid"]
    batches_valid = DataFlowValid(X_valid, y_valid, batch_size)

    img_shape = X.shape[1:]
    n_labels = np.unique(y).shape[0]
    n_epochs = 100
    total_batches = batches.batches_per_epoch() * n_epochs
    model = TSCModel(img_shape, n_labels, total_batches)
    model.fit(batches, batches_valid)

    final_run = True
    if final_run:
        # finally test on original test set
        X_test, y_test = data["test"]
        batches_test = DataFlowValid(X_test, y_test, batch_size)
        metrics = model.evaluate(batches_test)
        for k, v in metrics.items():
            print("{:20}: {:.4}".format("Testing " + k, v))

        # and on a few additional test images captured in the wild
        X_test, y_test = load_test_data()
        batches_test = DataFlowValid(X_test, y_test, batch_size)
        metrics = model.evaluate(batches_test)
        for k, v in metrics.items():
            print("{:20}: {:.4}".format("Additional Testing " + k, v))
        # get categorical probabilities for visualization
        catprobs = model.evaluate_probabilities(batches_test)
        visualize_test_results(X_test, y_test, catprobs, signnames)
