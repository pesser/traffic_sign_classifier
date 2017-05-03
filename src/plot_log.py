import sys, os
import matplotlib.pyplot as plt


plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok = True)


def plot_data(data, logfname):
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
    keys = (["Training loss", "Validation loss"],
            ["Training accuracy", "Validation accuracy"])
    labels = ("loss", "accuracy")
    for i in range(2):
        ax = axes[i]
        for key in keys[i]:
            ax.plot(data["Batch"], data[key], label = key)
        ax.set_ylabel(labels[i])
        ax.set_xlabel("batch")
        ax.legend()
    fig.savefig(os.path.join(plot_dir, logfname + ".png"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Useage: {} logfile".format(sys.argv[0]))
        exit(1)
    logfname = sys.argv[1]

    data = {
            "Batch": [],
            "Training loss": [],
            "Training accuracy": [],
            "Validation loss": [],
            "Validation accuracy": []}
    with open(logfname, "r") as f:
        for line in f:
            for k in data:
                if line.startswith(k):
                    if k == "Batch":
                        value = int(line.split()[1])
                    else:
                        value = float(line.split()[3])
                    data[k].append(value)

    plot_data(data, logfname)
