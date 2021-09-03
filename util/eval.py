import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(truth, pred, out_file, norm=None):
    label_names = list(set(truth) | set(pred))
    label_names.sort()
    truth_compact = [label_names.index(x) for x in truth]
    pred_compact = [label_names.index(x) for x in pred]
    cm = confusion_matrix(
        truth_compact, pred_compact, labels=list(range(len(label_names))),
        normalize=norm)
    if norm is not None:
        cm *= 100
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation='vertical',
              values_format='.1f' if norm is not None else 'd')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
