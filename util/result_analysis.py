import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from util.static_variable import unique_labels


def make_markdown_table(array):
    """ Convert the array-like classification report into a markdown table """

    nl = "\n"

    markdown = nl
    markdown += f"| {' | '.join(array[0])} |"

    markdown += nl
    markdown += f"| {' | '.join(['---'] * len(array[0]))} |"

    markdown += nl
    for entry in array[1:]:
        markdown += f"| {' | '.join(entry)} |{nl}"
    return markdown


class ClassificationSummary:
    """
    Analyze the classification results with a class-domain table report and a confusion matrix.

    Args:
        class_label (str): Class label. (default: ``scene``)
        domain_label (str): Domain label. (default: ``device``)
    """
    def __init__(self, class_label='scene', domain_label='device'):
        self.class_labels = unique_labels[class_label]
        self.domain_labels = unique_labels[domain_label]

    def get_table_report(self, inputs):
        _y_true = inputs['y']
        _y_pred = inputs['pred']
        _d_indices = inputs['d']
        # Convert device indices to device labels
        d = [self.domain_labels[i] for i in _d_indices]
        # Create a dictionary to store the class-wise accuracy for each domain
        domain_class_accuracy = {}
        for domain_label in self.domain_labels:
            indices = [i for i, x in enumerate(d) if x == domain_label]
            domain_true = [_y_true[i] for i in indices]
            domain_pred = [_y_pred[i] for i in indices]
            class_accuracy = {}
            for class_index in set(domain_true):
                indices = [i for i, x in enumerate(domain_true) if x == class_index]
                class_true = [domain_true[i] for i in indices]
                class_pred = [domain_pred[i] for i in indices]
                class_accuracy[class_index] = accuracy_score(class_true, class_pred)
            domain_class_accuracy[domain_label] = class_accuracy
        # Create a table-like output of domain-wise and class-wise accuracy
        column_names = ["Class"] + self.domain_labels + ["Class Avg."]
        classification_report = [column_names]
        for class_label in self.class_labels:
            row = [str(class_label)]
            class_index = self.class_labels.index(class_label)
            class_ttl_acc = 0.0
            for domain_label in self.domain_labels:
                if class_index in domain_class_accuracy[domain_label]:
                    row.append(f"{domain_class_accuracy[domain_label][class_index] * 100:.1f}")
                    class_ttl_acc += domain_class_accuracy[domain_label][class_index]
                else:
                    row.append("N/A")
            class_avg_acc = class_ttl_acc / len(self.domain_labels)
            classification_report.append(row + [f"{class_avg_acc * 100:.1f}"])
        # Add a row that shows the macro average accuracy across all domains and class_labels
        num_classes = len(self.class_labels)
        total_accuracy = 0.0
        domain_avg_row = ['Domain Avg.']
        for domain_label in self.domain_labels:
            domain_accuracy = 0.0
            domain_class_count = 0
            for class_label in range(num_classes):
                if class_label in domain_class_accuracy[domain_label]:
                    domain_accuracy += domain_class_accuracy[domain_label][class_label]
                    domain_class_count += 1
            if domain_class_count > 0:
                domain_accuracy /= domain_class_count
                total_accuracy += domain_accuracy
            else:
                domain_accuracy = "N/A"
            domain_avg_row.append(f"{domain_accuracy * 100:.1f}")
        total_accuracy /= len(self.domain_labels)
        classification_report.append(domain_avg_row + [f"{total_accuracy * 100:.1f}"])
        markdown = make_markdown_table(classification_report)
        return markdown

    def get_confusion_matrix(self, inputs):
        _y_true = inputs['y']
        _y_pred = inputs['pred']
        # Compute confusion matrix
        cm = confusion_matrix(_y_true, _y_pred)
        # Convert to probability confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(8, 8))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]))
        ax.set_xticklabels(self.class_labels, fontsize=12)
        ax.set_yticklabels(self.class_labels, fontsize=12)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig