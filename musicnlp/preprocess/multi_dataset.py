"""
An implementation of T5's mixing multiple tasks/datasets taken from
     https://github.com/huggingface/datasets/issues/217#issuecomment-650346849
"""

import datasets
import numpy as np


class MultiDataset:
    def __init__(self, tasks):
        self.tasks = tasks

        # Create random order of tasks
        # Using size-proportional sampling
        task_choice_list = []
        for i, task in enumerate(self.tasks):
            task_choice_list += [i] * len(task)
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        # Add index into each dataset
        # - We don't want to shuffle within each task
        counters = {}
        self.task_choice_list = []
        for i in range(len(task_choice_list)):
            idx = counters.get(task_choice_list[i], 0)
            self.task_choice_list.append((task_choice_list[i], idx))
            counters[task_choice_list[i]] = idx + 1

    def __len__(self):
        return np.sum([len(t) for t in self.tasks])

    def __repr__(self):
        task_str = ", ".join([str(t) for t in self.tasks])
        return f"MultiDataset(tasks: {task_str})"

    def __getitem__(self, key):
        if isinstance(key, int):
            task_idx, example_idx = self.task_choice_list[key]
            task = self.tasks[task_idx]
            example = task[example_idx]
            example["task_name"] = task.info.builder_name
            return example
        elif isinstance(key, slice):
            raise NotImplementedError()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_multitask(*datasets):
    """Create multitask datasets per split"""

    def _get_common_splits(datasets):
        """Finds the common splits present in all self.datasets"""
        min_set = None
        for dataset in datasets:
            if min_set is not None:
                min_set.intersection(set(dataset.keys()))
            else:
                min_set = set(dataset.keys())
        return min_set

    common_splits = _get_common_splits(datasets)
    out = {}
    for split in common_splits:
        out[split] = MultiDataset([d[split] for d in datasets])
    return out


##########################################
# Dataset Flattening

def flatten(dataset, flatten_fn):
    for k in dataset.keys():
        if isinstance(dataset[k], datasets.Dataset):
            dataset[k] = dataset[k].map(flatten_fn, remove_columns=dataset[k].column_names)


# Squad
def flatten_squad(example):
    return {"source": "squad context: " + example['context'] + " question: " + example['question'],
            "target": example["answers"]["text"]}


squad = datasets.load_dataset("squad")
flatten(squad, flatten_squad)


# CNN_DM
def flatten_cnn_dm(example):
    return {"source": "cnn_dm: " + example['article'], "target": [example["highlights"]]}


cnn_dm = datasets.load_dataset("cnn_dailymail", "3.0.0")
flatten(cnn_dm, flatten_cnn_dm)

#############################################


if __name__ == '__main__':
    mtds = load_multitask(squad, cnn_dm)

    for example in mtds["train"]:
        print(example["task_name"], example["target"])
