from torch.utils.data import Dataset, DataLoader
import torch


class MoETraceDataset(Dataset):
    def __init__(self, traces_dict, labels):
        self.prompt_ids = sorted(traces_dict.keys())
        self.traces = traces_dict
        self.labels = labels

        # --- AUTO-DETECT CONFIGURATION ---
        # Grab the first prompt's data to measure dimensions
        first_pid = self.prompt_ids[0]
        first_token_path = self.traces[first_pid][0]  # List of lists: [[e1, e2], [e3, e4]...]

        # 1. Count how many lists are inside (Number of Layers)
        self.detected_num_layers = len(first_token_path)

        # 2. Count how many items in the first list (Top K)
        self.detected_top_k = len(first_token_path[0])

    def __len__(self):
        return len(self.prompt_ids)

    def __getitem__(self, idx):
        pid = self.prompt_ids[idx]
        token_indices = sorted(self.traces[pid].keys())

        # Convert to Tensor: (Num_Tokens, Num_Layers, Top_K)
        prompt_data = [self.traces[pid][t] for t in token_indices]
        x_tensor = torch.tensor(prompt_data, dtype=torch.long)
        y_label = torch.tensor(self.labels[pid], dtype=torch.float32)

        return x_tensor, y_label


def pad_collate(batch):
    """
    Pads variable-length sequences in a batch to the max length.
    """
    # batch is a list of tuples (x_tensor, y_label)
    (xx, yy) = zip(*batch)

    # Get the length of each prompt in the batch
    x_lens = [len(x) for x in xx]

    # Pad the prompts with 0s so they are all the same length as the longest one
    # batch_first=True makes output (Batch, Max_Len, 32, 2)
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)

    yy = torch.stack(yy)

    return xx_pad, yy, x_lens


def get_dataLoader(dataset, batch_size, shuffle, collate_fn=pad_collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
