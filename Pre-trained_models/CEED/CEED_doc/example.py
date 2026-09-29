# %%
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

# %%
ceed = load_dataset(
    "./ceed.py",
    name="station_test",
    # name="event_test",
    split="test",
    download_mode="force_redownload",
    trust_remote_code=True,
)

# print the first sample of the iterable dataset
for example in ceed:
    print("\nIterable test\n")
    print(example.keys())
    for key in example.keys():
        if key == "data":
            print(key, np.array(example[key]).shape)
        else:
            print(key, example[key])
    break

# %%
ceed = ceed.with_format("torch")
dataloader = DataLoader(ceed, batch_size=8, num_workers=0, collate_fn=lambda x: x)

for batch in dataloader:
    print("\nDataloader test\n")
    print(f"Batch size: {len(batch)}")
    print(batch[0].keys())
    for key in batch[0].keys():
        if key == "data":
            print(key, np.array(batch[0][key]).shape)
        else:
            print(key, batch[0][key])
    break

# %%
