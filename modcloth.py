import json
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class ModCloth(Dataset):
    def __init__(self, config, split):
        self.data = self._read_data(config["data_path"], split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {
            # user features and embedding
            "user_id": np.asarray(self.data[idx]["user_id"], dtype=np.int64),
            "cup_size": np.asarray(self.data[idx]["cup_size"], dtype=np.int64),
            "user_numeric": np.asarray(
                self.data[idx]["user_numeric"], dtype=np.float32
            ),
            # item features and embedding
            "item_id": np.asarray(self.data[idx]["item_id"], dtype=np.int64),
            "category": np.asarray(self.data[idx]["category"], dtype=np.int64),
            "item_numeric": np.asarray(
                self.data[idx]["item_numeric"], dtype=np.float32
            ),
            # target variable
            "fit": np.asarray(self.data[idx]["fit"], dtype=np.int64),
        }

    def _read_data(self, data_path, split):
        data = []
        data_path = data_path.split(".")[0] + "_" + split + ".jsonl"
        with open(data_path, "r") as fin:
            for line in tqdm(fin):
                record = json.loads(line)
                feature_record = {
                    "user_id": record["user_id"],
                    "cup_size": record["cup_size"],
                    "user_numeric": [
                        record["waist"],
                        record["hips"],
                        record["bra_size"],
                        record["height"],
                        record["shoe_size"],
                    ],
                    "item_id": record["item_id"],
                    "category": record["category"],
                    "item_numeric": [record["size"], record["quality"]],
                    "fit": record["fit"],
                }
                data.append(feature_record)

        return data
