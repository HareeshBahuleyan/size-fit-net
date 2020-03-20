import os
import argparse
import torch
import numpy as np

from utils import compute_metrics
from utils import to_var, load_config_from_json

from torch.utils.data import DataLoader
from modcloth import ModCloth
from model import SFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(
        os.path.join(args.saved_model_path, "config.jsonl")
    )

    # initialize model
    model = SFNet(model_config["sfnet"])
    model = model.to(device)

    if not os.path.exists(args.saved_model_path):
        raise FileNotFoundError(args.saved_model_path)

    checkpoint = os.path.join(args.saved_model_path, args.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print("Model loaded from %s" % (args.saved_model_path))

    # tracker to keep true labels and predicted probabilitites
    target_tracker = []
    pred_tracker = []

    print("Preparing test data ...")
    dataset = ModCloth(data_config, split="test")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=model_config["trainer"]["batch_size"],
        shuffle=False,
    )

    print("Evaluating model on test data ...")
    model.eval()
    with torch.no_grad():

        for iteration, batch in enumerate(data_loader):

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            _, pred_probs = model(batch)

            target_tracker.append(batch["fit"].cpu().numpy())
            pred_tracker.append(pred_probs.cpu().data.numpy())

    target_tracker = np.stack(target_tracker[:-1]).reshape(-1)
    pred_tracker = np.stack(pred_tracker[:-1], axis=0).reshape(
        -1, model_config["sfnet"]["num_targets"]
    )
    precision, recall, f1_score, accuracy, auc = compute_metrics(
        target_tracker, pred_tracker
    )

    print("-" * 50)
    print(
        "Metrics:\n Precision = {:.3f}\n Recall = {:.3f}\n F1-score = {:.3f}\n Accuracy = {:.3f}\n AUC = {:.3f}\n ".format(
            precision, recall, f1_score, accuracy, auc
        )
    )
    print("-" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_path", type=str, default="configs/data.jsonnet")
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="E20.pytorch")

    args = parser.parse_args()
    main(args)
