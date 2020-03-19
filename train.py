import os
import time
import jsonlines
import argparse
import torch

from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter
from utils import to_var, load_config_from_json

from torch.utils.data import DataLoader
from modcloth import ModCloth
from model import SFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(args.model_config_path)

    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = ModCloth(data_config, split=split)

    # initialize model
    model = SFNet(model_config["sfnet"])
    model = model.to(device)

    print("-"*50)
    print(model)
    print("-"*50)
    print("Number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    print("-"*50)

    save_model_path = os.path.join(model_config["logging"]["save_model_path"], model_config["logging"]["run_name"] + ts)
    os.makedirs(save_model_path)

    if model_config["logging"]["tensorboard"]:
        writer = SummaryWriter(os.path.join(save_model_path, 'logs'))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))


    loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["trainer"]["optimizer"]["lr"])

    step = 0
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    
    for epoch in range(model_config["trainer"]["num_epochs"]):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=model_config["trainer"]["batch_size"],
                shuffle=split=='train',
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval() # what about with torch.nograd() for validation set?

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['item_id'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logits = model(batch)

                # loss calculation
                loss = loss_criterion(logits, batch['fit'])

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['Total Loss'] = torch.cat((tracker['Total Loss'], loss.view(1)))

                if model_config["logging"]["tensorboard"]:
                    writer.add_scalar("%s/Total Loss"%split.upper(), loss.item(), epoch*len(data_loader) + iteration)

                if iteration % model_config["logging"]["print_every"] == 0 or iteration+1 == len(data_loader):
                    print("{} Batch Stats {}/{}, Loss={:.2f}".format(split.upper(), iteration, len(data_loader)-1, loss.item()))

                if split == 'valid':
                    # print validation loss + metrics
                    pass

            #     if args.tensorboard_logging:
            #         writer.add_scalar("%s/BLEU-1" % split.upper(), bleu_scores[0], epoch * len(data_loader) + iteration)
            #         writer.add_scalar("%s/BLEU-2" % split.upper(), bleu_scores[1], epoch * len(data_loader) + iteration)
            #         writer.add_scalar("%s/BLEU-3" % split.upper(), bleu_scores[2], epoch * len(data_loader) + iteration)
            #         writer.add_scalar("%s/BLEU-4" % split.upper(), bleu_scores[3], epoch * len(data_loader) + iteration)

            print("%s Epoch %02d/%i, Mean Total Loss %9.4f"%(split.upper(), epoch + 1, model_config["trainer"]["num_epochs"], torch.mean(tracker['Total Loss'])))

            if model_config["logging"]["tensorboard"]:
                writer.add_scalar("%s-Epoch/Total Loss"%split.upper(), torch.mean(tracker['Total Loss']), epoch)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch + 1))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)

    # Save Model Config File
    with jsonlines.open(os.path.join(save_model_path, 'config.jsonl'), 'w') as fout:
        fout.write(model_config)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, default='configs/data.jsonnet')
    parser.add_argument('--model_config_path', type=str, default='configs/model.jsonnet')

    args = parser.parse_args()
    main(args)