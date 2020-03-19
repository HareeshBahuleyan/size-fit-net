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
    data_config = load_config_from_json(args.data_config_path)
    model_config = load_config_from_json(args.model_config_path)

    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = ModCloth(data_config, split=split)

    # initialize model
    model = SFNet(model_config["sfnet"])
    model = model.to(device)

    print(model)
    print("Number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    # if args.tensorboard_logging:
    #     writer = SummaryWriter(os.path.join(args.logdir))
    #     writer.add_text("model", str(model))
    #     writer.add_text("args", str(args))

    # save_model_path = os.path.join(args.save_model_path, ts)
    # os.makedirs(save_model_path)

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

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Total Loss"%split.upper(), loss.item(), epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch Stats %04d/%i, Loss=%4.2f, Recon-Loss=%4.2f, KL-Loss=%4.2f, KL-Weight=%4.3f, Weighted-KL=%4.2f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight, weighted_kl_loss.item()/batch_size ))

                if split == 'valid':
                    # tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)
                    tracker['target_sents'] += idx2word(batch['target'].cpu().data.numpy(), i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    preds = torch.topk(logp, k=1, dim=-1)[1] # model predictions on validation set for that batch
                    tracker['pred_sents'] += idx2word(preds.view(batch_size, -1).cpu().data.numpy(), i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)

            # Print a sample of 10 validation set sentences for each epoch
            if split == 'valid':
                for i in range(10):
                    print('A:{}'.format(tracker['target_sents'][i]))
                    print('G:{}\n'.format(tracker['pred_sents'][i]))

                hypotheses_val = []
                references_val = []
                # Compute BLEU scores on the validation set
                for k, (actual, pred) in enumerate(zip(tracker['target_sents'], tracker['pred_sents'])):
                    pred = word_tokenize(pred)
                    actual = word_tokenize(actual)
                    hypotheses_val.append([w for w in pred if w not in special_tokens])
                    references_val.append([[w for w in actual if w not in special_tokens]])

                bleu_scores = list(utils.calculate_bleu_scores(references_val, hypotheses_val))
                print('BLEU 1-4: ', ' | '.join(map(str, bleu_scores)))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/BLEU-1" % split.upper(), bleu_scores[0], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/BLEU-2" % split.upper(), bleu_scores[1], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/BLEU-3" % split.upper(), bleu_scores[2], epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/BLEU-4" % split.upper(), bleu_scores[3], epoch * len(data_loader) + iteration)

            print("%s Epoch %02d/%i, Mean Total Loss %9.4f"%(split.upper(), epoch + 1, args.epochs, torch.mean(tracker['Total Loss'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/Total Loss"%split.upper(), torch.mean(tracker['Total Loss']), epoch)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch + 1))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)

    # Save Model Config File
    with open(os.path.join(args.save_config_path, 'experiment_' + str(len(os.listdir(args.save_config_path))+1)) + '_' + str(ts), 'w') as f:
        f.write(str(args.__dict__))
        # f.write('\n'.join(experiment_name(args, ts).split('_')))

    # Save a dump of of generated and target sentences at the end of training
    dump = ''
    for i in range(len(tracker['target_sents'])):
        dump += 'A:{} \n'.format(tracker['target_sents'][i]) + 'G:{}\n\n'.format(tracker['pred_sents'][i])

    if not os.path.exists(os.path.join('dumps', ts)):
        os.makedirs('dumps/'+ts)
    with open(os.path.join('dumps/'+ts+'/valid_E%i.txt'%(epoch + 1)), 'w') as dump_file:
        dump_file.write(dump)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, default='configs/data.jsonnet')
    parser.add_argument('--model_config_path', type=str, default='configs/model.jsonnet')

    args = parser.parse_args()
    main(args)