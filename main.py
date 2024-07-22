import argparse
import logging
import os
import random
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from dataset import GATDataset
from utils import train_process
from model import PKGN


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_gat_options():
    parser = argparse.ArgumentParser(description='psycho-PKGN')
    parser.add_argument('--main_name', type=str, default=__name__, help='the logger name')
    parser.add_argument('--load_saved_data', type=bool, default=True, help='Load reprocessed data choice')
    parser.add_argument('--saved_data', type=str, default="./neighbor_graph/processed_dataset.pkl", help='the dumped reprocessed data from dataset')
    parser.add_argument('--data_path', type=str, default=None, help='the data path')
    parser.add_argument('--output_dir', type=str, default='result', help='the output directory')
    parser.add_argument('--save_path', type=str, default='res.pt', help='parameter output file')
    parser.add_argument('--cate_length', type=int, default=18, help='LIWC categories count')
    parser.add_argument('--max_node', type=int, default=220, help='psycho graph node count')
    parser.add_argument('--max_length', type=int, default=50, help='post tokens max length')
    parser.add_argument('--epoch', type=int, default=50, help='the training epoch count')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='the number of accumulation steps of gradient')
    parser.add_argument('--iteration', type=int, default=1, help='the number of layer of GAT')
    parser.add_argument('--bert_path', type=str, default='./bert-base-uncased', help='pretrained embed model and tokenizer config')
    parser.add_argument('--max_grad_norm', type=int, default=1, help='maxinum of gradient')
    parser.add_argument('--alpha', type=float, default=0.02, help='negative_slope for leakeyReLU')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--hidden_size', type=int, default=768, help='embedding hidden size')
    parser.add_argument('--head_num', type=int, default=4, help='attention head count')
    parser.add_argument('--label_num', type=int, default=4, help='the personality dimensions')
    parser.add_argument('--main_cuda', type=str, default='cuda:0', help='main cuda device')
    parser.add_argument('--bert_lr', type=float, default=1e-5, help='bert model learning rate')
    parser.add_argument('--gat_lr', type=float, default=1e-3, help='GAT model learning rate')
    parser.add_argument('--output_log', type=str, default="./result/result.log", help='log output file path')
    parser.add_argument('--cate_ablation', type=int, default=-1, help='the category ablation study index, -1 is do not')
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    args = parse_gat_options()
    set_seed(2022)
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    formator = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler = logging.FileHandler(args.output_log, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formator)
    logger.addHandler(file_handler)
    torch.autograd.set_detect_anomaly(True)
    logger.info('hyperparameter = {}'.format(args))
    logger.info('this-process-pid = {}'.format(os.getpid()))

    device = torch.device(args.main_cuda if torch.cuda.is_available() else 'cpu')
    dataset = GATDataset(args)
    dataset.dump_datas("./processed_dataset.pkl")
    if args.cate_ablation != -1:
        dataset.cate_ablation_study(args.cate_ablation)
    model = PKGN(args, dataset.tokenizer).to(device)
    ptm_id = list(map(id, model.emb_encoder.parameters()))
    other_params = filter(lambda p: id(p) not in ptm_id, model.parameters())
    optimizer_grouped_parameters = [
        {'params': other_params, 'lr': args.gat_lr},
        {'params': model.emb_encoder.parameters(), 'lr': args.bert_lr}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    # running model
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    test_dataset, train_dataset = random_split(dataset, [test_size, train_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    logger.info('data_size: train = {}, test = {}'.format(len(train_dataset), len(test_dataset)))
    train_process(
        args=args,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device,
    )