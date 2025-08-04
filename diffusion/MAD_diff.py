import torch.utils.data as Data
import torch.nn.init
import numpy as np
import random
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models_baselines import MAD
import torch.optim as optim
import logging
from datasets.data_util import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os


def define_args():
    parser = argparse.ArgumentParser(description="physics-polyINR-diffusion")

    # general
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_use', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', default='diffusion')
    parser.add_argument('--batch_size_BC', type=int, default=2000, metavar='N')
    parser.add_argument('--batch_size_PDE', type=int, default=20000, metavar='N')
    parser.add_argument('--device_num', type=int, default=0, metavar='N')


    # model
    parser.add_argument('--inr_dim_in', type=int, default=2)
    parser.add_argument('--inr_dim_out', type=int, default=1)
    parser.add_argument('--inr_dim_hidden', type=int, default=128)
    parser.add_argument('--inr_num_layers', type=int, default=6)
    parser.add_argument('--hyper_dim_in', type=int, default=100)
    parser.add_argument('--hyper_dim_hidden', type=int, default=64)
    parser.add_argument('--hyper_num_layers', type=int, default=5)
    
    #loss
    parser.add_argument('--c_bc', type=int, default=1)
    parser.add_argument('--c_pde', type=int, default=1)

    return parser

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
parser = define_args()
args = parser.parse_args()

def dataset_advection():
    data_train = np.load('../datasets/diff_train.npz', allow_pickle=True)
    u_bcs_train, y_bcs_train, s_bcs_train =  \
        data_train['u_bcs_train'], data_train['y_bcs_trainb'], data_train['s_bcs_train']
    u_res_train, y_res_train, s_res_train = data_train['u_res_train'], data_train['y_res_train'], data_train[
        's_res_train']

    data_test = np.load('../datasets/diff_test.npz', allow_pickle=True)
    u_test, y_test, s_test = data_test['u_test'], data_test['y_test'], data_test['s_test']

    data_data = np.load('../datasets/diff_data.npz', allow_pickle=True)
    u_data, y_data, s_data = data_data['u_data'], data_data['y_data'], data_data['s_data']

    u_bcs_train, y_bcs_train, s_bcs_train = torch.from_numpy(u_bcs_train).to(torch.float32), \
                                            torch.from_numpy(y_bcs_train).to(torch.float32), \
                                            torch.from_numpy(s_bcs_train).to(torch.float32),
    u_res_train, y_res_train, s_res_train = torch.from_numpy(u_res_train).to(torch.float32), \
                                            torch.from_numpy(y_res_train).to(torch.float32), \
                                            torch.from_numpy(s_res_train).to(torch.float32)
    u_data, y_data, s_data =  torch.from_numpy(u_data).to(torch.float32).to(device),\
                              torch.from_numpy(y_data).to(torch.float32).to(device),\
                              torch.from_numpy(s_data).to(torch.float32).to(device)
    idx = torch.randperm(u_data.size(0))
    u_data, y_data, s_data = u_data[idx], y_data[idx], s_data[idx]

    u_test, y_test = torch.from_numpy(u_test).to(torch.float32).to(device),\
                     torch.from_numpy(y_test).to(torch.float32).to(device)
    dataset_BC = Data.TensorDataset(u_bcs_train, y_bcs_train, s_bcs_train)
    dataset_Res = Data.TensorDataset(u_res_train, y_res_train, s_res_train)

    s_test = s_test.reshape(100*100*10)
    data = [u_data, y_data, s_data]
    test = [u_test, y_test, s_test]

    return  dataset_BC, dataset_Res, data, test

def physics_loss(params_bcs, y_bcs, params_res, y_res, s_res, model):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)

    sb_pre = model(params_bcs, y_bcs)
    sb = torch.zeros_like(sb_pre)
    loss_BC = criterion(sb_pre, sb)

    y = y_res.clone()
    y.requires_grad = True
    s = model(params_res, y)
    s = s.reshape(s.shape[0])
    ds_dX = torch.autograd.grad(
        inputs=y,
        outputs=s,
        grad_outputs=torch.ones_like(s),
        retain_graph=True,
        create_graph=True
    )[0]  #
    ds_dt = ds_dX[:, 1]
    ds_dx = ds_dX[:, 0]
    ds_dxx = torch.autograd.grad(
        inputs=y,
        outputs=ds_dX,
        grad_outputs=torch.ones_like(ds_dX),
        retain_graph=True,
        create_graph=True
    )[0][:, 0]
    res = ds_dt - 0.01 * ds_dxx - 0.01 * s.mul(s)
    res = res.reshape((len(res), 1))
    loss_PDE = criterion(res, s_res)
    loss = 5.0*loss_BC + 2.0*loss_PDE

    return loss_BC, loss_PDE, loss


def main():
    args.log_dir = './logs_diff'

    if args.data_use:
        name = 'MAD-hybrid-diff'
    else:
        name = 'MAD-physics-diff'

    os.makedirs(args.log_dir, exist_ok=True)
    name = os.path.join(args.log_dir, name)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=name + '.log',
                        filemode='a')

    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    dataset_BC, dataset_Res, data, test = dataset_advection()
    loader_BC = Data.DataLoader(dataset=dataset_BC, batch_size=args.batch_size_BC, shuffle=True)
    loader_Res = Data.DataLoader(dataset=dataset_Res, batch_size=args.batch_size_PDE, shuffle=True)

    # test data creating
    params_data, y_data, s_data = data[0], data[1], data[2]
    params_test, y_test, s_test = test[0], test[1], test[2]
    logging.info(args.dataset + ' dataset is ready.')

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    # create model
    para_layer = [100, 64, 64, 64, 64, 64]
    truck_layer = [2 + 64, 64, 64, 64, 64, 1]
    model = MAD(para_layer, truck_layer)

    #model = PolyINRWithHypernet(args.inr_dim_in, args.inr_dim_out, args.inr_dim_hidden, args.inr_num_layers,\
    #             args.hyper_dim_in, args.hyper_dim_hidden, args.hyper_num_layers, 5)
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(args.log_dir, name + '.log'))

    params_num = sum(param.numel() for param in model.parameters())
    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    logging.info("Starting training loop at step %d." % (0,))

    for epoch in range(args.epochs):
        loss_bc_list, loss_pde_list, loss_data_list, loss_list = [], [], [], []
        for step, (BC_batch, Res_batch) in enumerate(zip(loader_BC, loader_Res)):
            optimizer.zero_grad()
            params_BC_batch, y_BC_batch = BC_batch[0].to(device), \
                                          BC_batch[1].to(device)

            params_Res_batch, y_Res_batch, s_Res_batch = Res_batch[0].to(device),\
                                                         Res_batch[1].to(device),\
                                                         Res_batch[2].to(device)

            loss_BC, loss_PDE, loss = physics_loss(params_BC_batch, y_BC_batch,
                                                   params_Res_batch, y_Res_batch, s_Res_batch, model)

            if args.data_use:
                s_pre = model(params_data, y_data)
                loss_data = criterion(s_pre, s_data )
                loss_data_list.append(loss_data.item())
                loss = loss + loss_data

            loss.backward()
            optimizer.step()
            loss_bc_list.append(loss_BC.item())
            loss_pde_list.append(loss_PDE.item())
            loss_list.append(loss.item())

        scheduler.step()
        loss_bc_avg = np.mean(np.array(loss_bc_list))
        loss_pde_avg = np.mean(np.array(loss_pde_list))
        loss_avg = np.mean(np.array(loss_list))
        if args.data_use:
            loss_data_avg = np.mean(np.array(loss_data_list))

        if epoch > 0:
            with torch.no_grad():
                test_pre = model(params_test, y_test)
                test_pre = test_pre.reshape(100 * 100 * 10).cpu().numpy()
                L2_Re = np.linalg.norm((test_pre - s_test), 2) / np.linalg.norm(s_test, 2)
                if args.data_use:
                    logging.info(
                    "Running physics driven, Epoch: {}, BC loss: {}, PDE loss: {}, data loss:{}, loss: {}, L2 error: {} " \
                        .format(epoch, loss_bc_avg, loss_pde_avg, loss_data_avg, loss_avg, L2_Re.item()))
                else:
                    logging.info(
                        "Running physics driven, Epoch: {}, BC loss: {}, PDE loss: {}, loss: {}, L2 error: {} " \
                        .format(epoch,  loss_bc_avg, loss_pde_avg, loss_avg, L2_Re.item()))

    logging.info("Data generation and training is completed")
    if args.data_use:
        torch.save(model, rf'{args.log_dir}/MAD_diff_hybrid.pkl')
    else:
        torch.save(model, rf'{args.log_dir}/MAD_diff_physics.pkl')


if __name__ == '__main__':
    main()
