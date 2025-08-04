import torch.utils.data as Data
import torch.nn.init
import numpy as np
import random
import argparse
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models_poly_chebcode import PolyINRWithHypernet
import torch.optim as optim
import logging
from datasets.data_util import *


def define_args():
    parser = argparse.ArgumentParser(description="physics-polyINR-allen-cahn")

    # general
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_use', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', default='allen-cahn')
    parser.add_argument('--batch_size_IC', type=int, default=1000, metavar='N')
    parser.add_argument('--batch_size_PDE', type=int, default=10000, metavar='N')
    parser.add_argument('--device_num', type=int, default=0, metavar='N')

    # model
    parser.add_argument('--inr_dim_in', type=int, default=2)
    parser.add_argument('--inr_dim_out', type=int, default=1)
    parser.add_argument('--inr_dim_hidden', type=int, default=64)
    parser.add_argument('--inr_num_layers', type=int, default=8)
    parser.add_argument('--cheby_orders', type=int, default=5)

    parser.add_argument('--hyper_dim_in', type=int, default=100)
    parser.add_argument('--hyper_dim_hidden', type=int, default=64)
    parser.add_argument('--hyper_num_layers', type=int, default=6)


    # loss
    parser.add_argument('--c_ic', type=int, default=5)
    parser.add_argument('--c_pde', type=int, default=1)

    return parser


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
parser = define_args()
args = parser.parse_args()


def dataset_ac():
    data_train = np.load('../datasets/ac_train.npz', allow_pickle=True)
    u_bcs_train, y_bcs_train, s_bcs_train = \
        data_train['u_bcs_train'], data_train['y_bcs_trainb'], data_train['s_bcs_train']
    u_res_train, y_res_train, s_res_train = data_train['u_res_train'], data_train['y_res_train'], data_train[
        's_res_train']

    data_test = np.load('../datasets/ac_test.npz', allow_pickle=True)
    u_test, y_test, s_test = data_test['u_test'], data_test['y_test'], data_test['s_test']

    u_bcs_train, y_bcs_train, s_bcs_train = torch.from_numpy(u_bcs_train).to(torch.float32), \
                                            torch.from_numpy(y_bcs_train).to(torch.float32), \
                                            torch.from_numpy(s_bcs_train).to(torch.float32),
    u_res_train, y_res_train, s_res_train = torch.from_numpy(u_res_train).to(torch.float32), \
                                            torch.from_numpy(y_res_train).to(torch.float32), \
                                            torch.from_numpy(s_res_train).to(torch.float32)

    u_test, y_test = torch.from_numpy(u_test).to(torch.float32).to(device), \
                     torch.from_numpy(y_test).to(torch.float32).to(device)
    dataset_BC = Data.TensorDataset(u_bcs_train, y_bcs_train, s_bcs_train)
    dataset_Res = Data.TensorDataset(u_res_train, y_res_train, s_res_train)

    test = [u_test, y_test, s_test]

    return dataset_BC, dataset_Res, test


def physics_loss(params_BC, y_BC, s_BC, params_res, y_res, model):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    s_BC_pre = model(params_BC, y_BC)
    loss_BC = criterion(s_BC_pre, s_BC)

    y_res.requires_grad = True
    s = model(params_res, y_res)
    s = s.reshape(s.shape[0])

    ds_dX = torch.autograd.grad(
        inputs=y_res,
        outputs=s,
        grad_outputs=torch.ones_like(s),
        retain_graph=True,
        create_graph=True
    )[0]  #
    ds_dt = ds_dX[:, 1]
    ds_dx = ds_dX[:, 0]
    ds_dxx = torch.autograd.grad(
        inputs=y_res,
        outputs=ds_dX,
        grad_outputs=torch.ones_like(ds_dX),
        retain_graph=True,
        create_graph=True
    )[0][:, 0]

    res = ds_dt - 0.0005 * ds_dxx - 0.1*s*(1-s*s)
    res_zero = torch.zeros_like(res)
    loss_PDE = criterion(res, res_zero)
    loss = loss_BC + loss_PDE

    return loss_BC, loss_PDE, loss

def main(args):
    args.log_dir = './logs_ac_cheborder'

    if args.data_use:
        name = 'PolyINR-cheb-hybrid={}-epoch={}-lr={}-inr_dim_hidden={}-' \
           'inr_layers={}-hyper_dim_hidden={}-hyper_num_layers={}-c_IC={}-c_PDE={}_cheborder={}'.format(
        args.dataset, args.epochs, args.lr, args.inr_dim_hidden,
        args.inr_num_layers, args.hyper_dim_hidden, args.hyper_num_layers, args.c_ic, args.c_pde, args.cheby_orders)
    else:
        name = 'PolyINR-cheb-physics={}-epoch={}-lr={}-inr_dim_hidden={}-' \
                   'inr_layers={}-hyper_dim_hidden={}-hyper_num_layers={}-c_IC={}-c_PDE={}_cheborders={}'.format(
                args.dataset, args.epochs, args.lr, args.inr_dim_hidden,
                args.inr_num_layers, args.hyper_dim_hidden, args.hyper_num_layers, args.c_ic, args.c_pde, args.cheby_orders)

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

    dataset_IC, dataset_Res, test = dataset_ac()
    loader_IC = Data.DataLoader(dataset=dataset_IC, batch_size=args.batch_size_IC, shuffle=True)
    loader_Res = Data.DataLoader(dataset=dataset_Res, batch_size=args.batch_size_PDE, shuffle=True)

    # test data creating
    params_test, y_test, s_test = test[0], test[1], test[2]
    # params_test, X_test = params_test.to(torch.float32).to(device), X_test.to(torch.float32).to(device)
    logging.info(args.dataset + ' dataset is ready.')

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    # create model
    model = PolyINRWithHypernet(args.inr_dim_in, args.inr_dim_out, args.inr_dim_hidden, args.inr_num_layers, \
                                args.hyper_dim_in, args.hyper_dim_hidden, args.hyper_num_layers, args.cheby_orders)

    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(args.log_dir, name + '.log'))

    params_num = sum(param.numel() for param in model.parameters())
    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    logging.info("Starting training loop at step %d." % (0,))

    for epoch in range(args.epochs):
        loss_bc_list, loss_pde_list, loss_data_list, loss_list = [], [], [], []
        for step, (BC_batch, Res_batch) in enumerate(zip(loader_IC, loader_Res)):
            optimizer.zero_grad()
            params_BC_batch, y_BC_batch, s_BC_batch, = BC_batch[0].to(device), \
                                                       BC_batch[1].to(device), \
                                                       BC_batch[2].to(device)
            params_Res_batch, y_Res_batch, s_Res_batch = Res_batch[0].to(device), \
                                                         Res_batch[1].to(device), \
                                                         Res_batch[2].to(device)

            loss_BC, loss_PDE, loss = physics_loss(params_BC_batch, y_BC_batch, s_BC_batch, \
                                                   params_Res_batch, y_Res_batch, model)

            loss.backward()
            optimizer.step()
            loss_bc_list.append(loss_BC.item())
            loss_pde_list.append(loss_PDE.item())
            loss_list.append(loss.item())

        scheduler.step()
        loss_bc_avg = np.mean(np.array(loss_bc_list))
        loss_pde_avg = np.mean(np.array(loss_pde_list))
        loss_avg = np.mean(np.array(loss_list))

        if epoch > 0:
            with torch.no_grad():
                test_pre = model(params_test, y_test)
                test_pre = test_pre.reshape(100 * 100 * 10, 1).cpu().numpy()
                L2_Re = np.linalg.norm((test_pre - s_test), 2) / np.linalg.norm(s_test, 2)

                logging.info("Running physics driven, Epoch: {}, BC loss: {}, PDE loss: {}, loss: {}, L2 error: {} " \
                        .format(epoch, loss_bc_avg, loss_pde_avg, loss_avg, L2_Re.item()))

    logging.info("Data generation and training is completed")

    if args.data_use:
        torch.save(model, rf'{args.log_dir}/polyINR-cheb_ac_hybrid.pkl')
    else:
        torch.save(model, rf'{args.log_dir}/polyINR-cheb_ac_physics.pkl')


if __name__ == '__main__':
    main(args)
