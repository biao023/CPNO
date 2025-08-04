import torch.utils.data as Data
import torch.nn.init
import numpy as np
import random
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models_baselines import HyperPINN
import torch.optim as optim
import logging


def define_args():
    parser = argparse.ArgumentParser(description="physics-polyINR")

    # general
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_use', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', default='burgers')
    parser.add_argument('--batch_size_IC', type=int, default=2000, metavar='N')
    parser.add_argument('--batch_size_BC', type=int, default=2000, metavar='N')
    parser.add_argument('--batch_size_PDE', type=int, default=20000, metavar='N')
    parser.add_argument('--device_num', type=int, default=0, metavar='N')

    # model
    parser.add_argument('--inr_dim_in', type=int, default=2)
    parser.add_argument('--inr_dim_out', type=int, default=1)
    parser.add_argument('--inr_dim_hidden', type=int, default=128)
    parser.add_argument('--inr_num_layers', type=int, default=6)
    parser.add_argument('--hyper_dim_in', type=int, default=101)
    parser.add_argument('--hyper_dim_hidden', type=int, default=64)
    parser.add_argument('--hyper_num_layers', type=int, default=5)
    
    #loss
    parser.add_argument('--c_ic', type=int, default=20)
    parser.add_argument('--c_bc', type=int, default=1)
    parser.add_argument('--c_pde', type=int, default=1)
    return parser

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
parser = define_args()
args = parser.parse_args()

def dataset_burgers():
    data = np.load('../datasets/burgers_data.npz', allow_pickle=True)
    tt_params, tt_coords, tt_values = data['tt_params'], data['tt_coords'], data['tt_values']
    dd_params, dd_coords, dd_values = data['dd_params'], data['dd_coords'],  data['dd_values']
    X_IC, U_IC, params_IC = data['X_IC'], data['U_IC'], data['params_IC']
    X_BC1, X_BC2, params_BC1, params_BC2  = data['X_BC1'], data['X_BC2'], data['params_BC1'], data['params_BC2']
    X_res, U_res, params_res  = data['X_res'], data['U_res'], data['params_res']

    X_IC, U_IC, params_IC, X_BC1, params_BC1, \
    X_BC2, params_BC2, X_res, params_res = torch.from_numpy(X_IC).to(torch.float32), \
                                           torch.from_numpy(U_IC).to(torch.float32), \
                                           torch.from_numpy( params_IC).to(torch.float32), \
                                           torch.from_numpy( X_BC1).to(torch.float32), \
                                           torch.from_numpy(params_BC1).to(torch.float32), \
                                           torch.from_numpy(X_BC2).to(torch.float32), \
                                           torch.from_numpy(params_BC2).to(torch.float32), \
                                           torch.from_numpy(X_res).to(torch.float32), \
                                           torch.from_numpy(params_res).to(torch.float32)
    tt_params, tt_coords = torch.from_numpy(tt_params).to(torch.float32),\
                           torch.from_numpy(tt_coords).to(torch.float32)
    dd_params, dd_coords, dd_values = torch.from_numpy(dd_params).to(torch.float32).to(device),\
                                      torch.from_numpy(dd_coords).to(torch.float32).to(device),\
                                      torch.from_numpy(dd_values).to(torch.float32).to(device)

    dataset_IC = Data.TensorDataset(X_IC, U_IC, params_IC)
    dataset_BC = Data.TensorDataset(X_BC1, X_BC2, params_BC1, params_BC2)
    dataset_Res = Data.TensorDataset(X_res, params_res)
    test = [tt_params, tt_coords, tt_values]
    data = [dd_params, dd_coords, dd_values]

    return dataset_IC, dataset_BC, dataset_Res, data, test

def physics_loss(params_IC, X_IC, U_IC, params_BC1, params_BC2, X_BC1, X_BC2,
                                params_Res, X_Res, model):

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    IC_pre = model(params_IC, X_IC)
    loss_IC = criterion(IC_pre, U_IC)

    X_BC1.requires_grad = True
    X_BC2.requires_grad = True
    yb1 = model(params_BC1, X_BC1)
    yb2 = model(params_BC2, X_BC2)
    dy1_dx = torch.autograd.grad(inputs=X_BC1, outputs=yb1, grad_outputs=torch.ones_like(yb1),
                                 retain_graph=True, create_graph=True)[0][:, 0]
    dy2_dx = torch.autograd.grad(inputs=X_BC2, outputs=yb2, grad_outputs=torch.ones_like(yb2),
                                 retain_graph=True, create_graph=True)[0][:, 0]
    loss_BC = criterion(yb1, yb2) + criterion(dy1_dx, dy2_dx)

    x = X_Res.clone()
    x.requires_grad = True
    u = model(params_Res, x)
    du_dX = torch.autograd.grad(
        inputs=x,
        outputs=u,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]  #
    du_dt = du_dX[:, 1]
    du_dx = du_dX[:, 0]
    du_dxx = torch.autograd.grad(
        inputs=x,
        outputs=du_dX,
        grad_outputs=torch.ones_like(du_dX),
        retain_graph=True,
        create_graph=True
    )[0][:, 0]
    loss_PDE = criterion(du_dt + u.squeeze() * du_dx, 0.01 * du_dxx)

    loss =  args.c_ic*loss_IC + args.c_bc*loss_BC +  args.c_pde*loss_PDE
    
    return loss_IC, loss_BC, loss_PDE, loss


def main(args):
    args.log_dir = './logs_burgers'
    if args.data_use:
        name = 'hyperpinn-physics'
    else:
        name = 'hyperpinn-hybrid'

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

    dataset_IC, dataset_BC, dataset_Res, data, test = dataset_burgers()
    loader_IC = Data.DataLoader(dataset=dataset_IC, batch_size=args.batch_size_IC, shuffle=True)
    loader_BC = Data.DataLoader(dataset=dataset_BC, batch_size=args.batch_size_BC, shuffle=True)
    loader_Res = Data.DataLoader(dataset=dataset_Res, batch_size=args.batch_size_PDE, shuffle=True)

    dd_params, dd_coords, dd_values = data[0], data[1], data[2]

    # test data creating
    params_test, X_test, U_test = test[0], test[1], test[2]
    params_test, X_test = params_test.to(torch.float32).to(device), X_test.to(torch.float32).to(device)
    logging.info("batch_size_IC: {}, batch_size_BC: {}, batch_size_PDE: {}"\
                             .format(args.batch_size_IC, args.batch_size_BC, args.batch_size_PDE))
    logging.info(args.dataset + ' dataset is ready.')

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    # create model
    Ns = 64
    latent_dim = Ns * 8 + Ns * Ns * 4
    para_layer =  [101, 64, 64, 64, 64, latent_dim]
    model = HyperPINN(para_layer, Ns)
    #model = PolyINRWithHypernet(args.inr_dim_in, args.inr_dim_out, args.inr_dim_hidden, args.inr_num_layers,\
    #             args.hyper_dim_in, args.hyper_dim_hidden, args.hyper_num_layers)

    model=model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)

    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(args.log_dir, name + '.log'))

    params_num = sum(param.numel() for param in model.parameters())
    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    logging.info("Starting training loop at step %d." % (0,))

    for epoch in range(args.epochs):
        loss_ic_list, loss_bc_list, loss_pde_list, loss_data_list, loss_list = [], [], [], [], []
        for step, (IC_batch, BC_batch, Res_batch) in enumerate(zip(loader_IC, loader_BC, loader_Res)):
            optimizer.zero_grad()
            X_IC_batch, U_IC_batch, params_IC_batch = IC_batch[0].to(device),\
                                                      IC_batch[1].to(device),\
                                                      IC_batch[2].to(device)
            X_BC1_batch, X_BC2_batch, params_BC1_batch, params_BC2_batch = BC_batch[0].to(device),\
                                                                           BC_batch[1].to(device),\
                                                                           BC_batch[2].to(device),\
                                                                           BC_batch[3].to(device)
            X_Res_batch, params_Res_batch = Res_batch[0].to(device), Res_batch[1].to(device)


            loss_IC, loss_BC, loss_PDE, loss = physics_loss(params_IC_batch, X_IC_batch, U_IC_batch,
                                params_BC1_batch, params_BC2_batch, X_BC1_batch, X_BC2_batch,
                                params_Res_batch, X_Res_batch, model)
            if args.data_use:
                dd_pre = model(dd_params, dd_coords)
                loss_data = criterion(dd_pre, dd_values)
                loss_data_list.append(loss_data.item())
                loss = loss + loss_data
            loss.backward()
            optimizer.step()
            loss_ic_list.append(loss_IC.item())
            loss_bc_list.append(loss_BC.item())
            loss_pde_list.append(loss_PDE.item())
            loss_list.append(loss.item())

        scheduler.step()
        loss_ic_avg = np.mean(np.array(loss_ic_list))
        loss_bc_avg = np.mean(np.array(loss_bc_list))
        loss_pde_avg = np.mean(np.array(loss_pde_list))
        loss_avg = np.mean(np.array(loss_list))
        if args.data_use:
            loss_data_avg = np.mean(np.array(loss_data_list))

        if epoch > 0:
            with torch.no_grad():
                test_pre = model(params_test, X_test).cpu().numpy()
                L2_Re = np.linalg.norm((test_pre - U_test), 2) / np.linalg.norm(U_test, 2)

                if args.data_use:
                    logging.info(
                        "Running physics driven, Epoch: {}, IC loss: {}, BC loss: {}, PDE loss: {}, data loss: {}, loss: {}, L2 error:{} " \
                        .format(epoch, loss_ic_avg, loss_bc_avg, loss_pde_avg, loss_data_avg, loss_avg, L2_Re.item()))
                else:
                    logging.info("Running physics driven, Epoch: {}, IC loss: {}, BC loss: {}, PDE loss: {},  loss: {}, L2 error:{} "\
                             .format(epoch, loss_ic_avg, loss_bc_avg, loss_pde_avg, loss_avg, L2_Re.item()))
        if epoch% 10==0 and epoch >0:
            if args.data_use:
                torch.save(model, rf'{args.log_dir}/hyperpinn_burgers_hybrid.pkl')
            else:
                torch.save(model, rf'{args.log_dir}/hyperpinn_burgers_physics.pkl')

    logging.info("Data generation and training is completed")


if __name__ == '__main__':
    main(args)
