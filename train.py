from model import lipmlp
import jaxgptoolbox as jgp

import jax.numpy as np
from jax import jit, value_and_grad
from jax.experimental import optimizers

import numpy as onp
import numpy.random as random
import matplotlib.pyplot as plt
import tqdm


def train(hyper_params):
    # initialize a mlp
    model = lipmlp(hyper_params)
    params = model.initialize_weights()

    # get training sample generator
    get_training_samples = get_training_samples_generator(hyper_params)

    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
    opt_state = opt_init(params)

    # define loss function and update function
    def loss(params_, alpha, x_, y0_, y1_):
        out0 = model.forward(params_, np.array([0.0]), x_)  # star when t = 0.0
        out1 = model.forward(params_, np.array([1.0]), x_)  # circle when t = 1.0
        loss_sdf = np.mean((out0 - y0_) ** 2) + np.mean((out1 - y1_) ** 2)
        loss_lipschitz = model.get_lipschitz_loss(params_)
        return loss_sdf + alpha * loss_lipschitz

    @jit
    def update(epoch, opt_state, alpha, x_, y0_, y1_):
        params_ = get_params(opt_state)
        value, grads = value_and_grad(loss, argnums=0)(params_, alpha, x_, y0_, y1_)
        opt_state = opt_update(epoch, grads, opt_state)
        return value, opt_state

    def save_loss():
        plt.close(1)
        plt.figure(1)
        plt.semilogy(loss_history[:epoch])
        plt.title('Reconstruction loss + Lipschitz loss')
        plt.grid()
        plt.savefig(hyper_params["save_loss_history_path"])

    nb_samples_total = 1000 * hyper_params["samples_per_epoch"]
    X = np.array(random.rand(nb_samples_total, hyper_params["dim_in"]))
    Y0, Y1 = get_training_samples(X)

    # training
    loss_history = onp.zeros(hyper_params["num_epochs"])
    pbar = tqdm.tqdm(range(hyper_params["num_epochs"]))
    for epoch in pbar:
        # sample a bunch of random points
        # x = np.array(random.rand(hyper_params["samples_per_epoch"], hyper_params["dim_in"]))
        # y0, y1 = get_training_samples(x)

        indices = random.choice(nb_samples_total, hyper_params["samples_per_epoch"], replace=False)
        x = X[indices]
        y0 = Y0[indices]
        y1 = Y1[indices]

        # update
        loss_value, opt_state = update(epoch, opt_state, hyper_params["alpha"], x, y0, y1)
        loss_history[epoch] = loss_value
        pbar.set_postfix({"loss": loss_value})

        if epoch % 1000 == 999:  # plot loss history every 1000 iter
            save_loss()

    save_loss()
    params = get_params(opt_state)
    return model, params


def normalize_V(V):
    V = V[:, [1, 2, 0]]
    V = V * np.array([[1, 1, 1]])

    V = V - np.min(V, axis=0)  # in [0, +inf[^d
    V = V / np.max(V)  # in [0, 1]^d
    V = V + (1 - np.max(V, axis=0) + np.min(V, axis=0)) / 2  # in [0, 1]^d, centered

    # import open3d as o3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1024, height=1024)
    #
    # opt: o3d.visualization.RenderOption = vis.get_render_option()
    # opt.background_color = np.array([1, 1, 1])
    # opt.mesh_show_back_face = True
    # opt.point_size = 1000 / 64
    # # opt.point_color_option = o3d.visualization.PointColorOption.
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(V)
    # vis.add_geometry(pcd)
    #
    # box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(1, 1, 1))
    # box.color = (0, 0, 0)
    # vis.add_geometry(box)
    #
    # view: o3d.visualization.ViewControl = vis.get_view_control()
    # view.set_zoom(1)
    # view.set_front([-0.3, 0.3, 0.6])
    # view.set_lookat([0.5, 0.5, 0.5])
    # # view.set_up([0, 0, 0])
    #
    # vis.run()
    # vis.destroy_window()
    return V


def get_training_samples_generator(hyper_params):
    if hyper_params["dim_in"] == 2:
        return lambda x: (jgp.sdf_star(x),
                          jgp.sdf_circle(x))
    elif hyper_params["dim_in"] == 3:
        v0, f0 = jgp.read_mesh(hyper_params["shape_0"])
        v1, f1 = jgp.read_mesh(hyper_params["shape_1"])

        v0, v1 = normalize_V(v0), normalize_V(v1)

        # print(v0.shape, f0.shape)
        # print(v1.shape, f1.shape)

        return lambda x: (jgp.signed_distance(x, v0, f0)[0],
                          jgp.signed_distance(x, v1, f1)[0])

    raise ValueError("dim_in must be 2 or 3")
