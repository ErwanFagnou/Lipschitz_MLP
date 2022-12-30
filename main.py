import pickle

import jax

from render import *
from train import *


hyper_params = {
    "reload_model": False,
    "load_model_path": "./outputs/rabbit_to_cat_epochs_100000_lr_1e-4_alpha_1e-6_tanh/lipschitz_mlp.pkl",

    # Input
    "dim_in": 3,
    "shape_0": "data/rabbit.stl",
    "shape_1": "data/cat.obj",

    # Output
    "save_model_path": "outputs/lipschitz_mlp.pkl",
    "save_loss_history_path": "outputs/loss_history.png",
    "output_video_path": "outputs/video.avi",
    "grid_size": 32 * 2,  # grid size for rendering
    "nb_frames": 100,  # number of frames for rendering

    # Model
    "dim_t": 1,
    "dim_out": 1,
    "h_mlp": [64, 64, 64, 64, 64],
    "activation_fn": jax.nn.relu,  # jax.nn.relu, jax.nn.tanh

    # Training
    "step_size": 1e-3,
    "num_epochs": 100000,
    "samples_per_epoch": 512,

    # Lipschitz regularization
    "alpha": 1e-7,
}


if __name__ == '__main__':

    random.seed(1)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if hyper_params["reload_model"]:
        model = lipmlp(hyper_params)
        with open(hyper_params["load_model_path"], 'rb') as handle:
            params = pickle.load(handle)
    else:
        model, params = train(hyper_params)

        # save final parameters
        with open(hyper_params["save_model_path"], 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # normalize weights during test time
    params = model.normalize_params(params)

    # save result as a video
    if hyper_params['dim_in'] == 2:
        render_video_2d(hyper_params, model, params, min_t=0, max_t=1, nb_frames=hyper_params['nb_frames'])
    elif hyper_params['dim_in'] == 3:
        render_video_3d(hyper_params, model, params, min_t=0, max_t=1, nb_frames=hyper_params['nb_frames'])
        # render_video_3d_2(hyper_params, model, params, min_t=0, max_t=1, nb_frames=10)
