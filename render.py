import numpy as onp
import jax.numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

import open3d as o3d

import jaxgptoolbox as jgp


def render_video_2d(hyper_params, model, params, min_t=0, max_t=1, nb_frames=50):
    """ Renders 2D neural SDF using Matplotlib. """
    fig = plt.figure()
    x = jgp.sample_2D_grid(hyper_params["grid_size"])  # sample on unit grid for visualization

    sdf_cm = mpl.colors.LinearSegmentedColormap.from_list('SDF', [(0, '#eff3ff'), (0.5, '#3182bd'), (0.5, '#31a354'),
                                                                  (1, '#e5f5e0')], N=256)

    def animate(t):
        plt.cla()
        out = model.forward_eval(params, np.array([t]), x)
        levels = onp.linspace(-0.5, 0.5, 21)
        im = plt.contourf(out.reshape(hyper_params['grid_size'], hyper_params['grid_size']), levels=levels, cmap=sdf_cm)
        plt.axis('equal')
        plt.axis("off")
        return im

    anim = animation.FuncAnimation(fig, animate, frames=np.linspace(min_t, max_t, nb_frames), interval=50)
    anim.save(hyper_params["output_video_path"])


def render_video_3d(hyper_params, model, params, min_t=0, max_t=1, nb_frames=50):
    """ Renders 3D neural SDF using Open3D and Matplotlib. """
    x = sample_3D_grid(hyper_params['grid_size'])

    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(1, 1, 1))
    box.color = (0, 0, 0)

    t_index = 0
    t_lst = np.linspace(min_t, max_t, nb_frames)

    all_images = []
    color = (0.725, 0.843, 0.843)

    def generate_next_sdf(vis: o3d.visualization.Visualizer):
        nonlocal t_index
        t = t_lst[t_index]
        print(t)

        sdf_pred = model.forward_eval(params, np.array([t]), x)

        # Render using point clouds (much faster but approximate)
        points = sdf_to_point_cloud(x, sdf_pred, hyper_params['grid_size'])

        # Render using a mesh
        # geometry = point_cloud_to_mesh(points, hyper_params['grid_size'])
        # geometry.paint_uniform_color(color)

        # Render using voxels (quite slow)
        # geometry = point_cloud_to_voxels(points, hyper_params['grid_size'], color)
        # size = 1 / hyper_params['grid_size']
        # geometries = [o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size).translate(p) for p in points]

        # Reset view
        vis.clear_geometries()
        vis.add_geometry(box)

        # Add point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)

        # Add other kinds of geometries (mesh or voxels)
        # vis.add_geometry(geometry)
        # for g in geometries:
        #     g: o3d.geometry.TriangleMesh
        #     g.compute_vertex_normals()#
        #     vis.add_geometry(g)

        view: o3d.visualization.ViewControl = vis.get_view_control()
        view.set_zoom(1)
        view.set_front([-0.3, 0.3, 0.6])
        view.set_lookat([0.5, 0.5, 0.5])

        img = vis.capture_screen_float_buffer(True)
        all_images.append(img)

        t_index += 1
        if t_index >= len(t_lst):
            # stop animation
            vis.close()

        return False

    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(generate_next_sdf)
    vis.create_window(width=1024, height=1024)

    opt: o3d.visualization.RenderOption = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.mesh_show_back_face = True
    opt.point_size = 1000 / hyper_params['grid_size']
    # opt.point_color_option = o3d.visualization.PointColorOption.

    vis.run()
    vis.destroy_window()

    def animate(i):
        plt.cla()
        plt.imshow(all_images[i])
        plt.axis('equal')
        plt.axis("off")
        return all_images[i]

    # save video
    fig = plt.figure()
    fig.set_size_inches(10.24, 10.24)
    writer = animation.FFMpegWriter(fps=10)
    anim = animation.FuncAnimation(fig, animate, frames=len(all_images), interval=50)
    anim.save(hyper_params["output_video_path"], writer=writer, dpi=100)


def render_video_3d_2(hyper_params, model, params, min_t=0, max_t=1, nb_frames=50):
    """ Renders 3D neural SDF using Matplotlib only (slower). """
    x = sample_3D_grid(hyper_params['grid_size'])

    t_lst = np.linspace(min_t, max_t, nb_frames)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    grid_size = hyper_params['grid_size']

    def animate(t):

        sdf_pred = model.forward_eval(params, np.array([t]), x)
        points = sdf_to_point_cloud(x, sdf_pred, grid_size)

        ax.cla()
        print("voxels", points.shape)
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
        print("done")
        return im

    print("start animation")
    animate(0.5)
    print("end animation")
    plt.show()

    # save video
    fig.set_size_inches(10.24, 10.24)
    writer = animation.FFMpegWriter()
    anim = animation.FuncAnimation(fig, animate, frames=t_lst, interval=50)
    print(anim)
    anim.save(hyper_params["output_video_path"], writer=writer, dpi=100, fps=10)


def sdf_to_point_cloud(points, sdf, grid_size, remove_boundary=False):
    is_solid = (sdf <= 0).reshape((grid_size, grid_size, grid_size))
    is_diff_nonzero = False
    out_grid = is_solid.astype(float)

    for i in range(3):
        diff = onp.diff(out_grid, n=2, axis=i, prepend=remove_boundary, append=remove_boundary)
        is_diff_nonzero = is_diff_nonzero | (diff != 0)

    mask = (is_solid & is_diff_nonzero).flatten()
    return points[mask]


def sample_3D_grid(resolution, low=0, high=1):
    idx = onp.linspace(low, high, num=resolution)
    x, y, z = onp.meshgrid(idx, idx, idx)
    V = onp.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    return np.array(V)


def point_cloud_to_voxels(points, grid_size, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # pcd.colors = o3d.utility.Vector3dVector(onp.repeat(onp.array(color)[None], len(points), axis=0))
    pcd.colors = o3d.utility.Vector3dVector(onp.random.random(points.shape))

    voxel_grid: o3d.geometry.VoxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1 / grid_size)
    # voxel_grid.compute_vertex_normals()
    # voxel_grid.get_voxels()[0] &
    return voxel_grid


def point_cloud_to_mesh(points, resolution):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Using alpha shapes
    radius = 3 / resolution
    alpha = radius
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(10)  # to do: fine-tune this parameter

    # Using ball pivoting
    # smallest_distance = 1 / resolution
    # radii = smallest_distance * onp.array([1, 3])
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Apply filter to smooth mesh
    # mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    # mesh = mesh.filter_smooth_simple(number_of_iterations=10)

    mesh.compute_vertex_normals()  # for shadows
    return mesh
