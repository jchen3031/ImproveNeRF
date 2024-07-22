import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from run_nerf_helpers import init_nerf_model
import numpy as np
from load_llff import load_llff_data


def compute_loss(model, task_inputs, task_targets):
    predictions = model(task_inputs)

    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(task_targets, predictions)

    return loss


def generate_task_data(scene_data, num_views_per_task):
    indices = np.random.choice(len(scene_data['images']), num_views_per_task, replace=False)
    task_views = [scene_data['images'][i] for i in indices]
    task_poses = [scene_data['poses'][i] for i in indices]

    return task_views, task_poses


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, factor=8,
                                                              recenter=True, bd_factor=.75)
    scene_data = {
        'images': [images],
        'poses': [poses]
    }
    model = init_nerf_model(
        D=8, W=256, use_viewdirs=True, transfer=False)

    learning_rate = 0.001
    meta_step_size = 0.01
    num_iterations = 500
    num_adaptation_steps = 5
    batch_size = 1
    num_views_per_task = 2

    optimizer = optimizers.Adam(learning_rate)

    for iteration in range(num_iterations):

        x, y = generate_task_data(scene_data, num_views_per_task)

        with tf.GradientTape() as meta_tape:
            task_model = tf.keras.models.clone_model(model)
            task_model.set_weights(model.get_weights())

            with tf.GradientTape() as task_tape:
                task_loss = compute_loss(task_model, x, y)

            gradients = task_tape.gradient(task_loss, task_model.trainable_variables)
            task_optimizer = optimizers.SGD(learning_rate=meta_step_size)
            task_optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))

            adapted_loss = compute_loss(task_model, x, y)

        meta_gradients = meta_tape.gradient(adapted_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {adapted_loss.numpy()}")

    model.save('meta_model.h5')
