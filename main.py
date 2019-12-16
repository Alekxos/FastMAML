import torch
from load_data import DataGenerator
import click
from models.MatchingNetworkCNN import MatchingNetworkCNN
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.optim import SGD, Adam
import util
from os import path
import os
import time
from pathlib import Path
import random

# Runtime parameters
@click.command()
@click.option('--experiment_name', type=str, default='omniglot')
@click.option('--meta_train_iterations', type=int, default=15000, help='number of meta-training iterations')
@click.option('--meta_lr', type=float, default=0.001, help='outer loop learning rate')
@click.option('--inner_lr', type=float, default=0.04, help='inner loop learning rate')
@click.option('--meta_batch_size', type=int, default=32, help='number of tasks sampled for a single meta-step')
@click.option('--img_size', type=int, default=28, help='height=width of input images')
@click.option('--num_channels', type=int, default=1, help='number of channels in input images')
@click.option('--k_shot', type=int, default=1, help='number of examples used for inner gradient update training (K '
                                                    'for K-shot learning)')
@click.option('--n_way', type=int, default=5,
              help='number of classes used in classification (e.g. 5-way classification)')
@click.option('--data_path', type=str, default='./datasets/omniglot_resized', help='path to the dataset')
@click.option('--num_inner_updates', type=int, default=1, help='number of inner gradient updates during inner loop training')
@click.option('--output_dir', type=str, default='./output', help='directory to store output')

def main(experiment_name, meta_train_iterations, meta_lr, inner_lr, meta_batch_size, img_size, num_channels, k_shot, n_way, data_path,
         num_inner_updates, output_dir):
    # Training params
    train_print_spacing = 10
    # Create output directory
    if not path.exists(output_dir):
        os.mkdir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    custom_savename = f"{experiment_name}_K_{k_shot}_N_{n_way}_mlr_{meta_lr}_ilr_{inner_lr}_{timestamp}"
    current_run_savedir = str(Path(output_dir).joinpath(custom_savename))
    if not path.exists(current_run_savedir):
        os.mkdir(current_run_savedir)
    # Instantiate metric data structures
    tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies = [], [], [], []
    #tracked_meta_train_losses, tracked_meta_train_accuracies, tracked_meta_val_losses, tracked_meta_val_accuracies = [], [], [], []
    # Create engine to sample data from
    data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot * 2, config={'data_folder': data_path})
    # Instantiate base model (meta parameters) and optimizer
    model = MatchingNetworkCNN(n_way).double()
    meta_optimizer = Adam(model.parameters(), lr=meta_lr)
    for meta_step in range(meta_train_iterations):
        print(f"Meta-Step {meta_step}")
        # Sample meta-batch of data and samples for model initialization
        (meta_image_batch, meta_label_batch) = data_generator.sample_batch("meta_train", meta_batch_size, shuffle=True)
        sample_input, sample_label = meta_image_batch[0, 0, 0, :], meta_label_batch[0, 0, 0]
        sample_input = torch.tensor(sample_input).view(-1, 1, img_size, img_size)
        # Record results of inner loop training
        support_losses = []
        support_accuracies = []
        query_losses = []
        query_accuracies = []
        meta_gradients = []
        # TODO: Is following for loop parallellizable (akin to tf.map_fn)?
        for task_index in range(meta_batch_size):
            image_batch, label_batch = meta_image_batch[task_index], meta_label_batch[task_index]
            # Split into support/query
            support_input, query_input = image_batch[:, :k_shot, :], image_batch[:, k_shot:, :]
            support_labels, query_labels = label_batch[:, :k_shot, :], label_batch[:, k_shot:, :]
            # Reshape support/query and convert to Tensor
            support_labels = torch.argmax(torch.tensor(support_labels), dim=-1).view(n_way * k_shot)
            query_labels = torch.argmax(torch.tensor(query_labels), dim=-1).view(n_way * k_shot)
            support_input = torch.tensor(support_input).view(-1, 1, img_size, img_size)
            query_input = torch.tensor(query_input).view(-1, 1, img_size, img_size)
            # Instantiate adapted model parameters and optimizer
            adapted_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
            # Updated adapted model based on gradients from support dataset
            for inner_loop_num in range(num_inner_updates):
                # Forward input batch through adapted models
                support_output = model.forward(support_input, adapted_weights, cuda=False) if inner_loop_num != 0 else model.forward(support_input, cuda=False)
                support_output = support_output.view(n_way * k_shot, n_way)
                # Evaluate support (inner train) loss and accuracy
                support_loss = F.cross_entropy(support_output, support_labels)
                support_predictions = torch.argmax(support_output, dim=-1)
                support_accuracy = util.accuracy(support_predictions, support_labels)
                # Update adapted model
                if inner_loop_num == 0:
                    inner_grads = torch.autograd.grad(support_loss, model.parameters(), create_graph=True)
                else:
                    inner_grads = torch.autograd.grad(support_loss, adapted_weights.values(), create_graph=True)
                adapted_weights = OrderedDict((name, param - inner_lr * grad) for ((name, param), grad) in zip(adapted_weights.items(), inner_grads))

                # Track initial performance on support set
                if inner_loop_num == 0:
                    support_losses.append(support_loss)
                    support_accuracies.append(support_accuracy.item())
                if inner_loop_num == num_inner_updates - 1:
                    # Evaluate output and loss on query (inner test) dataset
                    query_output = model.forward(query_input, adapted_weights, cuda=False)
                    query_output = query_output.view(n_way * k_shot, n_way)
                    query_loss = F.cross_entropy(query_output, query_labels)
                    query_losses.append(query_loss)
                    # Keeping adapted model fixed, evaluate meta-gradients using query dataset
                    raw_task_meta_gradients = torch.autograd.grad(query_loss, model.parameters())
                    task_meta_gradients = {name: g for ((name, _), g) in zip(model.named_parameters(), raw_task_meta_gradients)}
                    meta_gradients.append(task_meta_gradients)
            # Also evaluate query accuracy to average over tasks and store later
            query_predictions = torch.argmax(query_output, dim=-1)
            query_accuracy = util.accuracy(query_predictions, query_labels)
            query_accuracies.append(query_accuracy)
        # Update model meta-parameters
        sample_output = model.forward(sample_input).view(1, n_way)
        sample_label = torch.argmax(torch.tensor(sample_label), dim=-1).view(1)
        #print(f"sample output: {sample_output.shape}, sample label: {sample_label.shape}")
        sample_loss = F.cross_entropy(sample_output, sample_label)
        # Apply meta-PCGrad update rule
          # Iterate through meta-gradients in random order
        shuffled_order = list(range(meta_batch_size))
        random.shuffle(shuffled_order)
          # Precompute magnitudes of meta-gradient subspaces
        meta_gradient_magnitudes = []
        for task_meta_gradient in meta_gradients:
            meta_gradient_subspace_magnitude = {}
            for meta_grad_key in task_meta_gradient.keys():
                meta_gradient_subspace_magnitude[meta_grad_key] = torch.sqrt(torch.sum(torch.mul(task_meta_gradient[meta_grad_key], task_meta_gradient[meta_grad_key])))
            meta_gradient_magnitudes.append(meta_gradient_subspace_magnitude)
        for first_iteration_index, first_meta_grad_index in enumerate(shuffled_order):
            first_meta_gradient = meta_gradients[first_meta_grad_index]
            for second_meta_grad_index in shuffled_order[first_iteration_index:]:
                second_meta_gradient = meta_gradients[second_meta_grad_index]
                for meta_grad_key in first_meta_gradient.keys():
                    # Compute cosine product between m_i and m_j
                    dot_product = torch.sum(torch.mul(first_meta_gradient[meta_grad_key], second_meta_gradient[meta_grad_key]))
                    #cosine_similarity = dot_product / (meta_gradient_magnitudes[first_meta_grad_index] * meta_gradient_magnitudes[second_meta_grad_index])
                # If negative, update m_i
                    if dot_product < 0:
                        first_meta_gradient[meta_grad_key] = first_meta_gradient[meta_grad_key] - dot_product /  meta_gradient_magnitudes[second_meta_grad_index][meta_grad_key]**2 * second_meta_gradient[meta_grad_key]
            meta_gradients[first_meta_grad_index] = first_meta_gradient
            # Update individual keys/values at a time or entire 'vector' of key/value pairs?

        meta_gradient_dict = {k: sum(d[k] for d in meta_gradients) for k in meta_gradients[0].keys()}
        hooks = []
        for (k, v) in model.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return meta_gradient_dict[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        meta_optimizer.zero_grad()
        sample_loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        meta_optimizer.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()
        # Update metrics list
        tracked_support_losses.append(np.mean([support_loss.item() for support_loss in support_losses]))
        tracked_support_accuracies.append(np.mean(support_accuracies))
        tracked_query_losses.append(np.mean([query_loss.item() for query_loss in query_losses]))
        tracked_query_accuracies.append(np.mean(query_accuracies))
        if meta_step % train_print_spacing == 0:
            print(
                f"Iteration {meta_step}: pre-inner-loop accuracy: {tracked_support_accuracies[-1]}, post-inner-loop-accuracy: {tracked_query_accuracies[-1]}")
            print(
                f"Iteration {meta_step}: pre-inner-loop loss: {tracked_support_losses[-1]}, post-inner-loop loss: {tracked_query_losses[-1]}")
        # Save metrics to file
        np.save(f'{current_run_savedir}/support_loss.npy', np.array(tracked_support_losses))
        np.save(f'{current_run_savedir}/support_accuracy.npy', np.array(tracked_support_accuracies))
        np.save(f'{current_run_savedir}/query_loss.npy', np.array(tracked_query_losses))
        np.save(f'{current_run_savedir}/query_accuracy.npy', np.array(tracked_query_accuracies))
        # Use learned model for meta validation
        #meta_validation_model = MatchingNetworkCNN(n_way).double()
if __name__ == '__main__':
    main()
