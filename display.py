def present_losses_over_epochs(neural_network, threshold=1000):
    """Prints the losses over epochs according epoch % threshold == 0."""
    for epoch in neural_network.losses.keys():
        if epoch % threshold == 0:
            print(f"Epoch {epoch}, Loss: {neural_network.losses[epoch]:.4f}")
