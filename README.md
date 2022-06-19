# Federated learning with VAE model using MNIST dataset


### example
```sh
# preparation
python3 generate_noniid.py　　# Generate Non-IID train data filter

# train models
python3 central.py  #　Train a single model using all of the non-distributed dataset.
python3 fl.py  # Perform federated learning using pre-created Non-IID filter

# visualize
python3 generate_images.py  # Generate images using trained models
```

Try `-h` option for more information about the usage.
