From Stocks et al. 2002: 97% of all fires in Canada are smaller than 200 hectares in size.

$$ 200 \text{ ha} = 2000000m^2 \times {1 \text{ pixel} \over (60m)^2} \approx 556 \text{ pixels} \approx 24\text{ pixels}^2 $$

Would like the model to have a receptive field that is large enough to contain 97% of all fires in Canada, therefore want a receptive field of at least 24 pixels.

The effective receptive field ($erf$) of layer $t$ in a network can be defined as: $erf_t = k_t + (k_{t - 1} - 1)$ where $k_t$ is the kernel size of layer $t$.

In the case of a model with dilation kernel size should be replaced with the effective kernel size defined as $k + (k - 1)(d - 1)$ where $k$ is the kernel size and $d$ is the dilation rate.

The effective receptive field of UNet is largest at the end of the downstack.
