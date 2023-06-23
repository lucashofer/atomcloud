AtomCloud is a Python package designed to streamline the fitting of atom cloud images. While single function atom cloud fits can be manageable, things can become more challenging when dealing with complex multi-function fits and multiple stages of fitting. This is where AtomCloud shines. It creates an easy-to-use interface for these fits and abstracts away unnecessary details—all while providing the user with a high degree of control over the fitting process.

AtomCloud is built on top of the JAXFit fitting library, which provides GPU-accelerated fitting. This means fit speed-ups can be 10-100 times faster than traditional CPU-based fitting.

AtomCloud offers a wide spectrum of fitting capabilities. These include built-in 1D and 2D fit functions for common atom cloud distributions such as thermal clouds, condensates, and bimodal clouds. It also provides the flexibility to constrain fit parameters in multi-function fits. In the realm of multi-level fits, users can easily stack multiple fit functions together to create a custom-tailored fit.

However, the functionality of AtomCloud doesn't stop at fitting. We've also integrated a variety of analysis tools, such as fit parameter rescaling, integration of fitted density distributions, and temperature extraction.

Lastly, all the fitting and analysis tools natively incorporate error propagation, making experimental error analysis a breeze.

[https://github.com/lucashofer/atomcloud](https://github.com/lucashofer/atomcloud)
