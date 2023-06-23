.. figure:: docs/images/atomcloud_logo.jpg

AtomCloud: GPU accelerated atom cloud analysis 
==============================================

`Quickstart <#quickstart-colab-in-the-cloud>`__ \| `Install
guide <#installation>`__ \| `API Docs <https://atomcloud.readthedocs.io/>`__

What is AtomCloud?
------------------

AtomCloud is a Python package designed to streamline the fitting of atom cloud images. While single function atom cloud fits can be manageable, things can become more challenging when dealing with complex multi-function fits and multiple stages of fitting. This is where AtomCloud shines. It creates an easy-to-use interface for these fits and abstracts away unnecessary details—all while providing the user with a high degree of control over the fitting process.

AtomCloud is built on top of the JAXFit fitting library, which provides GPU-accelerated fitting. This means fit speed-ups can be 10-100 times faster than traditional CPU-based fitting.

AtomCloud offers a wide spectrum of fitting capabilities. These include built-in 1D and 2D fit functions for common atom cloud distributions such as thermal clouds, condensates, and bimodal clouds. It also provides the flexibility to constrain fit parameters in multi-function fits. In the realm of multi-level fits, users can easily stack multiple fit functions together to create a custom-tailored fit.

However, the functionality of AtomCloud doesn't stop at fitting. We've also integrated a variety of analysis tools, such as fit parameter rescaling, integration of fitted density distributions, and temperature extraction.

Lastly, all the fitting and analysis tools natively incorporate error propagation, making experimental error analysis a breeze.

Contents
~~~~~~~~

-  `Quickstart: Colab in the Cloud <#quickstart-colab-in-the-cloud>`__
-  `Installation <#installation>`__
-  `API Docs <https://atomcloud.readthedocs.io/>`__

Quickstart: Colab in the Cloud
------------------------------

The easiest way to test out AtomCloud is using a Colab notebook. 
We have a few tutorial notebooks including: 

- `Multi-Function Basics <https://colab.research.google.com/github/lucashofer/atomcloud/blob/main/docs/notebooks/Multi_Functions.ipynb>`__
- `Multi-Function Fits <https://colab.research.google.com/github/lucashofer/atomcloud/blob/main/docs/notebooks/Multi_Function_Fits.ipynb>`__
- `Sum Fits <https://colab.research.google.com/github/lucashofer/atomcloud/blob/main/docs/notebooks/Sum_Fits.ipynb>`__
- `Multi-Level-Fits <https://colab.research.google.com/github/lucashofer/atomcloud/blob/main/docs/notebooks/Multi_Level Fits.ipynb>`__



Installation
------------

AtomCloud is written in Python and can be installed via pip

::

   pip install atomcloud


In order to utilize the GPU accelerated fitting, you will need to install 
JAX and JAXFit as without it AtomCloud will default to CPU fitting using SciPy. JAXFit is written in pure Python and is based on the JAX package. JAX therefore needs to be installed before installing JAXFit via pip. JAX installation requires a bit of effort since it is optimized for the computer hardware you'll be using (GPU vs. CPU). 

Installing JAX on Linux is natively supported by the JAX team and instructions to do so can be found `here <https://github.com/google/jax#installation>`_. 

For Windows systems, the officially supported method is building directly from the source code (see `Building JAX from source <https://jax.readthedocs.io/en/latest/developer.html#building-from-source>`_). However, we've found it easier to use pre-built JAX wheels which can be found in `this Github repo <https://github.com/cloudhan/jax-windows-builder>`_ and we've included detailed instructions on this installation process below.

After installing JAX, you can now install JAXFit via the following pip command

::

    pip install jaxfit


Windows JAX install
~~~~~~~~~~~~~~~~~~~

If you are installing JAX on a Windows machine with a CUDA compatible GPU then 
you'll need to read the first part. If you're only installing the CPU version

Installing CUDA Toolkit
^^^^^^^^^^^^^^^^^^^^^^^

If you'll be running JAX on a CUDA compatible GPU you'll need a CUDA toolkit 
and CUDnn. We recommend using an Anaconda environment to do all this installation.

First make sure your GPU driver is CUDA compatible and that the latest NVIDIA 
driver has been installed.

To create a Conda environment with Python 3.9 open up Anaconda Prompt and do the 
following:

::

    conda create -n jaxenv python=3.9

Now activate the environment

::

    conda activate jaxenv

Since all the the pre-built Windows wheels rely on CUDA 11.1 and CUDnn 8.2, we 
use conda to install these as follows

::

    conda install -c conda-forge cudatoolkit=11.1 cudnn=8.2.0

However, this toolkit doesn't include the developer tools which JAX also need 
and therefore these need to be separately installed using

::

    conda install -c conda-forge cudatoolkit-dev

Pip installing pre-built JAX wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pick a jaxlib wheel from the CloudHan repo's list 
of `pre-built wheels <https://whls.blob.core.windows.net/unstable/index.html>`_. 
We recommend the latest build (0.3.14) as we've had issues with earlier 
versions. The Python version of the wheel needs to correspond to the conda 
environment's Python version (e.g. cp39 corresponds to Python 3.9 for our 
example) and pip install it. Additionally, you can pick a GPU version (CUDA111) 
or CPU only version, but we pick a GPU version below.

::

    pip install https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.14+cuda11.cudnn82-cp39-none-win_amd64.whl

Next, install the JAX version corresponding to the jaxlib library (a list of 
jaxlib and JAX releases can be found `here <https://github.com/google/jax/blob/main/CHANGELOG.md>`_)

::

    pip install jax==0.3.14



API Documentation
-----------------------

For details about the AtomCloud API, see the `API Docs <https://atomcloud.readthedocs.io/>`__.