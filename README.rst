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
- `Multi-Level-Fits <https://colab.research.google.com/github/lucashofer/atomcloud/blob/main/docs/notebooks/Multi_Level_Fits.ipynb>`__



Installation
------------

AtomCloud is written in Python and can be installed via pip

::

   pip install atomcloud


In order to utilize the GPU accelerated fitting, you will need to install 
JAX >= 0.4.4 and JAXFit >= 0.0.5 as without them AtomCloud will default to CPU fitting using SciPy. JAXFit is written in pure Python and is based on the JAX package. JAX therefore needs to be installed before installing JAXFit via pip. JAX installation requires a bit of effort since it is optimized for the computer hardware you'll be using (GPU vs. CPU). 

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

First, create a Conda environment with Python 3.11 open up Anaconda Prompt and do the 
following:

::

    conda create -n jaxenv python=3.11

For Windows, Python 3.11 is required to install the lastest GPU compatible pre-built JAX wheel. 
Now ;et's activate the environment

::

    conda activate jaxenv

Since all the pre-built Windows wheels for JAX 0.4.11 rely on CUDA >= 11.7 and CUDnn >= 8.6, we 
use conda to install these as follows

::

    conda install -c conda-forge cudatoolkit=11.7 cudnn=8.8.0 cudatoolkit-dev

Pip installing pre-built JAX wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pick a jaxlib wheel from the CloudHan repo's list 
of `pre-built wheels <https://whls.blob.core.windows.net/unstable/index.html>`_ and pip install it.
We recommend the latest build (0.4.11) as versions below 0.4.4 won't work with AtomCloud. The Python 
version of the wheel needs to correspond to the conda environment's Python version (e.g. cp311 
corresponds to Python 3.11 for our example). Additionally, you can pick a GPU version 
or CPU only version, but we pick a GPU version below.

::

    pip install https://whls.blob.core.windows.net/unstable/cuda/jaxlib-0.4.11+cuda.cudnn86-cp311-cp311-win_amd64.whl

Next, install the JAX version corresponding to the jaxlib library (a list of 
jaxlib and JAX releases can be found `here <https://github.com/google/jax/blob/main/CHANGELOG.md>`_) as well as jaxfit

::

    pip install jax==0.4.11 jaxfit



API Documentation
-----------------------

For details about the AtomCloud API, see the `API Docs <https://atomcloud.readthedocs.io/>`__.