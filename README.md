# Impact of Weight-Space Symmetries on Neural Network Training

Weight configurations where two or more weight vectors are identical have been studied as overlap singularities. Furthermore, if these points are critical points of the loss function being optimized, they are termed as permutation points and have been proven to be saddle points. My [Master thesis](https://www.dropbox.com/s/8z1i7377fglockg/Master_thesis_Manu_Srinath_Halvagal.pdf?dl=0) is an attempt to empirically study if and when such pathological configurations have a practical impact on neural network optimization.


## Using this package

**Requirements:** PyTorch, NumPy, Matplotlib

To generate the simulations described in the thesis, modify appropriately the <experimental_settings.yaml> file and run <main.py>. The jupyter notebooks provided also present some examples of how to use the plotting modules.