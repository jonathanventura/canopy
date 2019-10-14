# canopy
Automatic tree species classification from remote sensing data

Code from our paper:

[Fricker, G. A., Ventura, J. D., Wolf, J. A., North, M. P., Davis, F. W., & Franklin, J. (2019). A Convolutional Neural Network Classifier Identifies Tree Species in Mixed-Conifer Forest from Hyperspectral Imagery. Remote Sensing, 11(19), 2326.](https://www.mdpi.com/2072-4292/11/19/2326)

---

To run the example experiment:

    mkdir hyperspectral ; 
    python -m experiment.download ;
    python -m experiment.preprocess --out hyperspectral;
    python -m experiment.train --out hyperspectral;
    python -m experiment.test --out hyperspectral;
    python -m experiment.analyze --out hyperspectral;
