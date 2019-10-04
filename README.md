# canopy
Automatic tree species classification from remote sensing data

To run the example experiment:

    mkdir hyperspectral ; 
    python -m experiment.download ;
    python -m experiment.preprocess --out hyperspectral;
    python -m experiment.train --out hyperspectral;
    python -m experiment.test --out hyperspectral;
    python -m experiment.analyze --out hyperspectral;
