# Code for "Simulation-based inference captures non-Markovian effects as exempliﬁed in protein production kinetics through cell division"

This repository contains the code associated with the [PNAS-published](https://doi.org/10.1073/pnas.2517309123) article  **"Simulation-based inference captures non-Markovian effects as exempliﬁed in protein production kinetics through cell division"**

In this work, we introduce a neural network–based inference framework that estimates likelihoods from simulation data, enabling inference in biological models that are non-Markovian or otherwise have intractable likelihoods.

You can fin the final version on PNAS:[https://www.pnas.org/doi/10.1073/pnas.2517309123](https://www.pnas.org/doi/10.1073/pnas.2517309123) 

<div class="image-container" style="display:inline-block; padding:10px; background-color:white;">
  <img src="https://github.com/PessoaP/FC-inference/blob/main/f1.png?raw=true" width="600" style="background-color:white;"/>
</div>


---

## Directory Structure

- `model1_DCD/` – Deterministic Cell Division (Model 1)
- `model2_SCD/` – Stochastic Cell Division (Model 2)
- `model3_FCYeast/` – Fluorescence-based flow cytometry inference (Model 3)
- `real/` – Inference from experimental *S. cerevisiae* data

---

## How to Run

### Models 1 and 2

To reproduce the results for **Model 1** (Figure 3 in the manuscript), run:

```bash
cd model1_DCD
python 1DCD_training.py
```

Then run the Jupyter notebook `2DCD_figure.ipynb` 
which generates Fig3 of the manuscript.

Analogously for **Model 2** run the rescpective files in the `model2_SCD1` directory


### Real Data Preprocessing

To run inference on experimental flow cytometry data, place the data file  found 
[here](https://drive.google.com/file/d/1ZkPdYIHGolHsSyp6VHx1ooLljS5-ofEd/view?usp=sharing)
into the folder `clean_data/`  then run the script:

```bash
cd real
python clean_data.py
```

This will clean and preprocess the raw flow cytometry data for use in the inference in the real model and the autofluorescence calibration (in both **Model 3** and real data)


### Models 3 and Real

In both the `model3_FCYeast/` and `real/` directories, a `bash.sh` script is provided to automate the workflow.  
This script prepares the data, trains the necessary normalizing flows, and runs MCMC inference.

To execute, navigate into the respective folder and run:

```bash
cd model3_FCYeast
bash bash.sh

cd real
bash bash.sh
```

These will generate the results and trained models used to produce Figures 5 and 6 of the manuscript.
The corresponding Jupyter notebooks in each directory can then be used to visualize and export the figures.

## Requirements

This codebase builds on the [`normflows`](https://github.com/bayesiains/normflows) library.  
Please make sure it is installed before running the scripts.

---

## Any questions?

If you encounter any issues or have questions about the code or manuscript, feel free to open an issue or reach out to the authors via the contact information provided in the paper.

---

## Citation

If you find this work useful, we appreciate the citation. Here's the BibTeX:
```
@article{pessoa2025simulation,
      author = {Pedro Pessoa  and Juan Andres Martinez  and Vincent Vandenbroucke  and Frank Delvigne  and Steve Pressé },
      title = {Simulation-based inference captures non-Markovian effects as exemplified in protein production kinetics through cell division},
      journal = {Proceedings of the National Academy of Sciences},
      volume = {123},
      number = {15},
      pages = {e2517309123},
      year = {2026},
      doi = {10.1073/pnas.2517309123}
}
```
