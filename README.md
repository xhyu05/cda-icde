# Cost-Sensitive Data Acquisition (CDA) for Incomplete Datasets

This repository contains the experimental functions conducted for the paper "Cost-Sensitive Data Acquisition for Incomplete Datasets". Here's an overview of the contents:

- **`conformal-risk-main`**: Contains the functions used for the rows' selection step.
- **`data`**: Includes three example datasets.
- **`notebooks`**: Includes hands-on Jupyter notebooks and outputs.
- **`utils`**: Contains functional blocks used in the experiments.

## Quick Start Example

To conduct a simple quick start example using the forest dataset from the UCI data repository, considering using the MICEforest model as the predictive model, a complete random missing mechanism, and a missing rate of 30%, follow these steps:

1. **Set Up Your Python Environment**:
   - If you are using a virtual environment, create and activate it:
     ```bash
     virtualenv venv
     source venv/bin/activate  # for Unix/Linux
     venv\Scripts\activate     # for Windows
     ```
   - If you are not using a virtual environment, skip this step.

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Project:**:
   ```bash
   python3 quick-start.py
   ```

   - The running process may take more than 10 minutes, depending on your hardware. You may soon get the sense of the benefit from employing a CDA algorithm for incomplete datasets.