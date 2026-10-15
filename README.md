# AtmoFuseNet: Physics-Guided Neural Networks for Air Quality Prediction

AtmoFuseNet is a PyTorch-based deep learning framework implementing physics-guided differential equation neural networks for accurate air quality forecasting.

## Framework
![AtmoFuseNet framework](img/AtmoFuseNet_Framework.png)

## Requirements
* scipy>=1.5.2
* numpy>=1.19.1
* pandas>=1.1.5
* pyyaml>=5.3.1
* torch>=1.7.1
* future>=0.18.2
* torchdiffeq>=0.2.0
* haversine
* torch_geometric
* arrow

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Data Preparation
The sample air quality data files should be placed into the `data/` folder for running the code:
- `data/station.csv`
- `data/train.npz`
- `data/val.npz`
- `data/test.npz`

Sample air quality data files are available at [Google Drive](https://drive.google.com/drive/folders/1RWOA7kaPFAgjedoszOLCAjoQR1XSv8Dt).

## Model Training and Evaluation
The following code can be used to train and evaluate the model:
```bash
python main.py --config_filename config.yaml
```

## Acknowledgements
AtmoFuseNet is built upon the open-source AirPhyNet implementation by Kethmi Hettige et al. (MIT License). Original work: K. Hettige et al., 'AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction', ICLR 2024. Please cite the original paper if you use this code.
