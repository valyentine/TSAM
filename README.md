# Transformer with Sparse Adaptive Mask for Network Dismantling
This work presents ***Transformer with Sparse Adaptive Mask for Network Dismantling***, a novel Transformer-based model designed for network dismantling problem, accepted at [ECML PKDD 2025](https://ecmlpkdd.org/2025/).
![image](https://github.com/valyentine/img/blob/main/framework.png)
## Dependecies
Create conda environment:
```
conda create -n TSAM python=3.9
conda activate TSAM
```
Install pytorch and other dependencies:
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Usage
### Dismantling networks in the paper
1. unzip the `data.zip` and put them in the folder `./data`.
   
The file directory of networks is as follows:
```
├── realworld
│   ├── AirTraffic
|   |   ├── AirTraffic.txt
│   ├── ...
├── synthetic
│   ├── BA_1000_4
|   |   ├── 0.edge
|   |   ├── 1.edge
|   |   ├── ...
│   ├── ...
```
2. Specify the target network using the `--network` argument. For example:  
```
python train.py --network AirTraffic
```
Refer to `parameter.py` for additional configurable parameters.
### Dismantling your own networks
1. Use `complete.py` to preprocess `your_network.txt`. Configure the `network` parameter in the script to match your network name and execute it.
```
python complete.py
```
This will generate `your_network_new.txt`. Then place `your_network_new.txt` in the appropriate subdirectory under `./data`, following the existing directory structure.

2. Add your network to the `graph_type` dictionary in train.py and then you can dismantle your own network:
```
python train.py --network your_network
```

## Citation
Please cite our work if you find our code/paper is helpful to your work.
