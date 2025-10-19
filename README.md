# TPP-SD: Accelerating Transformer Point Process Sampling with Speculative Decoding

Pytorch implementation of the paper "TPP-SD: Accelerating Transformer Point Process Sampling with Speculative Decoding".

## Environment Configuration

In order to run the code, you need to run
```bash
conda create -n tpp-sd python=3.9
pip install -r requirements.txt
```

## Running the Code
### Synthetic Dataset Generation
For synthetic data generation, you can first modify the parameters in `code/generate_dataset.py` and then run the following command to generate the dataset:
```bash
cd code
python generate_dataset.py
```
We support the generation of Poisson process, Uni-variate Hawkes process and Multi-variate Hawkes process. The generated dataset will be saved in the `data/synth` folder by default.

### Training
For training the model, run
```bash
cd code
python train.py --config scripts/train_config_{dataset}.yaml
```
where `{dataset}` can be `inhomo_poi`(Poisson process), `myhawkes`(Uni-variate Hawkes process) or `multi_hawkes`(Multi-variate Hawkes process). 

You can also modify the training scripts in `code/scripts` folder to customize the training process, including the choice of encoder type (THP, SAHP, AttNHP) and model hyperparameters (Mixture components of Log-normal, etc.).

### TPP-SD: Accelerating Transformer Point Process Sampling with Speculative Decoding

To run the accelerated sampling with speculative decoding, you can run the following command:
```bash
cd code
python sd_sampling_exp.py --config scripts/sd_config_{dataset}.yaml
```
where `{dataset}` can be `inhomo_poi`(Poisson process), `myhawkes`(Uni-variate Hawkes process) or `multi_hawkes`(Multi-variate Hawkes process). The results, including the likelihood comparison plot, the Kolmogorov-Smirnov plot (KS plot) and the sampling comparison plot, will be saved in the `code/plots` folder by default.