# awd-lstm-att-lm

awd-lstm-att-lm is based on [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm), all unmodified code is under its original license. 

## Running
### Setup
```shell script
conda create --name my_env python=3.7
conda activate my_env
conda install pytorch=0.4.1 cuda90 -c pytorch
```
### Models
Download trained models:

[PTB_att.pt](https://box.cyfronet.pl/s/Nfna7FL7wC72zf8)

[PTB_att_pruned.pt](https://box.cyfronet.pl/s/rcTjb2WXk5PobiC)

[WT2_att.pt](https://box.cyfronet.pl/s/TxarCDf9wyq3qnn)

[WT2_att_pruned.pt](https://box.cyfronet.pl/s/mycmxzsAArjHR88)
### Data
To get data run:
```shell script
./getdata.sh
```
### Inference
For PTB run:
```shell script
python infer_new.py --data data/penn --load PTB_att.pt
```
For PTB pruned run:
```shell script
python infer_new.py --data data/penn --load PTB_att_pruned.pt
```
For WikiText-2 run:
```shell script
python infer_new.py --data data/wikitext-2 --load WT2_att.pt
```
For WikiText-2 pruned run:
```shell script
python infer_new.py --data data/wikitext-2 --load WT2_att_pruned.pt
```
### Results
|      |                           | Number of params | test ppl | valid ppl |
|------|---------------------------|------------------|----------|-----------|
| PTB  | awd-lstm-att-lm           | 28.7M            | 45.22    | 46.50     |
|      | awd-lstm-att-lm + pruning | 18.5M            | 45.94    | 47.29     |
| WT-2 | awd-lstm-att-lm           | 47.1M            | 35.27    | 36.97     |
|      | awd-lstm-att-lm + pruning | 31.9M            | 38.31    | 40.07     |