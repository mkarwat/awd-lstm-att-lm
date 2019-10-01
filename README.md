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

[WT2_att.pt](https://box.cyfronet.pl/s/2GXey8KdJYNz8yn)
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
For WikiText-2 run:
```shell script
python infer_new.py --data data/wikitext-2 --load WT2_att.pt
```
### Results
|      |                           | Number of params | test ppl | valid ppl |
|------|---------------------------|------------------|----------|-----------|
| PTB  | awd-lstm-att-lm           | 27M              | 45.22    | 46.50     |
|      | awd-lstm-att-lm + pruning | 19M              | 45.98    |           |
| WT-2 | awd-lstm-att-lm           | 40M              | 39.12    | 40.75     |
|      | awd-lstm-att-lm + pruning | 27M              | 38.44    |           |