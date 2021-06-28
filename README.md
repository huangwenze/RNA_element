# RNA_element
# This model is used for detect RNA element for RNA function

## Data Preparing

### TSV format data file

See `data/utr5.tsv`.


### Generating training/validation set

```
./generate_data.sh utr5
```

## Training 

```shell
# train utr5 with a 64 hidden unit LSTM
./train.sh 64 utr5
```



## Saliency matrix


```shell 
./export_saliency.sh 64 utr5  
```
