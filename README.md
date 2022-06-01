# NumNet+, initialized with GenBERT's weights

In this fork we implemented BERT, initialized with GenBERT's weights, as NumNet+'s encoder.

GenBERT: [[CODE]](https://github.com/ag1988/injecting_numeracy), [[PDF]](https://arxiv.org/pdf/2004.04487.pdf)

---

## Requirements

`pip install -r requirements.txt`

## Usage
### Data and pretrained BERT preparation.
- Download drop data.
  
  `wget -O drop_dataset.zip https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip`
  
  `unzip drop_dataset.zip`

- Download GenBERT pre-trained model.
 
  `cd drop_dataset`
  
  `pip install --upgrade --no-cache-dir gdown`
  
  `gdown --folder https://drive.google.com/drive/folders/1-KmhWF4Jex4gyuz1J2VANd1ycmtsggQ3`
  
  
### Train 

- Train with simple multi-span extraction (NumNet+).

    `sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 mspan`
    
- Train with tag based multi-span extraction (NumNet+ v2).
    
    `sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 tag_mspan`

### Eval
- Assuming, the model is saved as `SAVE_PATH/model.pt`.
    
    - Simple multi-span extraction (NumNet+).
    
        `sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json mspan SAVE_PATH/model.pt drop_dataset/genbert`
    
    - Tag based multi-span extraction (NumNet+ v2).
    
        `sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json tag_mspan SAVE_PATH/model.pt drop_dataset/genbert` 
    
    
    `python drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json`
