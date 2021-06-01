
## Dependencies  
- python>=3.7
- torch==1.5.1
- transformers==4.3.0
- spacy==
- nltk==
- gensim==
  
## Dataset
|       | Train  | Test| 
| ----- | ------ | --- |  
|ORF    | 11,949 |2,988| 


## Model Architecture  
  
#### 1.BERT-DNN
![bert-dnn](https://github.com/yinghao1019/image/blob/master/BERT-DNN.png)   
    - bert layer's model be in common use 
    - We used [sep] to combine company profile/benefits(desc/require)
    - when text field is empty,we use spec ['empty'] to replace
#### 2.GRU-DNN
![GRU-DNN](https://github.com/yinghao1019/image/blob/master/GRU-DNN%20.png)
    - The GRU model is independent.
    - Use [sep] to represent sentence bound.
    - We use [pad] to process empty string
    - Attenion part
     1.use scaled-dot-product attention mechanism(query=title embed,value=GRU desc output hidden)

### 3.Universal setting
    - The title embed not use pad to align,and combine to one vector using Mean pooling
    - use concatenate to combine diff hidden vector
    - Meta data contain topic_distr(3),company_profile & desc wordNume,has link(desc),
      edu/job level,has lower edu/job,has logo,telecomuting

## Training Architecture
![](https://github.com/yinghao1019/image/blob/master/training%20architecture.png)  
    - In extract stage,filter word,tokenize,lemmatization,NER tag.Also extract
      
    - After extract,build vocab list which set max num & lowest freq,
      and then build pretrain vocab embed.
    - Using build vocab list & first stage process text to build topic model
    - Using build topic_model,vocab list,and specific tokenizer to convert to
      specfic format dataset
    - Using dataset to create data stream to training Model.

## Build model command (Usage)
  
```bash
#For build vocab
$ python vocab_process.py --task {task_name} \
                          --mode {mode} \
                          --select_context_name {context name}\
                          --select_item_name {item name}\
                          --embed_dim {Dim}\
                          --embed_type {embed model type}\
                          --spec_first

#For build topic model
$ python topic_model.py --eval_mode {u_mass} \
                        --select column {column_name} \
                        --best_topic {topic_num} \
                        --do_train

#For Train MOdel
$ python main.py --task fakeJob \
                 --pos_weights {weights} \
                 --max_textLen {textLen} \
                 --used_model {model_name}
                 --do_train --do_eval 
```  
  
## Default hyperparams setting  
- BERT-DNN: lr=1e-3 embed_type cbow embed_dim 256 hid_dim 512 MaxTextLen 64 warm_steps 128 max_norm 1 dropout 0.3
- GRU-DNN: lr=1e-3 embed_type cbow embed_dim 128 hid_dim 256 MaxTextLen 64 warm_steps 128 max_norm 1 dropout 0.3
## Results  
- use Linear lr rate warm up strategy
- split train set to val set(0.8/0.2)

|           |                  | balance acc (%)| F1_score (%)| precision (%) | recall(%) | TNR (%) |
| --------- | ---------------- | -------------- | ----------- | ------------- | --------- | ------- |
| **val**   | BERT-DNN         | 79.57          | 43.08       | 32.23         |   64.85   |  98.46  |
|           | GRU-DNN          | 82.92          | 57.9        | 50.25         |   68.72   |  98.66  |
| **test**  | BERT- DNN        | 78.76          | 41.97       | 31.41         |   63.33   |  98.39  |
|           | GRU-DNN          | 82.53          | 61.05       | 52.3          |   73.33   |  98.8   |
