import pandas as pd
import re
import os
import gdown
import time
from transformers import BertTokenizer,BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

PRICE=0
ROOM=1
LOCATION=2
FOOD=3
FACILITIES=4
STAFF=5
idx2topic={PRICE:"price",ROOM:"room",LOCATION:"location",FOOD:"food",FACILITIES:"facilities",STAFF:"staff"}

def get_pos(review):
    if review[0]=="P":
        return review.split("Negative:")[0].split("Positive:")[1]
    else:
        return review.split("Positive:")[1]

def get_neg(review):
    if review[0] == "P":
        return review.split("Negative:")[1]
    else:
        return review.split("Positive:")[0].split("Negative:")[1]

def split_2_neg_pos_columns(df):
    df["Positive"]=df["review"].apply(lambda x:get_pos(x))
    df["Negative"]=df["review"].apply(lambda x:get_neg(x))
    return df


def get_pos_neg_topic_class(labels,topic):
    tmp_list=np.array(labels)[:,topic]
    pos_size=int(len(labels)/2)
    pos_list=list(map(int, np.array(tmp_list)[:pos_size]))
    neg_list=list(map(int, np.array(tmp_list)[pos_size:]))
    return pos_list,neg_list

def finalize(df,labels):
    pos_price,neg_price=get_pos_neg_topic_class(labels, PRICE)
    pos_room,neg_room=get_pos_neg_topic_class(labels,ROOM)
    pos_location,neg_location=get_pos_neg_topic_class(labels,LOCATION)
    pos_food,neg_food=get_pos_neg_topic_class(labels,FOOD)
    pos_facilities,neg_facilities=get_pos_neg_topic_class(labels,FACILITIES)
    pos_staff,neg_staff=get_pos_neg_topic_class(labels,STAFF)

    pos_topics={PRICE:pos_price,ROOM:pos_room,LOCATION:pos_location,FOOD:pos_food,FACILITIES:pos_facilities,STAFF:pos_staff}
    neg_topics={PRICE:neg_price,ROOM:neg_room,LOCATION:neg_location,FOOD:neg_food,FACILITIES:neg_facilities,STAFF:neg_staff}

    for idx in range(6):
        df["topic_"+idx2topic[idx]+"_positive"] = pos_topics[idx]
        df["topic_"+idx2topic[idx]+"_negative"] = neg_topics[idx]
    return df


def remove_special_chars_punc_xtra_spaces(text):
    # convert text to lower case
    text = text.lower()
    # remove punctuation*'
    # text = ''.join([c for c in text if c not in string.punctuation])
    #remove numbers
    text =''.join([x for x in text if not x.isdigit()])
    # remove extra whitespaces and tabs*'
    pattern = r'^\s*|\s\s*'
    text=re.sub(pattern, ' ', text).strip()
    # # define the pattern to keep .,!?/:;
    # pat = r'[^a-zA-z0-9\"\'\s]'
    return text #re.sub(pat, '', text)

def predict(df,first_time,num_labels=6):
    module_path = os.path.dirname(os.path.realpath(__file__))
    if first_time:
        print('Downloading model weights ...')
        gdrive_file_id = '118TMkiqtGGGk8N2hHsUvo90uOKLHSzKL'
        url = f'https://drive.google.com/uc?id={gdrive_file_id}'
        weights_path = os.path.join(module_path, 'fixed_bert_booking_final.pt')
        gdown.download(url, weights_path, quiet=False)
    else:
        weights_path = os.path.join(module_path, 'fixed_bert_booking.pt')
    print('Loading model ...')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type=="cpu":
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(weights_path))
        model.to(device)
    print("Encoding data")
    #prepear data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # tokenizer
    test_comments=list(df.content.values)
    test_encodings = tokenizer.batch_encode_plus(test_comments, max_length=156, pad_to_max_length=True)
    test_input_ids = test_encodings['input_ids']
    test_attention_masks = test_encodings['attention_mask']

    test_inputs = torch.tensor(test_input_ids)
    test_masks = torch.tensor(test_attention_masks)

    params = {'batch_size': 8,'shuffle': False}
    test_data = TensorDataset(test_inputs, test_masks)
    test_dataloader = DataLoader(test_data, **params)

    print("Start predicting")
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    # track variables
    logit_preds, pred_labels, tokenized_texts = [], [], []

    # Predict
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    comment_texts = [tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False) for text in
                     tokenized_texts]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    pred_bools = [pl>0.50 for pl in pred_labels]
    print("Finished predicting")
    return pred_bools

def prepeare_data(read_file_name,first_time, save_file_name, read_type):
    start=time.time()
    if read_type=="csv":
        df = pd.read_csv(read_file_name)
    else:
        df = pd.read_excel(read_file_name)

    df = split_2_neg_pos_columns(df)

    # df=creat_labels(df)
    pos_df=df[["review_id","Positive"]].rename(columns={"Positive": "content"},inplace=False)
    neg_df=df[["review_id","Negative"]].rename(columns={"Negative": "content"},inplace=False)
    text_df=pos_df.append(neg_df, ignore_index=True)

    # text_df= pd.DataFrame(df[["review_id","Positive"]].append(df[["review_id","Negative"]], ignore_index=True),columns=[["review_id","content"]])
    text_df["content"]=text_df["content"].apply(lambda x: remove_special_chars_punc_xtra_spaces(x) )
    labels=predict(text_df,first_time)
    labels_df=finalize(df,labels)

    labels_df.to_csv(save_file_name,index=True)
    print(time.time()-start)

def main(file_path,first_time,save_path=None):
    save_path=save_path if save_path else "labeled_"+file_path
    prepeare_data(file_path,first_time,save_path,"csv")

if __name__ == "__main__":
    file_path="reviews_features_train.csv"#"yourfilepath.csv"
    first_time=False
    main(file_path,first_time)