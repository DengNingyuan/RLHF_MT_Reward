#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated, Union
import json 
import pandas as pd
import typer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import os
import inspect
import sys
import torch
from itertools import chain
import json
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
from chatglm3_local.modeling_chatglm import ChatGLMForConditionalGeneration
from utils_chatglm3 import generate_inputs
import random
from collections import defaultdict
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()



def _resolve_path(path):
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(model_dir):
    model_dir = _resolve_path(model_dir)
    if (model_dir/'adapter_config.json').exists():
        print('yes_adapter')
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer
model_dir='/home/adminsh/liushuo/rlhf_ny/ChatGLM3/finetune_demo/output_lora_all_groupby/checkpoint-13822'
model,tokenizer=load_model_and_tokenizer(model_dir)
#model,tokenizer=load_model_and_tokenizer("/home/adminsh/liushuo/rlhf_ny/ChatGLM-RLHF/output_chatglm3_finetunelora13822nocontent_rlhf1_8wzhbertcomet/checkpoint-2000")

def main():

    data_output=pd.DataFrame(  columns=['input','fineresult','reference'])
    testdata_dir='/home/adminsh/liushuo/rlhf_ny/data/groupby/10.nocontentnokeyword.test.json'
    dataset = json.loads(Path(testdata_dir).read_text(encoding="utf8"))

    prompt_t3 = " I hope you can help me translate the following English patent paragraph into Chinese: \n "

    #for i,dict_input_i in tqdm(enumerate(list_input),total=len(list_input)):
    for i,dict_input_i in tqdm(enumerate(dataset),total=len(dataset)):

        content = dict_input_i[0]['问']
        content=content.replace('{','')
        content=content.replace('}','')
   
        response, _ = model.chat(tokenizer, content,temperature=0.2,max_length=512)
        response=response.replace('以下是我为您翻译的专利段落：','')
        response=response.replace('\n','')
        data_output.loc[i,'input']=content.split(prompt_t3)[-1].replace('{','').replace('}','')
        data_output.loc[i,'fineresult']=response

        reference_i=dict_input_i[0]['好答'][0]

        data_output.loc[i,'reference']=reference_i
       
        
        if i%50==0:
            data_output.to_csv('output_result/chatglm3_lora_test0.2tem.csv',index=False)
    data_output.to_csv('output_result/chatglm3_lora_test0.2tem.csv',index=False)
    print(response)




if __name__ == '__main__':
    main()
