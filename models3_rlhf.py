"""
classes for ChatGLM RLHF
Critic model
Action model is ChatGLM, 所以可省略
Reward model

"""
import torch
from chatglm3_local.modeling_chatglm import ChatGLMModel
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np
from functools import partial
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModel
"""
critic 的词表最好和action模型的词表一样这样便于对每个生成的token进行打分，
不一致的词表会导致打分不对齐，所以选择用一样的模型但是加一下打分的输出
为了减小打分模型的大小，可以把原来的模型的layers缩减层数。
这样直接继承了，原来的token embedding
"""
class Critic(nn.Module):
    def __init__(self, device="cpu_float") -> None:
        super().__init__()
        #model = ChatGLMModel.from_pretrained("/home/adminsh/liushuo/rlhf_ny/model_download/chatglm3_6b", trust_remote_code=True)
        
        #layers_keep = len(model.layers)//9
        #layers_keep = 1
        #model.layers = model.layers[:layers_keep]

        model = ChatGLMModel.from_pretrained("/home/adminsh/liushuo/rlhf_ny/model_download/chatglm3_6b", trust_remote_code=True,num_layers=1)
        # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        if "cuda" in device:
            model = model.half().cuda(device) # half for gpu only
            model.requires_grad_(False)
        elif "cpu" == device:
            model = model.bfloat16()
            model.requires_grad_(False)
        else:
            model = model.float()
            model.requires_grad_(False)
        self.model = model
        self.model.hidden_size=4096
        self.output_linear = nn.Linear(self.model.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = self.model.device
    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)


"""
一样的原因，不需要再把生成的token ids转成文字在再转到目标ids，
所以也用chatglm直接做基模型，
只是这里只取最后的token算出对整句生成的奖励分数，具体取哪个位置可以
后续在代码里面指定，比如用torch.gather
""" 
Reward = Critic


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def jaccard(s1, s2):
    """
    可能有字符串重合但是语义不一致问题，
    TODO 可以用多阶的jaccard来解决
    """
    if isinstance(s2,int):
        return 'skip_data'
    assert len(s1)+len(s2)>0
    s1 = set(s1)
    s2 = set(s2)
    s_or = s1 | s2
    s_and = s1 & s2
    jaccard_distance = len(s_and)/len(s_or)
    return jaccard_distance

# 基于和good_answers和bad_answers比较的Reward模型，具有通用性和易学习，基本上都是基于Bert余弦值
class RewardBySimilarity(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = BertTokenizer.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model = BertModel.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    def forward(self, gen_texts=["你好"],
                 good_answers=['你好', "hello"],
                 bad_answers=['再见', 'bye bye'],
                 weight_for_cos_and_jaccard = [0.5, 0.5]):
        examples = good_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts)>0 and example_num>0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(good_answers):] = -1
        sentences = gen_texts + examples
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt')
        ids = self.tokenizer.batch_encode_plus(sentences, add_special_tokens=False)["input_ids"]
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"]/max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor().long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]
        answers_vecs = sentence_embeddings[len(gen_texts):]
        reward_ = []
        for i in range(gen_text_vecs.shape[0]):
            gen_text_vecs_ = gen_text_vecs[i:i+1]
            # 用一下广播计算cos
            coses = torch.cosine_similarity(gen_text_vecs_, answers_vecs, dim=1)
            # 余弦截断
            coses[(coses<0)] = 0
            # 计算 jaccard距离
            
            jaccard_s1 = partial(jaccard, ids[i])
            jaccards = np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object))

            if 'skip_data' in jaccards:
                return 'skip_data'
            else:
                jaccards = torch.tensor(jaccards ,dtype=coses.dtype, device=coses.device)

            #jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard[0]*coses + weight_for_cos_and_jaccard[1]*jaccards
            value, index = similarity.max(dim=-1)
            reward_.append(value*reward_direction[index])
        reward = torch.stack(reward_)
        return reward

class RewardBySimilarity_zhbertcometjaccard(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = BertTokenizer.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model = BertModel.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        model_path_comet = '/home/adminsh/liushuo/rlhf_ny/model_download/unbabel_wmt22_cometkiwi_da/checkpoints/model.ckpt'
        model_comet = load_from_checkpoint(model_path_comet)
        self.model_comet = model_comet.to(device)
    
    def forward(self, gen_texts=["你好"],
                 good_answers=['你好', "hello"],
                 bad_answers=['再见', 'bye bye'],
                 src_language=['hello'],
                 weight_for_cos_and_jaccard_comet = [0.33, 0.33,33]):
        examples = good_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts)>0 and example_num>0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(good_answers):] = -1
        sentences = gen_texts + examples
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt')
        ids = self.tokenizer.batch_encode_plus(sentences, add_special_tokens=False)["input_ids"]
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"]/max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor().long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]
        answers_vecs = sentence_embeddings[len(gen_texts):]
        reward_ = []

        for i in range(gen_text_vecs.shape[0]):
            gen_text_vecs_ = gen_text_vecs[i:i+1]
            # 用一下广播计算cos
            coses = torch.cosine_similarity(gen_text_vecs_, answers_vecs, dim=1)
            # 余弦截断
            coses[(coses<0)] = 0
            # 计算 jaccard距离
            
            jaccard_s1 = partial(jaccard, ids[i])
            jaccards = np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object))

            if 'skip_data' in jaccards:
                return 'skip_data'
            else:
                jaccards = torch.tensor(jaccards ,dtype=coses.dtype, device=coses.device)

            #jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard_comet[0]*coses + weight_for_cos_and_jaccard_comet[1]*jaccards
            value, index = similarity.max(dim=-1)
            reward_.append(value*reward_direction[index])
        reward_cos= torch.stack(reward_)

        src=src_language[0]  
        reward_ = []
        sim_list=[]
        for gen_text_i in gen_texts:
            
            tem_dict={}
            tem_dict['src']=src
            tem_dict['mt']=gen_text_i
            sim_list.append(tem_dict)

        model_output = self.model_comet.predict(sim_list, batch_size=8, gpus=1)
        reward_=model_output.scores
        
        reward_ = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in reward_]
        reward=torch.stack(reward_)*weight_for_cos_and_jaccard_comet[2]+reward_cos

        return reward


class RewardBySimilarity_zhbertcomet(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = BertTokenizer.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model = BertModel.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        model_path_comet = '/home/adminsh/liushuo/rlhf_ny/model_download/unbabel_wmt22_cometkiwi_da/checkpoints/model.ckpt'
        model_comet = load_from_checkpoint(model_path_comet)
        self.model_comet = model_comet.to(device)
    
    def forward(self, gen_texts=["你好"],
                 good_answers=['你好', "hello"],
                 bad_answers=['再见', 'bye bye'],
                 src_language=['hello'],
                 weight_for_cos_and_jaccard_comet = [0.8,0.2]):
        examples = good_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts)>0 and example_num>0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(good_answers):] = -1
        sentences = gen_texts + examples
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors='pt')
        ids = self.tokenizer.batch_encode_plus(sentences, add_special_tokens=False)["input_ids"]
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"]/max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor().long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]
        answers_vecs = sentence_embeddings[len(gen_texts):]
        reward_ = []

        for i in range(gen_text_vecs.shape[0]):
            gen_text_vecs_ = gen_text_vecs[i:i+1]
            # 用一下广播计算cos
            coses = torch.cosine_similarity(gen_text_vecs_, answers_vecs, dim=1)
            # 余弦截断
            coses[(coses<0)] = 0
            # 计算 jaccard距离
            
            #jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard_comet[0]*coses 
            value, index = similarity.max(dim=-1)
            reward_.append(value*reward_direction[index])
            
        reward_cos= torch.stack(reward_).to(self.model.device)
    

        src=src_language[0]  
        reward_ = []
        sim_list=[]
        for gen_text_i in gen_texts:
            
            tem_dict={}
            tem_dict['src']=src
            tem_dict['mt']=gen_text_i
            sim_list.append(tem_dict)

        model_output = self.model_comet.predict(sim_list, batch_size=8, gpus=1)
        reward_=model_output.scores
        
        reward_ = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in reward_]
        reward_ = torch.stack(reward_).to(reward_cos.device)
        
        reward=reward_*weight_for_cos_and_jaccard_comet[1]+reward_cos
        return reward

class RewardBySimilarity_comet(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        '''
        tokenizer = BertTokenizer.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model = BertModel.from_pretrained('/home/adminsh/liushuo/rlhf_ny/model_download/chinese-macbert-base-similarity')
        model.eval()
        '''

        model_path = '/home/adminsh/liushuo/rlhf_ny/model_download/unbabel_wmt22_cometkiwi_da/checkpoints/model.ckpt'
        model = load_from_checkpoint(model_path)
        self.model = model.to(device)
        self.device = device

    def forward( self, gen_texts=["你好"],
                 good_answers=['你好', "hello"],
                 bad_answers=['再见', 'bye bye'],
                 keyword=['你好'],
                 src_language=['hello'],
                 weight_for_cos_jaccard_keyword = [0.33, 0.33,0.33]):
        
        src=src_language[0]  
        reward_ = []
        sim_list=[]
        for gen_text_i in gen_texts:
            
            tem_dict={}
            tem_dict['src']=src
            tem_dict['mt']=gen_text_i
            sim_list.append(tem_dict)

        model_output = self.model.predict(sim_list, batch_size=8, gpus=1)
        reward_=model_output.scores
        
        reward_ = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in reward_]
        reward=torch.stack(reward_)

        return reward

def test_reward_by_similarity():
    reward_model = RewardBySimilarity()
    reward = reward_model()
    print(reward)

def test_critic():

    tokenizer = AutoTokenizer.from_pretrained("/home/adminsh/liushuo/rlhf_ny/model_download/chatglm3_6b", trust_remote_code=True)
    critic = Critic()
    input_ids = torch.tensor(tokenizer.encode("你好"), dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.repeat(2,1)
    output = critic(input_ids=input_ids)  
    print(output.shape)

def test_reward():
    # with torch.no_grad():
    # input_ids_RM =  sequences.to(RM_device)
    # rewards_ = reward_model(input_ids = input_ids_RM)
    # # 由于只对最后的整句进行reward，所以只有最后一个action后的state有reward
    # rewards = torch.zeros_like( sequences, dtype=rewards_.dtype, device=rewards_.device)
    # pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    # masks = ( sequences!=pad_id).long().to(RM_device)

    # final_position = masks.sum(dim=-1)-1
    # index=final_position.unsqueeze(-1)
    # reward = rewards_.gather(dim=1, index=index)
    # rewards.scatter_(dim=1, index=index, src=reward)
    pass

if __name__ == "__main__":
    test_reward_by_similarity()