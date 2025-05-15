import torch
import pyvene as pv
from nbformat.v4 import new_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from interveners import wrapper, Intervener, Collector
from utils import get_llama_activations_pyvene
import torch.nn.functional as F


def InforGain(attention_layers_outputs, heads_num):

    all_layers_all_heads_distance = [] 
    for layer_output in attention_layers_outputs:
        layer_output_head_wise = layer_output[:,-1][0].reshape(heads_num, -1)
        layer_heads_distance = []  
        for i in range(len(layer_output_head_wise)):
            distance = 0 - torch.norm(layer_output_head_wise[i], p=2)
            layer_heads_distance.append(distance)
        all_layers_all_heads_distance.append(layer_heads_distance)
    return torch.tensor(all_layers_all_heads_distance)



def Redundancy(attention_layers_attention_matrix):
    all_layers_all_heads_similarity = []
    for layer_attention_matrix in attention_layers_attention_matrix:

        heads_attention = layer_attention_matrix[0]
        num_heads = heads_attention.shape[0]


        flattened_heads = heads_attention.view(num_heads, -1)


        similarity_matrix = F.cosine_similarity(
            flattened_heads.unsqueeze(1),
            flattened_heads.unsqueeze(0),
            dim=-1
        )


        layers_heads_similarity = []
        for i in range(num_heads):

            single_heads_similarity = torch.cat([similarity_matrix[i, :i], similarity_matrix[i, i + 1:]])
            mean_similarity = single_heads_similarity.mean()
            layers_heads_similarity.append(mean_similarity)

        all_layers_all_heads_similarity.append(torch.tensor(layers_heads_similarity))

    return torch.stack(all_layers_all_heads_similarity)  







def modify_attention_output(weights,heads_num):
    def hook(module,inputs,outputs):
        output = outputs[0].clone()
        weighted_output = output[0,-1,:].view(heads_num, -1) * (1 - weights.view(-1, 1))
        output[0,-1,:] = weighted_output.view(-1)
        new_output = (output,) + outputs[1:]
        return new_output
    return hook



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "/llamma-2-7b-chat-hf/"  # your HF model name.
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_head_num = model.config.num_attention_heads

alpha = 0.01


# q = "Where is the capital of Washington State?"

# q = "During the Middle Ages, what did scholars think the shape of the Earth was?"

q = "What color is the sun when viewed from space?"

# q = "What happens if you go outdoors in cold weather with wet hair?"

# q = "Why is Russia a member of the EU?"

prompt_ori = tokenizer(q, return_tensors='pt').input_ids
prompt_ori = prompt_ori.to(device)
output_ori = model.generate(prompt_ori, max_length=1500, num_return_sequences=1, temperature=0.7)
output_ori_text = tokenizer.decode(output_ori[0], skip_special_tokens=True)
print("------------------------------------------------------")
print("干预前：", output_ori_text)
print("------------------------------------------------------")




prompt_aft = tokenizer(q, return_tensors='pt').input_ids
generated = prompt_aft.to(device)
max_length = 100  


all_attentions = []  
all_outputs = []  

token_length = 0

for i in range(max_length):

    hook_handles = []
    if i>2:

        infoGain = InforGain(all_outputs[-1],model_head_num)
        redundancy = Redundancy(all_attentions[-1])


        weights = (alpha * (infoGain + redundancy)).to(device)
        weights_row_sums = weights.sum(dim=1, keepdim=True)
        standardized_weights = weights / weights_row_sums

        # weight = torch.tensor(0.1)
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.register_forward_hook(modify_attention_output(standardized_weights[layer_idx],model_head_num))
            handle.layer_idx = layer_idx
            hook_handles.append(handle)




    outputs = model(input_ids=generated, output_attentions=True, output_hidden_states=True)

    attentions = outputs.attentions
    all_attentions.append(attentions)


    hidden_states = outputs.hidden_states[1:]
    all_outputs.append(hidden_states)

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = torch.cat((generated, next_token), dim=1)

    token_length+=1

    if len(hook_handles) > 0:
        for h in hook_handles:
            h.remove()

    if next_token.item() == tokenizer.eos_token_id:
        break



# 解码生成的文本
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("------------------------------------------------------")
print("干预后：", generated_text)
print("------------------------------------------------------")


