import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

def to_device(obj):
    device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

    if isinstance(obj, torch.Tensor):
        return obj.to(device) 
    elif isinstance(obj, np.ndarray):
        return torch.Tensor(obj).to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v) for k, v in obj.items()} 
    elif isinstance(obj, list):
        return [to_device(v) for v in obj]
    return obj


def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    max_x = torch.max(x, dim=-1, keepdim=True).values
    e_x = torch.exp(x - max_x)
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)


def layer_norm(x, g_b, eps: float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    g = to_device(g)
    b = to_device(b)

    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)

    
    return g * (x - mean) / torch.sqrt(var + eps) + b


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    return (x @ w) + b


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    x = linear(x, w_b1)

    x = gelu(x)

    return linear(x, w_b2)


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    scores = torch.matmul(q, k.T) / math.sqrt(k.size(-1))
    scores = scores + mask
    return torch.matmul(softmax(scores), v)

def mha(x, attn, past_key_value, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']

    if len(past_key_value) == 0:
        # qkv projection
        x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

        qkv = x.chunk(3, dim=-1)  # need to modify

        past_key_value += [qkv[1], qkv[2]]
    
    else:

        qkv_w = c_attn['w'].chunk(3, dim=-1)
        qkv_b = c_attn['b'].chunk(3, dim=-1)
        q_wb = {'w':qkv_w[0], 'b':qkv_b[0]}
        k_wb = {'w':qkv_w[1], 'b':qkv_b[1]}
        v_wb = {'w':qkv_w[2], 'b':qkv_b[2]}

        q = linear(x, q_wb)
        past_key_value[0] = torch.cat((past_key_value[0], linear(x[-1:, :], k_wb)), dim=-2)
        past_key_value[1] = torch.cat((past_key_value[1], linear(x[-1:, :], v_wb)), dim=-2)

        qkv = [q, past_key_value[0], past_key_value[1]]

    #print(len(past_key_value[1]))

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in
                 qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    causal_mask = torch.triu(torch.full((x.size(0), x.size(0)), float('-inf')), diagonal=1) # need to modify
    causal_mask = to_device(causal_mask)

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]

    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, -1)

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, block, past_key_value, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, past_key_value, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, past_key_values, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    i = 0
    x = torch.Tensor(x)
    x = to_device(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, past_key_values[i], n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
        i+=1

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return (x @ wte.T).to("cpu")  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    past_key_values = []

    for _ in params['blocks']:
        past_key_values += [[]]

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, past_key_values, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids





def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    params = to_device(params)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
