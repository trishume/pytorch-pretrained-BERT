#!/usr/bin/env python3
import torch

import argparse
import logging
from tqdm import trange

import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from timeit import default_timer as timer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONTEXT = """
model.to(device)
model.eval()

if args.length == -1:
    args.length = model.config.n_ctx // 2
elif args.length > model.config.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

context_tokens = []
raw_text = CONTEXT.strip()
context_tokens = enc.encode(raw_text)
generated = 0

parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nsamples", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--length", type=int, default=-1)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
"""

CONTEXT2 = """
parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint'
"""

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
    print(context.size())
    prev = context
    output = context.repeat(batch_size, 1)
    past = None
    start = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            past = past[:,:,:,:,1:,:]
            logits = logits[:, -1, :] / temperature
            # print(len(past), past[0].size(), logits.size())

            if i == 0:
                past = [p.repeat(1, batch_size, 1, 1, 1) for p in past]
                past = torch.stack(past)
                logits = logits.repeat(batch_size, 1)

            # print(len(past), past[0].size(), logits.size(), "after")

            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
            output = torch.cat((output, prev), dim=1)
            if i == 0:
                start = timer()
    if start: print("after context: ", timer()-start)
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    context_tokens = []
    raw_text = CONTEXT.strip()
    context_tokens = enc.encode(raw_text)
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        start = timer()
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens,
            start_token=None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        end = timer()
        print("time: ", end - start)
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
    print("=" * 80)

if __name__ == '__main__':
    run_model()


