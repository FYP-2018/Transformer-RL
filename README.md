# Transformer-RL

Implement Transformer with Mixed Reinforcement Training 

Current Code under RL assumes pre-trained model is provided in ./ folder


## Run
Execute ```main.py``` to run the model


## Reference Paper: 
1. Sequence level training with recurrent neural networks
2. A Deep Reinforced Model for Abstractive Summarization
3. Self-critical Sequence Training for Image Captioning
4. 

## Future Direction: 
After reviewing the _Attention is all you need_ paper, 
1. I suspect that REINFORCE
might not be a suitable training metric for transformer: while the strength of transformer 
is parallelization, REINFORCE totally ruins this advantage and dramatically destroy model's
computational efficiency 
2. Another suspicion on the unsatisfying result that Transformer obtained on CNNDM is that: 
vanilla transformer might be intrinsically more promising on Summarization with shorter length
Here is a quote from _Attention is all you need_ paper (at the bottom of Page 6): 
```
As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed 
operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, 
self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation
dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine
translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for 
tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r 
in the input sequence centered around the respective output position. This would increase the maximum path length to 
O(n/r). We plan to investigate this approach further in future work.
```
so for the next step, I'm planning to:
1. recheck the seq2seq version (someone suggested s2s shuold have promising results on short tasks, so it's possible 
that the previous version have some bugs inside) 
2. investigate the Encoder-only version of transformer, 
3. after step 2: replace the multi-head/original attention into Conv Attention (as suggested in paper _Generating 
Wikipedia by Summarizing Long Sequences _), see if the results can be improved

### P.S, for the atten_vis branch
The code under that branch is for generating simple attention visualization (generate png file with matplotlib). Since the code is totally not cleaned and not updated with the code in /master branch, just leave it there for future reference
