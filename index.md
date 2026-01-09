docs/index.md
---
title: How I Got 1.91× Faster Sampling in llama.cpp
---

# How I Got 1.91× Faster Sampling in llama.cpp

I found a small performance issue in llama.cpp that sits directly on the critical path of generation: **token sampling**.

It was a redundant O(vocab) heap allocation on every call

Pull request:
`https://github.com/ggml-org/llama.cpp/pull/18365`

---

## The Problem

While studying `llama_sampler_sample()`, I noticed a TODO:

> “do not allocate each time”

The code showed why: each generated token rebuilds a `std::vector<llama_token_data>` of size `n_vocab` shown here:

```cpp // llama-sampling.cpp (Before) llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) { // ... std::vector<llama_token_data> cur; cur.reserve(n_vocab); // Redundant O(vocab) allocation // ... }
```

That means that for a ~150k vocab model, we allocate and fill a ~150k-element vector **once per token generated**. If you generate hundreds or thousands of tokens, this becomes a steady stream of heap traffic in one of the hottest loops in the system.

Sampling is inherently O(vocab) since you must process every logit. repeatedly allocating/freeing that buffer is pure waste: it bogs down the OS with  memory requests, causes fragmentation, and ruins cache behavior. 

I found this by reading the sampling pipeline end-to-end. The model produces logits, and sampling wraps them into a token array and runs a configured sampler chain (temperature/top-k/top-p/penalties/etc.). The key detail: the samplers only need a *contiguous array view* during `apply()`, they don’t need a freshly allocated vector each step.

---

## The Solution
The sampler used by the CLI is typically a **sampler chain**: a pipeline of sampling stages applied in order on each token. GDB confirms the chain uses a vtable to dispatch the `apply` operation:

```gdb (gdb) print *smpl->iface 
$4 = { 
    ... apply = 0x7ffff77de758 <llama_sampler_chain_apply(llama_sampler*, llama_token_data_array*)>,
     ... 
}

The chain simply iterates through the sampling stages:

```bash
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    for (auto * smpl : chain->samplers) {
        llama_sampler_apply(smpl, cur_p);
    }
}
``` 

Because the chain has a stable lifetime for the entire generation session, I moved the buffer ownership to that stable object. The samplers get their contiguous view via the same reused pointer, avoiding the collocation “tax.”

### What I changed

I added a reusable buffer to the chain state:

```cpp
// inside struct llama_sampler_chain
std::vector<llama_token_data> cur;
```
Then, I updated the sampling call to use the existing buffer from the chain instead of allocating a local one:

```cpp
// llama-sampling.cpp (After)
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    
    // Reuse the persistent buffer from the chain
    chain->cur.resize(n_vocab); 
    llama_token_data_array cur_p = { chain->cur.data(), chain->cur.size(), false };
    
    // ... logic continues using the stable pointer
}
```

Now, the samplers get their contiguous array view via the same reused pointer, avoiding the allocation “tax”.


## Performance Results
To isolate the impact, I ran a micro-benchmark across common vocabulary sizes. This measures the total time for 10,000 sampling iterations.


  | Vocab Size | Old (alloc/call) | New (reuse buf) | Speedup |
  |--------|-----------|---------|---------|
  | 32,000 | 426.97 ms | 193.59 ms | 2.21x |
  | 65,536 | 826.57 ms | 386.59 ms | 2.141x |
  | 128,000 | 1671.90 ms | 825.43 ms | 2.03x |
  | 152,064 | 1978.95 ms | 1039.35 ms | 1.90x |

## Interpretation
The results show a consistent ~2x speedup in the sampling logic. While the absolute time saved per token is in microseconds (e.g., ~94μs for Qwen 2.5), it eliminates a linear overhead that scales with vocabulary size.

Optimization is often just about not doing unnecessary work.


