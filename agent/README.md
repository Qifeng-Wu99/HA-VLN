# HA-VLN-CE Agents

Our agents adapt settings and codes from [VLN-CE](https://github.com/jacobkrantz/VLN-CE/). Check the [VLN-CE](VLN-CE) for more details.


To implement the HA-VLN-CMA agent, you can use the following script:

```bash
# Training
python run.py --exp-config config/HAVLNCE_task.yaml --run-type train

# Evaluation
python run.py --exp-config config/HAVLNCE_task.yaml --run-type eval

# Inference
python run.py --exp-config config/HAVLNCE_task.yaml --run-type inference
```

### HA-VLN-VL Agent

#### Model Structure
In order to emphasize the impact of improving visual-language understanding on navigation performance, the model discards the A2C method in favor of simple imitation learning, retrained within the HA-VLN environment. Specifically, HA-VLN-VL adapts its model structure from [Recurrent VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT), with modifications to enhance visual-language understanding for navigation tasks. HA-VLN-VL leverages a BERT-like architecture to increase comprehension of complex instructions and resolve the misalignment between vision cues and navigation instructions (Challenge 2). The model can be defined as follows:

$$
s_t, p_t^a = \text{HA-VLN-VL}(s_{t-1}, X, V_t),
$$

where \( s_t \) is the state representation at time step \( t \), \( p_t^a \) denotes action probabilities, \( s_{t-1} \) is the previous state, \( X \) represents the language tokens of the instruction, and \( V_t \) are the visual tokens for the scene at time \( t \). HA-VLN-VL processes these inputs through a multi-layer Transformer, where the state token performs self-attention with other tokens to update its representation. The action probabilities are computed using the averaged attention weights of the final layer:

$$
p_t^a = \bar{\text{AveragePool}}_{s,v}^l,
$$

where it denotes the mean attention weights of the visual tokens relative to the state at the final layer \( l \).

#### Training Setting
HA-VLN-VL is trained in the HA-VLN environment. We initialize weights from Prevalent to leverage the prior knowledge obtained from large-scale pre-training. We followed the same training recipe as [Recurrent VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT).

---

### HA-VLN-CMA Agent

#### Model Structure
Our HA-VLN-CMA agent employs a cross-modal attention architecture to process visual observations and language instructions jointly. This model integrates three core components. 

First, the visual encoder processes RGB and depth inputs through a ResNet backbone, transforming each observation \( o_t \) at timestep \( t \) into a visual feature representation, denoted as:

$$
v_t = \text{ResNet}(o_t).
$$

Second, a BERT model serves as the language encoder, capturing the semantics of the navigation instructions \( I \) and generating language features:

$$
l = \text{BERT}(I).
$$

Finally, a cross-modal fusion module aligns the visual and language features using a multi-head attention mechanism, yielding the fused features:

$$
f_t = \text{MultiHeadAttention}(v_t, l).
$$

At each timestep \( t \), the model computes an action distribution \( P(a_t | f_t) \) over possible actions \( a_t \), defined by:

$$
P(a_t | f_t) = \text{Softmax}(\text{MLP}_{\text{action}}(f_t)).
$$
