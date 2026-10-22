# 🌍🧠 World Models from Scratch

![Course](https://img.shields.io/badge/type-course-8A2BE2.svg)
![Topic](https://img.shields.io/badge/topic-World%20Models-blue.svg)
![Focus](https://img.shields.io/badge/focus-JEPA-0467DF.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="assets/Logo_World_Model_from_scratch.png" alt="World Models from Scratch" width="60%">
</p>

## 📝 Description

A short course on **world models**, the networks that learn how an environment evolves so a machine can *imagine* what happens next. Each chapter is built around one reference figure from **Meta FAIR**, and carries one idea.

---

## 📚 Course outline

| # | Chapter | The idea |
|---|---|---|
| 1 | [What is a world model?](#-1-what-is-a-world-model) | Learn the dynamics, not a reflex |
| 2 | [The vocabulary in one figure](#-2-the-vocabulary-in-one-figure) | Encoder, predictor, representation, cost |
| 3 | [Where do you predict?](#-3-where-do-you-predict) | Pixels, latents, tokens or representations |
| 4 | [Inside the training loop](#-4-inside-the-training-loop) | Multi-step prediction |
| 5 | [The collapse problem](#-5-the-collapse-problem) | Why a naive JEPA learns nothing |
| 6 | [Why rollouts drift](#-6-why-rollouts-drift) | Error accumulation |
| 7 | [Planning with a world model](#-7-planning-with-a-world-model) | The action is the output of a search |
| 8 | [Surprise as a signal](#-8-surprise-as-a-signal) | Energy as the machine looking time |
| 9 | [Everything in one diagram](#-9-everything-in-one-diagram) | A video JEPA, end to end |

---

## 🌍 1. What is a world model?

> **s<sub>t+1</sub> = f(s<sub>t</sub>, a<sub>t</sub>)**

A model-free agent (DQN, PPO) learns a reflex, from what it sees to what it does. A world model learns `f`, how the world moves on, and that unlocks **planning**, **training in imagination** and **detecting the impossible**.

<details>
<summary>🧠 <b>What <code>f</code> makes possible</b></summary>

- **Plan** : roll out several futures and keep the best one.
- **Train without acting** : a policy learns inside the model's imagination (this is why DreamerV3 needs so little real experience).
- **Detect the impossible** : a future the model cannot predict breaks a rule it has learned.

</details>

---

## 🧩 2. The vocabulary in one figure

<p align="center">
  <img src="assets/fair_figures/ebjepa_archi_schema.png" alt="The four JEPA configurations" width="95%">
  <br><sub>Figure from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

Every figure of this course reuses these shapes. Panel **(b)** is the world model in its minimal form.

| Shape | Symbol | Meaning |
|---|---|---|
| 🔵 Dark blue trapezoid | **f<sub>θ</sub>** | The **encoder**, from raw input to representation |
| 🔷 Light blue half circle | **g<sub>φ</sub>** | The **predictor**, moves a representation forward in time |
| ⬛ Grey square | **z** | A **representation**, never an image |
| 🟧 Orange square | **C** | The **cost** (or energy), how wrong the prediction was |
| 🟠 Orange circle | **a** | A variable being **optimised** rather than given |

<details>
<summary>🔌 <b>The four panels, one by one</b></summary>

- **(a) Image** : two views of one image, one encoder, a cost between them. No predictor yet, this is plain joint embedding learning.
- **(b) Video** : the predictor appears. It turns `z_t` into `ẑ_{t+1}`, compared to the true `z_{t+1}`.
- **(c) Action conditioned video** : the predictor also receives `a_t`. The model answers *"what happens if I do this?"*.
- **(d) Planning** : the actions turn orange. The model is no longer trained, it is **searched**.

</details>

---

## 🎨 3. Where do you predict?

| Family | The model predicts | Representative models | Pays for |
|---|---|---|---|
| **Pixel generative** | The image itself | World Models (2018), DIAMOND, Genie | Every pixel, even the unpredictable ones |
| **Latent generative** | A code decoded back to pixels | PlaNet, Dreamer V1 to V3 | Fewer pixels, but still pixels |
| **Token generative** | Discrete tokens of the frame | IRIS, STORM | Quantisation loss, long sequences |
| **Joint embedding (JEPA)** | A **representation**, never decoded | I-JEPA, V-JEPA 2, DINO-WM | Risk of collapse |

The exact future of a video is not determined (think of leaves in the wind), and a pixel loss turns that uncertainty into blur. Each family answers this problem by changing **what is predicted**.

<details>
<summary>📜 <b>The four families in detail</b></summary>

- **Pixel generative** : World Models (Ha & Schmidhuber, 2018) chains a VAE, an MDN-RNN and a tiny controller, and trains an agent inside its own dream. DIAMOND (2024) predicts frames with a diffusion model. Genie (2024) even infers the actions on its own.
- **Latent generative** : the RSSM of PlaNet and Dreamer splits the state into a deterministic part and a stochastic part. DreamerV3 (2023) collects a diamond in Minecraft from scratch, with a policy trained 100 % in imagination.
- **Token generative** : IRIS (2023) quantises frames with a VQ-VAE, then a GPT style Transformer predicts the tokens.
- **JEPA** : no decoder at all. The model may ignore what it cannot predict, so what survives in the representation is what is predictable.

</details>

---

## 🔬 4. Inside the training loop

<p align="center">
  <img src="assets/fair_figures/ebjepa_video_jepa_arch.png" alt="Video JEPA training loop" width="95%">
  <br><sub>Figure from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

Panel (b) unrolled over three frames of Moving MNIST. Training is **multi-step**, the predictor is also trained on its own chained predictions.

<details>
<summary>🔍 <b>Reading the figure</b></summary>

- **The `R` boxes** are not a cost against anything. They are the regulariser that prevents collapse (chapter 5).
- **The three `L_2` costs** : `ŝ_1` against `s_1`, `ŝ_2` against `s_2`, and `ŝ_2` predicted from `s_0` through two predictor steps against the same `s_2`.
- **The projection** : costs are computed after an MLP projector, another anti-collapse trick borrowed from SimSiam and BYOL.

</details>

---

## 💥 5. The collapse problem

<p align="center">
  <img src="assets/fair_figures/ebjepa_image_jepa_arch.png" alt="Image JEPA with its regularisers" width="90%">
  <br><sub>Figure from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

Trained on `‖ŝ − s‖²` alone, a JEPA maps every input to the same vector and reaches zero loss while learning nothing. The `R` boxes exist to forbid this.

<details>
<summary>🛡️ <b>The three anti-collapse guards</b></summary>

- **Asymmetry** : the target branch is an EMA copy of the encoder with a stop-gradient, so it cannot move to meet the prediction.
- **VICReg** : a variance term keeps each dimension alive, a covariance term stops dimensions from copying each other.
- **Energy view** : the loss only pushes energy down on real pairs. A proper energy-based model also needs to push it up elsewhere.

To detect collapse, track the **standard deviation of the representations across the batch**. When it heads to zero, the model is collapsing, whatever the loss says.

</details>

---

## 📉 6. Why rollouts drift

<p align="center">
  <img src="assets/fair_figures/ebjepa_video_jepa_viz.png" alt="Rollout drift on Moving MNIST" width="95%">
  <br><sub>Figure from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

Row 2 predicts one step at a time and stays sharp. Row 3 feeds its own predictions back in, and the digits smear, this is **error accumulation**.

<details>
<summary>📝 <b>Notes</b></summary>

- The model has **no decoder**. One was trained afterwards only to make the representations visible, so the blur is information lost in the representation itself.
- Error accumulation caps how far ahead a world model can be trusted. In my [Snake World Model](https://github.com/Thibault-GAREL/AI_snake_world_model_version), planning quality peaked at a horizon of **5 steps**.

</details>

---

## 🎮 7. Planning with a world model

<p align="center">
  <img src="assets/fair_figures/ebjepa_train_plan_schema.png" alt="Action conditioned training and planning" width="80%">
  <br><sub>Figure from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

At plan time nothing is trained. The actions become the unknowns, and the model searches for the sequence whose imagined end state is closest to the goal.

<p align="center">
  <img src="assets/fair_figures/ebjepa_two_rooms_env.png" alt="Two Rooms environment" width="240">
  <img src="assets/fair_figures/ebjepa_planning_success.gif" alt="Agent reaching the goal by planning" width="240">
  <br><sub>The Two Rooms environment and an agent reaching the goal by planning only. Figures from <a href="https://github.com/facebookresearch/eb_jepa">facebookresearch/eb_jepa</a> (Meta FAIR, Apache 2.0)</sub>
</p>

The same recipe scales to a real robot arm, and the distance to the goal in representation space really goes down as the episode runs.

<p align="center">
  <img src="assets/fair_figures/wms_train_plan_schema.png" alt="JEPA world model training and planning on a real robot" width="95%">
  <img src="assets/fair_figures/wms_repdist_vis.png" alt="Representation distance to goal decreasing" width="45%">
  <br><sub>Figures from <a href="https://github.com/facebookresearch/jepa-wms">facebookresearch/jepa-wms</a> (Meta FAIR, CC BY-NC 4.0)</sub>
</p>

---

## 😲 8. Surprise as a signal

<p align="center">
  <img src="assets/2-VoE_pipeline.svg" alt="Violation of expectation pipeline" width="90%">
</p>

Planning **minimises** the energy. Here the model only watches, and the moments where the energy **spikes** are the moments where a learned law was broken, exactly like the looking time used to test infants.

<details>
<summary>🧪 <b>IntPhys 2, the benchmark built on this protocol</b></summary>

[IntPhys 2](https://github.com/facebookresearch/IntPhys2) shows quadruplets of videos, two possible and two impossible, and tests four principles infants acquire in their first year.

| Principle | Broken when |
|---|---|
| 🧱 **Permanence** | An object stops existing when hidden |
| 🎨 **Immutability** | An object changes shape or colour |
| 🌊 **Continuity** | An object teleports instead of moving |
| 🚧 **Solidity** | Two solid objects pass through each other |

V-JEPA reached **98 % on IntPhys v1** with no task specific training. On IntPhys 2, current video models and VLMs are still around **50 %**, pure chance.

</details>

---

## 🧭 9. Everything in one diagram

<p align="center">
  <img src="assets/1-JEPA_architecture.svg" alt="JEPA world model architecture" width="90%">
</p>

A video JEPA used as a surprise detector, with every piece of the course in place : encoder, predictor, EMA target, regulariser and energy.

---

## 📖 Inspiration / Sources

The figures in `assets/fair_figures/` are **not mine**. They come from Meta FAIR and are reproduced for teaching, with attribution, under their own licences (details in [`ATTRIBUTION.md`](assets/fair_figures/ATTRIBUTION.md)).
- 💻 [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa), the energy-based JEPA tutorial (Apache 2.0)
- 💻 [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms), JEPA world models for planning (CC BY-NC 4.0)

<details>
<summary>📄 <b>Papers</b></summary>

- 📄 [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018)
- 📄 [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3, 2023)
- 📄 [Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588) (IRIS, 2023)
- 📄 [Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399) (DIAMOND, 2024)
- 📄 [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) (LeCun, 2022)
- 📄 [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831) (Garrido, Ballas & LeCun, 2025)
- 📄 [IntPhys 2: Benchmarking Intuitive Physics Understanding in Complex Synthetic Environments](https://arxiv.org/abs/2506.09849) (2025)

</details>

Related work of mine :
- 🐍 [Snake World Model](https://github.com/Thibault-GAREL/AI_snake_world_model_version), a JEPA that plays Snake by planning only
- 🤖 [Language Models from Scratch](https://github.com/Thibault-GAREL/Language_Models), the same "build it and explain it" approach for Transformers

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
