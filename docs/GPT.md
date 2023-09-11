# GPT（Generative Pre-Training，生成式预训练）

> [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT1 是一种基于 Transformer 架构的预训练语言模型，具有多层模型、生成式预训练技术和独特的解码技术等特点

介绍了一种名为 GPT 的新型语言模型，该模型通过在大规模语料库上进行训练，能够学习自然语言的模式和规律，从而实现更好的语言理解

GPT 模型是一种基于神经网络的自回归语言模型，使用了 Transformer 的解码器部分

为了预训练GPT模型，研究团队使用了两个大规模的语料库：BooksCorpus 和英文维基百科

GPT 的主要技术特点

- 基于 **Transformer 架构**，其中包括**多头自注意力机制**和**前向神经网络**，使得 GPT1 可以在处理自然语言时捕捉**长距离依赖性**，并且具有高效的并行性
- 预训练技术：使用了一种称为 GPT 的技术，预训练分为两个阶段：预训练和微调（fine-tuning）。
  - 预训练阶段：使用了大量的无标注文本数据集，例如维基百科和网页文本等。通过最大化预训练数据集上的 log-likelihood 来训练模型参数
  - 微调阶段：将预训练模型的参数用于特定的自然语言处理任务，如文本分类和问答系统等
- 多层模型：模型由多个堆叠的 Transformer 编码器组成，每个编码器包含多个注意力头和前向神经网络。这使得模型可以从多个抽象层次对文本进行建模，从而更好地捕捉文本的语义信息。



# GPT-2

> [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

主要解决的问题：如何利用大规模未标注的自然语言文本来预训练一个通用的语言模型，从而提高自然语言处理的能力

与 GPT-1 模型的不同点：GPT-2 模型使用了更大的模型规模和更多的数据进行预训练，同时增加了许多新的预训练任务

主要技术特点

- 大规模预训练：GPT-2 使用了一种无监督学习的方法，在大规模文本语料库上进行预训练。在这个阶段，模型从语料库中学习文本序列的统计规律和语义信息
- 非监督多任务学习：GPT-2 通过训练模型来执行多个不同的自然语言处理任务，从而提高模型的鲁棒性和泛化能力
- 使用 Transformer 架构作为模型的基础：使得模型可以自适应地处理长距离依赖关系，从而更好地理解文本的语义
- 无需人工标注数据：GPT-2 在训练过程中可以自动从大规模文本语料库中学习自然语言的规律
- 零样本学习：GPT-2 能够在只看到少量样本的情况下学习和执行新任务



# GPT-3

> [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

主要解决的问题：如何使一个预训练的语言模型具有迁移学习的能力，即在只有少量标注数据的情况下，能够快速适应到新的任务中

GPT-3 采用了基于 Transformer 的架构，与前一代 GPT-2 类似，但是在模型规模、预训练数据量和使用的预训练任务上都有所增加

GPT-3 使用了多个来源的数据，包括互联网上的文本、书籍、新闻和 Wikipedia 等。这些数据经过清洗和处理后，用于预训练和微调

GPT-3 具有零样本学习的能力，即能够在没有任何样本数据的情况下进行学习和预测



# InstructGPT

> [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

提出的背景：使语言模型更大并不意味着它们能够更好地遵循用户的意图，如大型语言模型可以生成不真实、有毒或对用户毫无帮助的输出，与用户意图不一致

主要解决的问题：如何让语言模型能够更好地遵循人类给出的指令，并在实践中实现它们。此类模型可以广泛应用于自然语言生成、对话系统和语言翻译等领域。

在 GPT-3 基础上进一步强化，使用来自人类反馈的强化学习方案 RLHF（reinforcement learning from human feedback），通过对大语言模型进行微调，从而能够在参数减少的情况下，实现优于 GPT-3 的功能。即训练出奖励模型（rewardmodel）去训练学习模型，AI 训练 AI

- 定义指令集合：即人类需要模型生成的语言指令。这些指令通常是任务相关的，例如完成一项任务或回答某个问题
- 生成指令：通过 InstructGPT 生成一个或多个备选指令，每个指令都对应一个相应的生成概率。这些备选指令会显示在屏幕上供人类评估。
- 人类反馈：人类对生成的备选指令进行评估，并提供一个奖励信号，表示该指令与预期指令的匹配程度。奖励信号可以表示为基于 BLEU、ROUGE 等指标的分数。
- 强化学习训练：根据人类反馈，训练模型以优化生成指令的质量。具体来说，使用强化学习算法，将生成的指令和人类反馈作为训练数据，迭代训练模型，以最大化生成指令的奖励信号。

该方法的优点是可以让语言模型更加有针对性地生成文本，以适应特定任务或场景，并且可以根据人类反馈进行动态调整，提高生成文本的质量和多样性。



# GPT-4

> [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)



# GPT 影响

> [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](https://arxiv.org/abs/2303.10130)


