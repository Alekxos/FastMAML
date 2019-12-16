# FastMAML
Lightweight but powerful PyTorch 1.3 implementation of MAML (model-agnostic meta learning) from the 2017 paper by Finn et al. Inspired in part by the original implementation in TensorFlow 1.x and by Kate Rakelly's PyTorch 0.x implementation.

In comparison to the original MAML paper, the iterations in this approach take significantly longer to run. However, even with the same meta-parameters like meta-batch size and learning rate, it converges to higher accuracies in a much smaller number of iterations. I'm not sure where the tradeoff comes from, but both effects considered it tends to converge to higher accuracies much more quickly in absolute terms than other implementations of MAML. That's where the name comes from. :)
