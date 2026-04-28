import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


class BigramLanguageModel:
    def print_words_info(self):
        print(f"{50 * '-'}")

        print(f"Num of word: {len(self.words)}")
        print(f"Min length: {[min(len(w) for w in self.words)]}")
        print(f"Max length: {[max(len(w) for w in self.words)]}")

        print(f"{50 * '-'}")

    def _open_file(self) -> list[str]:
        with open("name.txt", "r", encoding="utf-8") as file:
            return file.read().splitlines()

    def __init__(self) -> None:
        self.words: list[str] = self._open_file()
        self.chars: list[str] = sorted(set("".join(self.words)))  # words 中所有的字符

        self.stoi: dict[str, int] = {s: i + 1 for i, s in enumerate(self.chars)}
        self.stoi["."] = 0  # 第一个字符是 '.'

        self.itos = {i: s for s, i in self.stoi.items()}

        """ 
       26 个字母 + ., 因而 27 * 27 
       初始值为 0 
       """
        self.N = torch.zeros(27, 27)  # 样本总量
        for word in self.words:
            """ 
           先查word, 找到连续字符, 再查 stoi, 再匹配 N 中坐标, 对应位置 +1 
           """
            word_ = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(word_, word_[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                self.N[ix1, ix2] += 1

    def cal_likelihood_loss_counting_method(
        self, n_example: int, penalty=0
    ):  # 对前 n_example 个 word 计算 loss 的
        sum_likeli: float = 0
        count: int = 0

        """ 
       当 penalty 很大时, 二元字符组原来的频次几乎可以忽略, 如 
       fa = 2+10000, fb = 35 + 10000, ... fn = 109 + 10000 (共 N 个) 
       p(fa) = (2+10000) / sum(2+10000, 3+10000, ...), 其值将接近 1 / N 
       概率分布区趋向平均, 平滑 
       """

        N_with_penalty = self.N
        N_with_penalty += penalty
        probN_count = N_with_penalty / N_with_penalty.sum(
            1, keepdim=True
        )  # 每个元素 / 元素所在行的和

        for word in self.words[:n_example]:
            chs = ["<.>"] + list(word) + ["<.>"]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]

                sum_likeli += math.log(probN_count[ix1, ix2])  # 对数
                count += 1
        """ 
       math.log() 的参数是 x, 返回值是 y 
       sum_likeli = math.log(p1) + math.log(p2) + ... + math.log(pn) = math.log(p1 * p2 * ... * pn) 
       由于 p1, p2 ... 均 < 1, 因而 sum_likeli 是负数, 因而取反作为最终结果 
       如果预期正确的字符 a 的概率是0.8, 那么错误的概率就是0.2 
       p1, p2 ... 越大, 即正确率越高, 那么 -log(p1 * p2 * ... * pn) 就越小, 可以说 loss 越小, 模型质量越高 
       """
        loss = -sum_likeli / count  # 负对数似然
        print(f"loss_counting_method: {loss:.4f}")  # 损失函数

    def cal_likelihood_loss_neural(self, n_example: int):
        xs = []
        ys = []
        for word in self.words[:n_example]:
            word_ = ["<.>"] + list(word) + ["<.>"]
            for ch1, ch2 in zip(word_, word_[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
                # print(ch1, ch2)

        xs = torch.tensor(xs, dtype=torch.long)
        ys = torch.tensor(ys, dtype=torch.long)

        # print(f"xs: {xs}")
        # print(f"ys: {ys}")

        """ 
       只有一个是“热”的（值为 1），其余都是“冷”的（值为 0） 
       假设你有 5 个可能的字符 [a, b, c, d, e]，它们的索引分别是 [0, 1, 2, 3, 4] 
           数字 0 的 One-hot 是：[1, 0, 0, 0, 0] 
           数字 3 的 One-hot 是：[0, 0, 0, 1, 0] 
       """

        xenc = F.one_hot(xs, num_classes=27).float()
        # print(f"xenc shape: {xenc.shape}")
        # print(f"xenc value: {xenc}")

        g = torch.Generator().manual_seed(214783647 + 1)
        """ 
       w 是符合正态分布的采样, 取值范围是(-∞,+∞), 但其表示的是某个二元组的权重 
       """
        w = torch.randn(27, 27, generator=g, requires_grad=True)

        self._fine_tune(xenc, w, n_example, ys)

    def _get_prob_log_softmax(self, xenc: torch.Tensor, w: torch.Tensor):
        """
        这里有点类似过滤, w 本来是 27 * 27
        经过 xenc @ 后, 仅仅把 xenc 那几行抽出来了, 其它行都被过滤掉了
        """

        logits = xenc @ w  # 强制认为它在对数空间
        # print(f"logits shape: {logits.shape}")
        # print(f"logits value: {logits}")
        """ 
       softmax 
       """

        counts = logits.exp()  # > 0, 正值化
        return counts / counts.sum(1, keepdim=True)  # 归一化

    def _fine_tune(
        self, xenc: torch.Tensor, w: torch.Tensor, n_example: int, ys: torch.Tensor
    ):
        n_char = ys.nelement()
        total_loss = None
        for i in range(500):
            probs = self._get_prob_log_softmax(xenc, w)

            """ 
           每一行的概率的平均值 
           probs[torch.arange(n_char), ys] 得到是所有正确答案组成的一个tensor1 
           tensor1.log() 返回 torch.Tensor类型的 [log(tensor1[0]),log(tensor1[1]), ...] : list 
           mean() 求平均, 分子为 log(tensor1[0])+ log(tensor1[1])+... = log(tensor1[0] * tensor1[1] * tensor1[2] * ...) 
           """

            data_loss = -probs[torch.arange(n_char), ys].log().mean()
            reg_loss = 0.01 * (w**2).mean()  # penalty
            total_loss = data_loss + reg_loss
            # print(f"loss_neural: {loss:.4f}")

            w.grad = None  # type of w is troch.tensor

            total_loss.backward()
            """ 
            loss = -probs[torch.arange(n_char), ys].log().mean() 
               这里 ys 的元素并不是所有组合的下标, 所有存在的组合的下标 
               ys 整体是所有正确答案的下标的组合 
               对于越高频的组合, 其会在ys 中出现多次,例如 ys 中会包含 1000 个 an 的下标 
               假设  an, am, ad 的频次比较高,对于 aj, aq 的这种不存在的下标, ys 中不会出现 
               sum(p) = log(p(an)) + ..1000个.. + log(p(an)) + 
                       log(p(am) + ..100个.. + log(p(am)) + 
                       log(p(ad)) + ..20个.. + log(p(ad)) + ... 
               此时 loss = sum(p)  / N 
               如何才能让 loss 减小? 那就是让 p(an) 尽可能的接近1 
               模型为了降低 loss, 会极端使 w 中 an 的权重 W_an 变大, 进而其它的权重W减小(为负值), 本质而言使W 的绝对值变大, 使模型变得激进 
 
               对于出现频次较多的组合, 模型会将其权重W配置的极大,如 an:100, am:80, ad:20 ... 
               而频次较低的组合, 将其权重W配置的极小如 aj:-5, aq: -8, ... 
               直接表现就是它认为所有的a后面一定是n, 不会再预测出其它值, 哪怕他们有一定的可能性 
 
           当加上了 penalty 与 w^ 2相关, 为了减小 loss, 模型同时会趋向于减小 w, 本质而言使模型的绝对值减小 
           模型需要在数据的压力 (Gradient from Data Loss) 和 正则化的压力 (Gradient from Penalty) 之间取得平衡 
           """

            if w.grad is not None:
                w.data += (-50) * w.grad

        if total_loss is not None:
            print(f"loss_neural: {total_loss:.4f}")


b = BigramLanguageModel()


n_example = len(b.words)

# b.cal_likelihood_loss_counting_method(n_example)

# b.cal_likelihood_loss_neural(n_example)


class MLP:
    def __init__(self, block_size=3, embedding_dim=2, n_neural=100) -> None:
        self.bigram_model = BigramLanguageModel()

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.n_neural = n_neural

    def build_data_set(self, start, end):
        X = []
        Y = []
        for word in self.bigram_model.words[start:end]:
            # print(word)
            context: list[int] = [
                0
            ] * self.block_size  # 初始值 [0,0,0 ...],对应着 stoi 中 字符 '.' 的下标
            for ch in word + ".":  # emma.
                ix = b.stoi[ch]
                X.append(context)
                Y.append(ix)

                """ 
               itos[ix] 本质就是 ch 
               这里是为了说明喂给模型的是数字, 而模型通过数字也能突出准确的字符 
               """
                # print(f"{''.join(b.itos[i] for i in context)} -> {b.itos[ix]}")
                context = context[1:] + [ix]
        # X, Y 的行数就表示样本的个数
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        print(f"X shape: {X.shape}")
        # print(f"X shape: {self.X.shape[0]}")
        print(f"Y shape: {Y.shape}")

        return X, Y

    def build_neural(self):
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn(27, self.embedding_dim, generator=g)  # 查找表: LookUp Table
        # emb = C[self.X]  # X.shape[0](行数) * block_size * n_embedding, 嵌入层

        W1 = torch.randn(
            self.block_size * self.embedding_dim, self.n_neural, generator=g
        )  # n_neural 个神经元, 每个有 block_size * n_embedding 个参数(需要接收每个样本的超参数)
        b1 = torch.randn(W1.shape[1], generator=g)  # n_embedding 个 bias

        # hidelayer_dim_1 = self.block_size * self.embedding_dim
        # emb.shape[0] * hidelayer_dim_1,  每行 hidelayer_dim_1 个元素, 每 n_embedding_dim 个对应 1 个字符, 对应着 block_size 个连续的字符(一个样本)
        # h = torch.tanh(emb.view(emb.shape[0], hidelayer_dim_1) @ W1 + b1)  # 隐藏层

        W2 = torch.randn(self.n_neural, 27, generator=g) * 0.01
        b2 = torch.randn(W2.shape[1], generator=g) * 0.0

        self.parameters = {"C": C, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
        for k, v in self.parameters.items():
            print(f"{k} shape: {v.shape}")

        # # print(f"num of parameters: {sum(p.nelement() for p in parameters)}")
        # logits = (
        #     h @ W2 + b2
        # )  # 32 * 27, 每行对应一个样本之后, 27 个字符可能出现的概率(准确的说是权重)

    def learn_rate_test(self, X: torch.Tensor, Y: torch.Tensor):
        for p in self.parameters.values():
            p.requires_grad = True

        learnrates_base = torch.linspace(-3, 0, 1000)  # [-3, 0] # 等差递增
        learn_rates = 10**learnrates_base  # [10^-3, 10^0], 指数递增

        lr_basei = []
        lossi = []
        loss = None
        for i in range(1000):
            """ 
           X.shape[0], X 共多少行, 多少个样本 
           随机数最小值是0, 最大值不超过行数 
           共32个随机数(32 没什么特殊考量, 16, 64, 128 都行) 
           """

            ix = torch.randint(0, X.shape[0], (32,))

            """ 
           全批量（Full Batch）：算出的下降方向最准，但每次计算量太大(几万个样本)，走一步太慢 
           小批量（Mini-batch）：每次只算 32 个样本，计算速度非常快；同时这 32 个样本又能大致代表整体数据的特征，算出来的方向大体上是往山谷（最低点）走的 
           """

            emb = self.parameters[
                "C"
            ][
                X[ix]
            ]  # C[X]:  X.shape[0](行数) * block_size * n_embedding, 共 X.shape[0] 组, 随机取 ix 组

            h = torch.tanh(
                emb.view(-1, self.block_size * self.embedding_dim)
                @ self.parameters["W1"]
                + self.parameters["b1"]
            )
            logits = h @ self.parameters["W2"] + self.parameters["b2"]
            for p in self.parameters.values():
                p.grad = None

            loss = F.cross_entropy(logits, Y[ix])  # 随机取 ix 组
            # print(f"loss: {loss.item():.4f}")
            loss.backward()

            for p in self.parameters.values():
                if p.grad is not None:
                    p.data += -learn_rates[i] * p.grad

            lr_basei.append(learnrates_base[i].item())
            lossi.append(loss.item())
        if loss is not None:
            print(f"loss: {loss.item():.4f}")
        plt.plot(lr_basei, lossi)
        plt.show()

    def train(
        self, X: torch.Tensor, Y: torch.Tensor, batch_size=32, step=1000, learn_rate=0.1
    ):
        for p in self.parameters.values():
            p.requires_grad = True

        for i in range(step):
            """ 
           X.shape[0], X 共多少行, 多少个样本 
           随机数最小值是0, 最大值不超过行数 
           共batch_size个随机数
           """

            ix = torch.randint(0, X.shape[0], (batch_size,))

            """ 
           全批量（Full Batch）：算出的下降方向最准，但每次计算量太大(几万个样本)，走一步太慢 
           小批量（Mini-batch）：每次只算 batch_size 个样本，计算速度非常快；同时这 batch_size 个样本又能大致代表整体数据的特征，算出来的方向大体上是往山谷（最低点）走的 
           """

            emb = self.parameters[
                "C"
            ][
                X[ix]
            ]  # C[X]:  X.shape[0](行数) * block_size * n_embedding, 共 X.shape[0] 组, 随机取 ix 组

            h = torch.tanh(
                emb.view(-1, self.block_size * self.embedding_dim)
                @ self.parameters["W1"]
                + self.parameters["b1"]
            )
            logits = h @ self.parameters["W2"] + self.parameters["b2"]
            for p in self.parameters.values():
                p.grad = None

            loss = F.cross_entropy(logits, Y[ix])  # 随机取 ix 组
            # print(f"loss: {loss.item():.4f}")
            loss.backward()

            for p in self.parameters.values():
                if p.grad is not None:
                    p.data += -learn_rate * p.grad

        # print(f"train loss: {loss.item():.4f}")

    def validate(self, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            emb = self.parameters["C"][X]
            h = torch.tanh(
                emb.view(-1, self.block_size * self.embedding_dim)
                @ self.parameters["W1"]
                + self.parameters["b1"]
            )
            logits = h @ self.parameters["W2"] + self.parameters["b2"]
            loss = F.cross_entropy(logits, Y)

            return loss

    def sample(self):
        with torch.no_grad():
            for i in range(20):
                out = []
                context = [0] * self.block_size
                C = self.parameters["C"]

                while True:
                    emb = C[torch.tensor(context)]

                    h = torch.tanh(
                        emb.view(1, -1) @ self.parameters["W1"] + self.parameters["b1"]
                    )

                    logits = h @ self.parameters["W2"] + self.parameters["b2"]
                    prob = torch.softmax(logits, dim=1)

                    ix = torch.multinomial(prob, num_samples=1, replacement=True).item()

                    if ix == 0:
                        break

                    context = context[1:] + [ix]
                    out.append(ix)

                print("".join(self.bigram_model.itos[i] for i in out))


mlp = MLP()
mlp.n_neural = 200
mlp.embedding_dim = 10

n_words = len(mlp.bigram_model.words)
# n_words = 5
n_train = int(0.8 * n_words)
n_dev = int(0.9 * n_words)


x_train, y_train = mlp.build_data_set(0, n_train)

mlp.build_neural()

random.seed(42)
# 把单词列表彻底打乱！
random.shuffle(mlp.bigram_model.words)

mlp.learn_rate_test(x_train, y_train)
# mlp.train(x_train, y_train, batch_size=128, step=10000)
# mlp.train(x_train, y_train, batch_size=128, step=10000, learn_rate=0.05)
# mlp.train(x_train, y_train, batch_size=128, step=10000, learn_rate=0.01)

# x_dev, y_dev = mlp.build_data_set(n_train, n_dev)
# validate_train_loss = mlp.validate(x_train, y_train)
# validate_dev_loss = mlp.validate(x_dev, y_dev)

# print(f"validate_train_loss: {validate_train_loss}")
# print(f"validate_dev_loss: {validate_dev_loss}")

# mlp.sample()


def draw_C():
    plt.figure(figsize=(8, 8))
    C = mlp.parameters["C"]
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(
            C[i, 0].item(),
            C[i, 1].item(),
            mlp.bigram_model.itos[i],
            ha="center",
            va="center",
            color="white",
        )
    plt.grid(which="minor")
    plt.show()
