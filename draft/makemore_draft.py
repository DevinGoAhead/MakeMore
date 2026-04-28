import torch
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


""" 
read(), 全部内容读入内存, 返回字符串 
splitlines() 按行分割, 返回 list[str] 
"""

with open("name.txt", "r", encoding="utf-8") as file:
    words: list[str] = file.read().splitlines()


def print_base_info(words):
    print(f"Num of word: {len(words)}")
    print(f"Min length: {[min(len(w) for w in words)]}")
    print(f"Max length: {[max(len(w) for w in words)]}")
    # print(words[:10])

    """ 
    join 的返回值是字符串, 因而 set 是按字符串字符迭代遍历的, 最后得到的是去重的字符 
    """
    # print(set(" ".join(words)))

    print(f"{'=' * 50}")


print_base_info(words)


def print_frequency(words):
    b: dict = {}
    for w in words:
        chs = ["<S>"] + list(w) + ["<E>"]

        # zip 以短的 list 为基准, 对两个list 配对返回 map
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram: tuple[str, str] = (ch1, ch2)

            """  
            a = dict[key1], 仅查询, 不存在则抛异常  
            dict[key2] = b, 查询, 若不存在则则插入, 若存在则覆盖  
            get(key, default), 存在返回value, 不存在返回 default  
            """
            b[bigram] = b.get(bigram, 0) + 1  # 每出现一次就 +1

    print(*b.items(), sep="\n")
    print(sorted(b.items(), key=lambda k: -k[1]))


# print_frequency(words)

print(f"{'=' * 50}")


chars = sorted(set("".join(words)))
# print(chars)


def frequency_image_1():
    stoi = {s: i for i, s in enumerate(chars)}  # [str, int]
    stoi["<S>"] = 26
    stoi["<E>"] = 27

    """ 
    26 个字母 + S + E, 因而 28 * 28 
    初始值为 0 
    """
    N = torch.zeros((28, 28), dtype=torch.int32)

    """ 
    先查word, 找到连续字符, 再查 stoi, 再匹配 N 中坐标, 对应位置 +1 
    """
    for word in words:
        chs = ["<S>"] + list(word) + ["<E>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1  # [ch1-> i, ch2 -> j], 对应坐标表示出现的次数
    # print(N)

    itos = {i: s for s, i in stoi.items()}
    plt.figure(figsize=(32, 32))  # 画布尺寸

    """ 
    将 N 渲染为图像(仅色块) 
    Ni 对应到 y 轴(前面的字母), Nj 对应到 x 轴(后面的字母) 
    就是说在 假设 ab 在 stoi 中对应的下标分别是 i, j, 那么在 N 中 ab 的坐标就是 ij, 但是在 plt 中对应的坐标就是 ji 
    """
    plt.imshow(N, cmap="Blues")
    for i in range(28):
        for j in range(28):
            ch = itos[i] + itos[j]

            """ 
            ha, 水平对齐方式, va, 竖直对齐方式 
            """
            plt.text(
                j, i, ch, ha="center", va="bottom", color="gray", fontsize=18
            )  # 叠加文字

            plt.text(
                j,
                i,
                f"{N[i, j].item()}",
                ha="center",
                va="top",
                color="gray",
                fontsize=18,
            )  # 叠加数字

    plt.axis("off")

    # plt.show()
    plt.savefig("frequency_image_1")


# frequency_image_1()


def get_stoi() -> dict[str, int]:
    stoi = {s: i + 1 for i, s in enumerate(chars)}  # [str, int], 0 留给特殊字符
    stoi["<.>"] = 0
    return stoi


def get_N(stoi: dict) -> torch.Tensor:
    """
    26 个字母 + ., 因而 27 * 27
    初始值为 0
    """
    N = torch.zeros((27, 27), dtype=torch.int32)

    """ 
    先查word, 找到连续字符, 再查 stoi, 找到是第 ix1, ix2 个字母, 再匹配 N 中坐标(ix1, ix2), 对应位置 +1 
    """
    for word in words:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # print(N)
    return N


stoi = get_stoi()
N = get_N(stoi)


def frequency_image_2():
    itos = {i: s for s, i in stoi.items()}
    plt.figure(figsize=(32, 32))  # 画布尺寸

    """ 
    将 N 渲染为图像(仅色块) 
    Ni 对应到 y 轴(前面的字母), Nj 对应到 x 轴(后面的字母) 
    就是说在 假设 ab 在 stoi 中对应的下标分别是 i, j, 那么在 N 中 ab 的坐标就是 ij, 但是在 plt 中对应的坐标就是 ji 
    """
    plt.imshow(N, cmap="Blues")
    for i in range(27):
        for j in range(27):
            ch = itos[i] + itos[j]

            """ 
            ha, 水平对齐方式, va, 竖直对齐方式 
            """
            plt.text(
                j, i, ch, ha="center", va="bottom", color="gray", fontsize=18
            )  # 叠加文字

            plt.text(
                j,
                i,
                f"{N[i, j].item()}",
                ha="center",
                va="top",
                color="gray",
                fontsize=18,
            )  # 叠加数字

    plt.axis("off")
    # plt.show()

    plt.savefig("frequency_image_2")


# frequency_image_2()


def prob():
    stoi = {s: i + 1 for i, s in enumerate(chars)}  # [str, int], 0 留给特殊字符
    stoi["<.>"] = 0

    """ 
    26 个字母 + ., 因而 27 * 27 
    初始值为 0 
    """
    N = torch.zeros((27, 27), dtype=torch.int32)

    """ 
    先查word, 找到连续字符, 再查 stoi, 再匹配 N 中坐标, 对应位置 +1 
    """
    for word in words:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1  # [ch1-> i, ch2 -> j], 对应坐标表示出现的次数
    print(N[0])  # N[0] 的效果就是N[0, :], 第一行

    N[0].float
    print(N[0] / N.sum())  # 概率


# prob()


def torch_prob_test():
    g = torch.Generator().manual_seed(2147483647)
    samps: torch.Tensor = torch.rand(3, generator=g)
    print(samps)
    prob = samps / samps.sum()  # 占比
    print(prob)

    """ 
    samps 的所有元素的和不一定为 1
    samps 元素的值本身表示该元素在 samps 中的占比, 因而采样的概率就是 元素值 / samps
    实际采样得到的是 samps 元素的下标
    input: 权重张量
    num_samples: 抽取多少个样本
    replacement: 抽取完成后是放回
    """
    x = torch.multinomial(samps, num_samples=100, replacement=True, generator=g)
    print(x)


# torch_prob_test()


def torch_prob():
    g = torch.Generator().manual_seed(2147483647)
    itos: dict[int, str] = {i: s for s, i in stoi.items()}
    for i in range(20):
        out = []
        idx = 0
        while True:
            p = N[idx].float()  # 第 idx 行, 一维, 数组
            p = p / p.sum()  # 第 idx 行的频次分布

            """ 
            idx 起始值为0, 表示第 0 行 
                而 <.><.> 出现的次数为0(因为没有空的word), 因而采样概率为0 
                即第0行不可能采样到 [0,0], 即 torch.multinomial 的采样值不可能为0 
                这也是 if idx == 0 则 break 的原因 
 
            假设 torch.multinomial 采样值为 5, 那么 5 对应的字母就是 e, 那么 e 之后哪个字母出现的概率最大呢? 
            那得去 e 所在的行去找, e 所在行数就是第 5 行, 将采样值 5 赋值给 idx 作为下一次采样的行即可 
            以此类推 
            """
            idx = int(
                torch.multinomial(
                    p, num_samples=1, replacement=True, generator=g
                ).item()
            )  # 按照第idx行的分布, 采样一个值

            out.append(itos[idx])

            if idx == 0:
                break

        print("".join(out))
    print(f"{'=' * 50}")


# torch_prob()


def torch_sum_test():

    print(N.shape)

    p1 = N.sum(0)
    print(p1.shape)
    print(p1)  # torch.Size([27])

    p2 = N.sum(0, keepdim=True)
    print(p2.shape)  # torch.Size([1, 27])
    print(p2)

    p3 = N.sum(1, keepdim=True)
    print(p3.shape)  # torch.Size([27, 1])
    print(p3)

    p4 = N.sum(1, keepdim=True)  # 对行求和, 压缩列, 得到 27 * 1 的 Tensor
    p4 = N / p4  # 行概率分布,行归一化
    print(p4.shape)  # torch.Size([27, 27])


# torch_sum_test()

"""
torch_prob 中需要频繁从 N 中取数据求和
这里预先计算出一个概率矩阵, 矩阵中每个元素都是所在行的采样概率
"""


def get_PN():
    # p = N.sum(1, keepdim=True) # 对行求和, 压缩列, 得到 27 * 1 的 Tensor
    return N / N.sum(1, keepdim=True)  # 行概率分布,行归一化, torch.Size([27, 27])


p_N = get_PN()


def torch_prob_with_torch_sum():
    g = torch.Generator().manual_seed(2147483647)
    itos: dict[int, str] = {i: s for s, i in stoi.items()}

    for i in range(20):
        out = []
        idx = 0
        while True:
            # p = N[idx].float()  # 第 idx 行, 一维, 数组
            # p = p / p.sum()  # 第 idx 行的频次分布

            """ 
            idx 起始值为0, 表示第 0 行 
                而 <.><.> 出现的次数为0(因为没有空的word), 因而采样概率为0 
                即第0行不可能采样到 [0,0], 即 torch.multinomial 的采样值不可能为0 
                这也是 if idx == 0 则 break 的原因 
 
            假设 torch.multinomial 采样值为 5, 那么 5 对应的字母就是 e, 那么 e 之后哪个字母出现的概率最大呢? 
            那得去 e 所在的行去找, e 所在行数就是第 5 行, 将采样值 5 赋值给 idx 作为下一次采样的行即可 
            以此类推 
            """
            idx = int(
                torch.multinomial(
                    p_N[idx], num_samples=1, replacement=True, generator=g
                ).item()
            )  # 按照第idx行的分布, 采样一个值

            out.append(itos[idx])

            if idx == 0:
                break

        print("".join(out))
    print(f"{'=' * 50}")


# torch_prob_with_torch_sum()


"""
每种情况的概率都很小, < 0.5, 还有许多 0.04 左右的, 着说明模型几乎无法准确判断出字符组的概率
一个好的模型应该能明确最大概率的情况, 且其概率接近 1,其他情况的概率将会极小
使用
"""


def prob_N(stoi: dict):
    for word in words[:3]:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            print(f"{(ch1, ch2)} -> {p_N[ix1, ix2]:.4f}")  # 字符组合及其概率


# prob_N(stoi)


def likelihood():
    sum_likeli: float = 0
    n: int = 0
    for word in words[:3]:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            sum_likeli += math.log(p_N[ix1, ix2])
            n += 1
    """
    math.log() 的参数是 x, 返回值是 y
    sum_likeli = math.log(p1 * p2 * ... * pn) = math.log(p1) + math.log(p2) + ... + math.log(pn)
    由于 p1, p2 ... 均 < 1, 因而 sum_likeli 是负数, 因而取反作为最终结果
    """
    print(-sum_likeli / n)  # 损失函数


# likelihood()


def neural() -> torch.Tensor:
    xs = []
    ys = []
    for word in words[:1]:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
            # print(ch1, ch2)

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    print(f"xs: {xs}")
    print(f"ys: {ys}")

    """
    只有一个是“热”的（值为 1），其余都是“冷”的（值为 0）
    假设你有 5 个可能的字符 [a, b, c, d, e]，它们的索引分别是 [0, 1, 2, 3, 4]
        数字 0 的 One-hot 是：[1, 0, 0, 0, 0]
        数字 3 的 One-hot 是：[0, 0, 0, 1, 0]
    """

    xenc = F.one_hot(xs, num_classes=27).float()
    print(f"xenc shape: {xenc.shape}")
    print(f"xenc value: {xenc}")

    def get_prob():
        g = torch.Generator().manual_seed(214783647 + 1)
        w = torch.randn(27, 27, generator=g)
        """
        这里有点类似过滤, w 本来是 27 * 27
        经过 xenc @ 后, 仅仅把 xenc 那几行抽出来了, 其它行都被过滤掉了
        """

        logits = xenc @ w
        counts = logits.exp()
        return counts / counts.sum(1, keepdim=True)

    return get_prob()


prob_nnn = neural()


def prob_test():
    xs = []
    ys = []
    for word in words[:1]:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
            # print(ch1, ch2)

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    nlls = torch.zeros(5)

    for i in range(5):
        x = int(xs[i].item())
        y = int(ys[i].item())

        itos = {i: s for s, i in stoi.items()}

        print(f"{50 * '-'}")

        print(f"bigram example {i + 1}: {itos[x], itos[y]} | index: ({x}, {y})")
        print(f"input to the neural net: {x}")
        print(f"output probabilities from the neural net: {prob_nnn[i]}")
        print(f"label next character: {y}")
        p = prob_nnn[i, y]
        print(
            f"probability assigned by the net to the the correct character: {p.item():.4f}"
        )
        logp = -torch.log(p)
        print(f"log likelihood(-logp): {logp.item():.4f}")

        nlls[i] = logp

    print(f"{50 * '='}")
    print(
        f"average negative log likelihood, i.e. loss = {nlls.mean().item():.4f}"
    )  # 似然数平均值


# prob_test()


def fine_tune():
    xs = []
    ys = []
    for word in words[:1]:
        chs = ["<.>"] + list(word) + ["<.>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
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
    w = torch.randn(27, 27, generator=g, requires_grad=True)
    """
    这里有点类似过滤, w 本来是 27 * 27
    经过 xenc @ 后, 仅仅把 xenc 那几行抽出来了, 其它行都被过滤掉了
    """

    logits = xenc @ w
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    """
    每一行的概率的平均值
    """

    loss = -probs[torch.arange(5), ys].mean().log()
    print(f"loss: {loss:.4f}")

    for i in range(100):
        logits = xenc @ w
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        """
        每一行的概率的平均值
        """

        loss = -probs[torch.arange(5), ys].mean().log()
        print(f"loss: {loss:.4f}")

        w.grad = None  # type of w is troch.tensor

        loss.backward()
        if w.grad is not None:
            w.data += (-1.5) * w.grad
        # print(f"loss: {loss}")


fine_tune()
