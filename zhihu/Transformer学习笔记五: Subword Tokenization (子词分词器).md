---
title: "Transformer学习笔记五: Subword Tokenization (子词分词器)"
url: https://zhuanlan.zhihu.com/p/460678461
author: 猛猿
date: 2026-10-14
---

关于[Transformer](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=Transformer&zhida_source=entity)的笔记, 预计出如下几篇:

1. [Positional Encoding (位置编码), 点击跳转](https://zhuanlan.zhihu.com/p/454482273)
2. [Self-attention (自注意力机制), 点击跳转](https://zhuanlan.zhihu.com/p/455399791)
3. [Batch Norm & Layer Norm (批量标准化/层标准化), 点击跳转](https://zhuanlan.zhihu.com/p/456863215)
4. [ResNet (残差网络), 点击跳转](https://zhuanlan.zhihu.com/p/459065530)
5. [Subword Tokenization](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=Subword+Tokenization&zhida_source=entity) (子词分词法)
6. 组装: Transformer

> 封面故事:  
> 绿色的刹车(Skids)和棕色的挡泥板(Mudflap)是"变形金钢2"里的汽车人双胞胎, 但它们最讨厌别人说彼此长得像. 在电影里负责吐槽搞笑讲骚话.  
> 在"变2"中, 双胞胎+[大黄蜂](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E5%A4%A7%E9%BB%84%E8%9C%82&zhida_source=entity)+前第七区被迫退休且没有拿到退休金的特工[西蒙](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E8%A5%BF%E8%92%99&zhida_source=entity)组成了一个奇妙的组合, 成功阻止了霸天虎下的巨型机器龙 (大概是这么个东西) 夺取太阳能量的野心. 凭借较小的身体, 双胞胎在机器龙的嘴巴里来回穿梭, 给了它不少的苦头.  
> 分词器对于自然语言模型来说, 也有些相似双胞胎 (这里笔者为了把它们关联上开始硬扯): 小, 常被用但经常被忽略其本身, 是任务开始的第一环.  
> 那就一起来看看它们吧!

这篇笔记是Transformer系列的最后一个零件, 讲述将语料处理成入模token的方法. 笔记结构如下:

一, 封面故事

二, 分词器的组成

三, 分词器的类别

- 2.1 Word-level: 基于空格的分词器
- 2.2 Character-level: 基于字符的分词器
- 2.3 Subword-level: 基于子词的分词器

四, Byte Pair Encoding ([BPE](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=BPE&zhida_source=entity))

- 3.1 BPE的训练步骤 (Trainer)
- 3.2 BPE的编码步骤 (Encoder)
- 3.3 BPE的解码步骤 (Decoder)

五, [WordPiece](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=WordPiece&zhida_source=entity)

- 4.1 WordPiece的训练步骤 (Trainer)
- 4.2 WordPiece的编码和解码步骤

六, Unigram Language Model (ULM)

- 5.1 ULM的训练步骤 (Trainer)
- 5.2 ULM的编码和解码步骤

七, 参考

  

## 一, 分词器的组成

  

如前所说, 分词器的作用是将预料拆分成能入模的token.

分词器一般包含以下几个部分:

- **Trainer (分词训练器/算法)**: 给定训练数据, 用Trainer封装的[分词算法](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E5%88%86%E8%AF%8D%E7%AE%97%E6%B3%95&zhida_source=entity), 生成词表
- **Vocabulary (词表)**: 词表是分词的标准. 基本上每个语言任务都需要维护一个词表, 词表可以是根据当前训练数据生成的, 也可以直接加载前人预训练好的词表.
- **Encoder (编码器)**: 拥有词表后, 给定语料, 编码器把语料按词表拆分token, 并记录token在词表中的索引, 拥有该索引则可将token表示成one-[hot向量](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=hot%E5%90%91%E9%87%8F&zhida_source=entity)入模, 再做后续的[embedding](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=embedding&zhida_source=entity)等优化操作.
- **Decoder (解码器)**: 给定一串索引, 将索引还原成最初的语料形式.

我们用HuggingFace中提供的Bert分词器做一个例子, 在Bert中使用分词算法是WordPiece (在这篇笔记里会介绍).

```python3
>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
>>> sequence = "A Titan RTX has 24GB of VRAM"

>>> tokenized_sequence = tokenizer.tokenize(sequence) # 用定义的分词器对语料分词
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

分词器将语料拆分为多个token. 以`'R', '##T', '##X'`为例, 这里是分词器在说: "我的词表里没有`RTX`这个词, 只有`R,T,X`三个字母, 因此我只好把它们按词表拆开. 但我知道`RTX`代表一个我不认识的整体, 因此我用`##`把它们连接起来, 团魂不灭!

我们也可以一步到位, 把上面的文字token转变为在词表中的索引, 其中101, 102是Bert中输入的特殊标志符号`CLS`和`SEP`.

```text
>>> inputs = tokenizer(sequence)
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]

>>> decoded_sequence = tokenizer.decode(encoded_sequence)
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

  

## 二, 分词器的类别

  

### 2.1 Word-level: 基于空格的分词器

早期的做法, 是按空格拆分单词, 将一个单词做为一个token纳入词表, 因此也说是word-level维度的. 若语料里出现不在词表中的token, 也称**OOV(Out Of Vocabulary)**, 则此时常用`<UNK>`(Unknown)这个特殊符号来代替. 这个分词器存在以下几个问题:

- 大词表问题. 词表大小和一门语言词汇量大小密切相关, 当词表太大时, 一来增加模型的存储负担. 二来, **token的one-hot向量也会非常长, 增加后续embedding的训练参数**.
- 罕见词难表示.
- 词汇冗余. 例如love, loving, loved其实拥有相似的含义, 但在词表中作为独立词存在.

### 2.2 Character-level: 基于字符的分词器

每个字符作为一个词. 例如英语中只有26个字符, 那词表大小就只有26个. 但这个level上依然存在问题:

- 单个字符本身缺少语义.
- 输入序列变长. 从原来的一句话变成若干个字符

### 2.3 Subword-level: 基于子词的分词器

子词分词器类似于借助[词根词源](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E8%AF%8D%E6%A0%B9%E8%AF%8D%E6%BA%90&zhida_source=entity)来学习一系列单词. 例如`transformer = trans + form + er`, `transfer = trans + fer`. 借助这些共通的基本单元, 就可以构造更多的词汇. 每一个单元就是词表的组成. 当然, 我们的分词器是基于统计意义的分词器, 不是基于语言学的. 只是当它看过大量训练数据以后, 它能学习到语言学上的含义. 主要的子词分词器有三类: BPE, WordPiece和ULM.

  

## 三, Byte Pair Encoding (BPE)

  

Transformer中, 在对[WMT](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=WMT&zhida_source=entity) 2014 English-German数据集进行处理的时候, 就用到了BPE. Sennrich et al. 2015最早将BPE用在Neural Machine Translation领域上, 这个算法是目前最流行的[NLP分词](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=NLP%E5%88%86%E8%AF%8D&zhida_source=entity)方法之一 (GPT-2采用的也是该方法).

### 3.1 BPE的训练步骤 (Trainer)

- 收集训练语料, 并确定期望的subword大小, 也就是词表大小
- 将语料中每一个单词的末尾添上特殊字符`</w>`, 该字符表示单词的结束. 举例来说, 同样是`er`, 它在`lower`和在`era`中的意义就不一样, 这就是为什么要加特殊字符.
- [统计单词](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E7%BB%9F%E8%AE%A1%E5%8D%95%E8%AF%8D&zhida_source=entity)在整个训练语料中的出现频率. 例如单词low在训练语料中出现了5次, 则记为`'low</w>': 5`
- 把单词拆分成字符, 即`'low</w>': 5`变成`l o w </w>': 5`
- 统计所有连续字符对的出现频率. 假设我们有`l o w </w>': 5`和`'h o w </w>': 3`, 则此时有`('l', 'o'): 5, ('o', 'w'): 5+3 = 8, ('h', 'o'): 3`. 取出现频率最高的字符对将它们合并. 例如`('o', 'w')`出现频率最高, 则此时变成`'l ow</w>': 5, 'h ow</w>':3`, 此时的词表为`'l', 'ow', 'h'`.
- 重复上一步, 直到词表达到预定的大小, 或者所有连续字符对的最高出现频率为1

沿用Sennrich et al. 2015在论文中给出的例子`'low</w>':5 'lower</w>':2 'newest</w>':6 'widest</w>':3`, 这里展示一个demo来看下算法运作的流程:

```py
import re, collections

def get_stats(vocab: dict)->dict:
    """
    从当前训练数据中，计算所有连续字符对出现的频率
    Args:
        vocab: 当前训练数据，key为当前每个单词的形式，value为对应频率
               形式诸如：{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    Return
        pairs：所有连续字符对出现的频率，key为连续字符对，value为频率
               形式诸如：{('l', 'o'): 7, ('o', 'w'): 7, ('w', '</w>'): 5, 
                        ('w', 'e'): 8, ('e', 'r'): 2, ('r', '</w>'): 2, 
                        ('n', 'e'): 6, ('e', 'w'): 6, ('e', 's'): 9, 
                        ('s', 't'): 9, ('t', '</w>'): 9, ('w', 'i'): 3, 
                        ('i', 'd'): 3, ('d', 'e'): 3})
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair:tuple, v_in:tuple)-> dict:
    """
    用找出的频率最大的字符对（pair）去对当前训练数据做合并

    Args:
        pair: 当前频率最大的子词对，形式诸如('e', 's')
        v_in: 当前训练数据
    Return:
        v_out: 合并结果。例如用'es'替换'e s'
    """
    v_out = {}
    bigram = re.escape(' '.join(pair)) # 对字符串中所有可能被解释为正则运算的字符进行转义。例如'e s'变为'e\ s'
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # 创建正则匹配对象
    for word in v_in:
        w_out = p.sub(''.join(pair), word) # 对word中满足正则匹配的部分，用''.join(pair)代替
        v_out[w_out] = v_in[word]
    return v_out

def len_token(token: str)->int:
    """
    计算一个token的长度。主要是将特殊标记'</w>'当成一个字符来处理
    """
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

def get_tokens(vocab:dict)->dict:
    """
    根据当前训练数据生成当下的bpe词表
    """
    bpe_vocab = collections.defaultdict(int)
    for word, freq in vocab.items():
        for symbol in word.split():
            bpe_vocab[symbol] += freq
    bpe_vocab = dict(sorted(bpe_vocab.items(), key=lambda x: len_token(x[0]), reverse = True))

    return bpe_vocab

def generate_bpe_vocab(v_in:dict, num_merges: int) -> dict:
    """
    生成bpe词表
    Args:
        v_in: 原始训练数据
        num_merges: 合并次数，也是预设的最终的词表大小
    Return:
        bpe_vocab: bpe词表
    """
    v_out = v_in.copy()
    bpe_vocab = get_tokens(v_in)
    print('执行BPE前：')
    print('训练数据为：{}'.format(v_in))
    print('初始化词表为: {}'.format(bpe_vocab.keys()))
    print('初始化词表长度为: {}'.format(len(bpe_vocab)))
    
    for i in range(num_merges):
        print('=' * 50)
        print('Iter: {}'.format(i))
        pairs = get_stats(v_out)
        try:
            best = max(pairs, key = pairs.get)
        except ValueError:
            print('所有词汇已不可拆分，停止生成词表')
            break
        if pairs[best] < 2:
            print('字符对出现频率小于2，停止生成词表')
            break
        v_out = merge_vocab(best, v_out)
        bpe_vocab = get_tokens(v_out)
        print('当前Iter出现频率最大的连续字符对为：{}'.format(best))
        print('当前Iter合并后的新训练数据为: {}'.format(v_out))
        print('当前Iter新生成的词表为：{}'.format(bpe_vocab.keys()))
        print('当前Iter新生成的词表长度为: {}'.format(len(bpe_vocab)))
    
    return bpe_vocab

if __name__ == '__main__':
    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    num_merges = 7
    bpe_tokens = generate_bpe_vocab(vocab, num_merges)
```

  

运行结果如下:

  

![图片](images/448df3146ee5.jpg)

###   
  
3.2 BPE的编码步骤 (Encoder)

  
训练步骤结束后, 我们得到一个词表, Encoder阶段要解决的事情, 就是给定一个语料, 基于词表对该语料进行分词, 具体步骤如下:  

- 将词表中的token按长度从长到短进行排序 (上述代码已经实现了这一点)
- 对语料中的每一个单词, [遍历排序](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E9%81%8D%E5%8E%86%E6%8E%92%E5%BA%8F&zhida_source=entity)好的词表, 查看当前的token是否是该单词的子字符串. 如果是, 则输出当前token, 并对剩余的字符串继续匹配.
- 当遍历完一次词表过后, 如果仍有字符串在词表中找不到匹配, 就把这些字符串用特殊符号输出, 比如`<UNK>`

以上述代码生成的词表`['est</w>', 'low', 'new', '</w>', 'e', 'r', 'w', 'i', 'd']`为例, 假设有一单词为`hiest`, 则该单词最终将被切分为`<UNK> i est</w>`.  

###   
3.3 BPE的解码步骤 (Decoder)

  
当相邻的token间没有特殊符号 (例如空格/终止符), 则表示这两个token同属一个单词, 将它们直接拼接即可. 否则需要在两个token间添加分隔符.  
  

##   
四, WordPiece

  
Bert模型在做分词的时候用的就是WordPiece方法. WordPiece方法和BPE总体很相似, 不同的是, 在选择合并字符时, BPE采用的标准是"频率最高", 而WordPiece则是"概率最大". 具体来说, WordPiece在每次合并时将选择一对相邻的子字符, 这对子字符满足: 当合并它们时, 它们对语料库语言模型的概率提升最大.  
来具体看一下算法.

###   
4.1 WordPiece的训练步骤 (Trainer)

  
假设句子 $S = (t_1, t_2,...,t_n)$ , 其中 $t_i$ 表示第i个子词 (初始状态是字符, 经过一定的合并步骤以后变成子词, 所以这里统一用 $t$ 来表示). 假设各个子词之间相互独立, 则有:

  
$logP(S) = \sum_{i=1}^{n} logP(t_i)$

  
对于某两个相邻位置x和y, 若将对应位置的子词合并, 记合并后的子词为z, 则合并前后语言模型概率的变化为:

  
$logP(t_z) - (logP(t_x) + logP(t_y)) = log(\frac{P(t_z)}{P(t_x)P(t_y)})$

  
则对于这个句子, 我们要做的就是找到使得 $log(\frac{P(t_z)}{P(t_x)P(t_y)})$ 最大的那个相邻 $(t_x, t_y)$ 对, 这个最大对在语言模型上具有较强的关联性, 因此我们选择合并它们.

  
Encoder和Decoder的方法和BPE相似, 因此这里略去. 同时也注意到, WordPiece和BPE一样, 都没有对不同的分词结果进行比较.  

##   
五, Unigram Language Model (ULM)

  
总结BPE和WordPiece, 我们发现它们都是通过一定的方法, 生成一个最终的词表, 然后对于某个单词或句子, 根据这个最终的词表, 生成**唯一**的划分结果.  
  
但是, **这个划分结果一定就是最好的吗**? 假设通过BPE或WordPiece, 生成了一张词表`est</w>, st</w>, w, e, s, t`, 那么对于单词`west`, 它有三种划分方式: `w est</w>`, `w e st</w>`和`w e s t`. 根据前两个算法的[编码规则](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E7%BC%96%E7%A0%81%E8%A7%84%E5%88%99&zhida_source=entity), 我们选择了第一个, 但是也许在后续做训练任务时, 其实第三个的表现更好呢?  
  
因此, ULM就作为一种改进办法出现了. 概括来说, 它的改进思路是:  

- 首先拥有一种大词表
- 基于这张词表, 对所有语料的所有子词划分结果做一个总体评分. 然后从词表中**剔除掉那些对总体评分贡献最小的子词**
- 重复第二步, 不断丢弃贡献度小的子词, 直到词表达到预设大小
- 生成最终的词表. 基于该词表, 一个语料仍可能有多种划分, 通过某种方式, 选择其中最优的划分.

基于这个改进思想, 来详细看下算法过程.

###   
  
5.1 ULM的训练步骤 (Trainer)  

- 初始化一个[大词表](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=3&q=%E5%A4%A7%E8%AF%8D%E8%A1%A8&zhida_source=entity). 这个词表可以是语料中所有字符 + 常见的子字符串, 也可以用BPE进行初始化.
- 基于该词表, 对所有语料的所有子词划分结果做一个总体评价. 具体的实现方式是:

设语料库中有 $|D|$ 条语料. $X^{(s)}$ 表示语料库中的第 $s$ 条语料. $\mathcal S(X^{(s)})$ 表示语料 $X^{(s)}$ 所有划分结果的集合. $\vec x$ 表示其中一种划分结果, 满足 $\vec x \in \mathcal S(X^{(s)})$ . 同时, 我们**假设词表中每一个子词相互独立 (Unigram Language Model的重要前提).** 则我们的**评价标准** $\mathcal L$ 定义为: 预料库中的所有语料的所有分词组合形成的概率相加, 即:

$\mathcal L = \sum_{s=1}^{|D|}log(P(X^{(s)})) = \sum_{s=1}^{|D|}log(\sum_{\vec x\in\mathcal S(X^{(s)})}P(\vec x) )$

  
其中, 假设某个划分结果 $\vec x$ 中含有m个子词, 即 $\vec x = (x_1, x_2, ..., x_m)$ , 那么有:

  
$P(\vec x) = \prod_{i=1}^{m} P(x_i)$

基于当前词表, 通过EM算法, 我们可以求得$$P(x\_i)$$的概率, 也就是每个子词在整个语料库上的概 率. 进而就可以求得 $\mathcal L$ .  

- 从当前词表中删除对 $\mathcal L$贡献度小的子词. 具体实现方式是:

对于每个字词, 计算当该子词从词表中移除时, 总的 $\mathcal L$ 降低了多少, 这个降低记为该字词的 $loss$ . 将 $loss$ 从小到大进行排序, **丢弃掉一定比例的** $loss$ **最小的子词**. 可以理解为, 这些子词对整个语料库的影响不大. 同时, **单字符的子词不要丢弃, 这是为了避免OOV的情况**.  

- 重复步骤2和3, 直到词表达到预设大小. 同时保留最终计算得到的 $P(x_i)$ 结果. 这个结果将在encoder阶段使用.

###   
5.2 ULM的编码步骤 (Encoder)

  
编码步骤里, 将基于最终生成的词表, 找到一个语料最优的划分方式, 设这个最有的划分方式为 $\vec x^*$ , 则我们的目的其实是:

  
$\vec x^* = argmax_{\vec x \in \mathcal S(X^{(s)})}P(\vec x)$

  
根据训练阶段保留下来的 $P(x_i)$ , 我们可以用 **[维特比算法](https://zhida.zhihu.com/search?content_id=190502949&content_type=Article&match_order=1&q=%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95&zhida_source=entity)(viterbi)** 来得到最终最优的划分.  
解码步骤和前面两个分词算法一样, 这里就不再提了.