import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import sys
import codecs

batch_size = 10
uses_device = 0

if uses_device >= 0:
    # GPU使用時
    import cupy as cp
else:
    # CPU使用時
    cp = np

# RNNクラス
class Generate_RNN(chainer.Chain):

    def __init__(self, n_words, nodes):
        super(Generate_RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, n_words)
            self.l1 = L.LSTM(n_words, nodes)
            self.l2 = L.LSTM(nodes, nodes)
            self.l3 = L.Linear(nodes, n_words)

    # 内部ステータスのリセット
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)

        return y

class RNNUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(RNNUpdater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )

    def update_core(self):
        # 累積していく損失
        loss = 0

        # IteratorとOptimizerを取得
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        # ニューラルネットワークを取得
        model = optimizer.target

        # 文を1バッチ分取得
        x = train_iter.__next__()

        # RNNのステータスをリセット
        model.reset_state()

        # 文の長さだけ繰り返しRNNに学習
        for i in range(len[0] - 1):
            # バッチ処理用の配列に変換
            batch = cp.array([s[i] for s in x], dtype=cp.int32)
            # 正解データ配列
            t = cp.array([s[i + 1] for s in x], dtype=cp.int32)
            # 終端文字であれば学習中断
            if cp.min(batch) == 1 and cp.max(batch) == 1:
                break
            # RNNを実行
            y = model(batch)
            # 誤差の判定
            loss += F.softmax_cross_entropy(y, t)

        # 重みをリセット
        optimizer.target.cleargrads()
        # 誤差関数から逆伝播
        loss.backward()
        # 新しい重みデータで更新
        optimizer.update()

# 読込
s = codecs.open("all-sentence.txt", "r", "utf-8")

# 全ての分
sentence = []

# 1行ずつ処理する
line = s.readline()
while line:
    # 1つの文(開始文字のみ)
    one = [0]
    # 行の中の単語を数字のリストにして追加
    one.extend(list(map(int, line.split(","))))
    # 終端文字を入れる
    one.append(1)
    # 新しい分を追加
    sentence.append(one)
    line = s.readline()

# クローズ処理
s.close()

# 単語の種類
n_word = max([max(l) for l in sentence]) + 1

# 最長の文の長さ
l_max = max([len(l) for l in sentence])
# バッチ処理の都合により長さを揃える
for i in range(len(sentence)):
    # 不足分は終端文字で埋める
    sentence[i].extend([1] * (l_max - len(sentence[i])))

# ニューラルネットワークの生成
model = Generate_RNN(n_word, 200)

if uses_device >= 0:
    # GPUを使用
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()
    # データの変換
    model.to_gpu()

# 誤差逆伝播法を選択
optimizer = optimizers.Adam()
optimizer.setup(model)

# iteratorを作成
train_iter = iterators.SerialIterator(sentence, batch_size, shuffle=False)

# Trainerを生成
updater = RNNUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (30, "epoch"), out="result")

# 進捗を可視化
trainer.extend(extensions.ProgressBar(update_interval=1))

# 学習の実行
trainer.run()

# 結果の保存
chainer.serializers.save_hdf5("sentence_model.hdf5", model)