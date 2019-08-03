import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

# 現在の単語の数
word_count = 0
# 単語リスト
word_list = {}

# 読込
f = codecs.open("text/alice-wakati.txt", "r", "utf-8")

# 1行ずつ処理
line = f.readline()
while line:
    # 行の中の単語をリスト化
    l = line.split()
    for w in l:
        if not w in word_list:
            word_list[w]  = word_count + 2
            word_count = word_count + 1

    line = f.readline()

# 単語と単語リストを保存する
r = codecs.open("text/all-words.txt", "w", "utf-8")
for w in word_list:
    r.write(str(word_list[w]) + "," + w + "\n")
r.close()

# 文章を単語インデックスのリストに変換する
f.seek(0)
r = codecs.open("text/all-sentense.txt", "w", "utf-8")

# 文章になる単語のリスト
sentence = []

line = f.readline()
while line:
    # 行の中の単語をリスト化
    l = line.split()
    for w in l:
        # 今の単語を分に追加する
        sentence.append(word_list[w])
    # 改行で保存する
    for i in range(len(sentence)):
        r.write(str(sentence[i]))
        if i < len(sentence) - 1:
            r.write(",")

    r.write("\n")
    sentense = []
    line = f.readline()

f.close()
r.close()