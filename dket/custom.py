from nltk.draw.tree import draw_trees
from nltk.parse.generate import generate
import nltk

myGrammar = nltk.data.load('file:../resources/grammar.cfg')
sent = "all/DT @JJ/JJ @NN/NN @VB/VB only/RB @JJ/JJ or/CC @JJ/JJ @NN/NN ./. <EOS>/<EOS>".split()
# sent = "a/DT @NN/NN is/VBZ also/RB a/DT @NN/NN that/WDT @VB/VB @NN/NN ./. <EOS>/<EOS>".split()

parser = nltk.RecursiveDescentParser(myGrammar) # 158 rules
# parser = nltk.ChartParser(myGrammar)
# parser = nltk.ShiftReduceParser(myGrammar)

for tree in parser.parse(sent):
    tree.pretty_print()
    print("@JJ @NN := A @VB . ( @JJ @NN U @JJ @NN ) <EOS>")
    exit()
    # print(tree)

# print(len(list(generate(myGrammar, n=1000000))))
for sentence in generate(myGrammar, n=1):
    print(' '.join(sentence))
    for tree in parser.parse(sentence):
        tree.pretty_print()


# for n, sent in enumerate(generate(myGrammar, n=2), 1):
#     print('%3d. %s' % (n, ' '.join(sent)))

# print(len(trees))
# if len(trees) > 0:
#     draw_trees(*trees)

# print(parser.grammar())

# a/DT bee/NN is/VBZ also/RB a/DT insect/NN that/WDT produce/VB honey/NN ./. <EOS>/<EOS>	bee := insect ^ E produce . ( honey ) <EOS>
# <UNK>/NN weigh/VB exactly/RB NUM/CD hit/NN ./. <EOS>/<EOS>	LOC#0 := = LOC#3 LOC#1 . ( LOC#4 ) <EOS>
# all/DT narrow/JJ soap/NN testify/VB only/RB rambunctious/JJ or/CC craniofacial/JJ region/NN ./. <EOS>/<EOS>	LOC#1 LOC#2 := A LOC#3 . ( LOC#5 LOC#8 U LOC#7 LOC#8 ) <EOS>
# branch/NN are/VBP part/NN of/IN tree/NN ./. <EOS>/<EOS>	branch := E part of . ( tree ) <EOS>
# leaf/NN are/VBP part/NN of/IN branch/NN ./. <EOS>/<EOS>	leaf := E part of . ( branch ) <EOS>
