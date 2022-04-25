import code
import sys
import os 
args = sys.argv

if args[0]=="train":
    os.system("nvidia-smi")
    print("モデルの学習を開始します")
    code.train()
    code.val()
    trained_param = code.train(args[1],args[2])




if args[0] == "classifier" :
    print("分類開始")
    code.classifier()
    print("FIN")
