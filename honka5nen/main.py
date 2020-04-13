# データの前処理はscdv_cheack.py
# 学習からハイパーパラメータ最適はmodel.py
import argparse
import numpy as np
from scdv_check import SparseCompositeDocumentVectors, build_word2vec, plot_scatter, get_resample
from model import default_voting,default_xgb_object,svm_object, xgb_object
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

#　Word2Vecの引数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Word2VecとSCDVのパラメータの設定"
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=200
    )
    parser.add_argument(
        '--min_count', type=int, default=0
    )
    parser.add_argument(
        '--window_size', type=int, default=5
    )
    parser.add_argument(
        '--sg', type=int, default=1
    )
    parser.add_argument(
        '--num_clusters', type=int, default=2
    )
    parser.add_argument(
        '--pname1', type=str, default="gmm_cluster.pkl"
    )
    parser.add_argument(
        '--pname2', type=str, default="gmm_prob_cluster.pkl"
    )

    return parser.parse_args()


def main(args):
    # X:Dataset Y:Label
    with open("C:/Users/adisonax/Desktop/gati/matome.txt") as f:
        X = f.read().split("\n")[:-1]
    with open("C:/Users/adisonax/Desktop/gati/matomelabel.txt") as f:
        Y = f.read().split("\n")[:-1]
    Y = [int(i) for i in Y]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    # ====================ここからscdv_check側の処理=============== #
    #new_X_train = delete_stopword(X_train)
    # Word2Vecを作る  embedding_dimは次元
    model = build_word2vec(
            X,
            args.embedding_dim,
            args.min_count,
            args.window_size,
            args.sg
    )
    vec = SparseCompositeDocumentVectors(
            model,
            args.num_clusters,
            args.embedding_dim,
            args.pname1,
            args.pname2
    )
    # 確率重み付き単語ベクトルを求める
    vec.get_probability_word_vectors(X_train)
    # 訓練データからSCDVを求める
    train_gwbowv = vec.make_gwbowv(X_train)
    # テストデータからSCDVを求める
    test_gwbowv = vec.make_gwbowv(X_test)
    # ========================ここまで============================= #
    X_res_train, Y_res_train = get_resample(train_gwbowv, Y_train)
    
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    study.optimize(lambda trial: xgb_object(trial,X_res_train,Y_res_train,test_gwbowv,Y_test), n_trials=100)

    print(study.best_value)
    print(study.best_params)
    df = study.trials_dataframe()
    df.to_csv('C:/Users/adisonax/Desktop/tuning_param.tsv', sep='\t')
    
    # svm_object(X_res_train, Y_res_train, test_gwbowv, Y_test)

if __name__ == '__main__' :
    main(parse_args())
