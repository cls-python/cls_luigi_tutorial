import json
import pickle

import luigi
import networkx as nx
from cls.debug_util import deep_str
from cls_luigi.grammar import ApplicativeTreeGrammarEncoder, get_hypergraph_dict_from_tree_grammar, build_hypergraph, \
    render_hypergraph_components
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoLars
from os.path import join as pjoin

from utils import print_tree

output_dir = "output"

class LoadDiabetesData(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return {
            "x": luigi.LocalTarget(f"{output_dir}/x.pkl"),
            "y": luigi.LocalTarget(f"{output_dir}/y.pkl"),
        }

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])
        x = df.drop(["target"], axis="columns")
        y = df[["target"]]

        x.to_pickle(self.output()["x"].path)
        y.to_pickle(self.output()["y"].path)

class Composable(luigi.Task, LuigiCombinator):
    abstract = True
    features = ClsParameter(tpe=LoadDiabetesData.return_type())

    def requires(self):
        return self.features()

    def output(self):
        return {"x": luigi.LocalTarget(f"{output_dir}/x_{self.task_id}.pkl")}


class Decomposition(Composable):
    abstract = True

class PCADecomposition(Decomposition):
    abstract = False

    def run(self):
        x = pd.read_pickle(self.input()["x"].path)
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(x)
        scaled_x = pd.DataFrame(pca.transform(x),
                                      columns=[f"component_{i}" for i in range(n_components)],
                                      index=x.index)
        scaled_x.to_pickle(self.output()["x"].path)

class ICA(Decomposition):
    abstract = False

    def run(self):
        x = pd.read_pickle(self.input()["x"].path)
        n_components = 2
        ica = FastICA(n_components=n_components)
        ica.fit(x)
        scaled_x = pd.DataFrame(ica.transform(x),
                                      columns=[f"component_{i}" for i in range(n_components)],
                                      index=x.index)
        scaled_x.to_pickle(self.output()["x"].path)


class Scaler(Composable):
    abstract = True
    features = ClsParameter(tpe=Decomposition.return_type())

    def requires(self):
        return self.features()


class MinMax_Scaler(Scaler):
    abstract = False

    def run(self):
        x = pd.read_pickle(self.input()["x"].path)
        scaler = MinMaxScaler()
        scaler.fit(x)
        scaled_x = pd.DataFrame(scaler.transform(x),
                                      columns=scaler.feature_names_in_,
                                      index=x.index)
        scaled_x.to_pickle(self.output()["x"].path)



class Robust_Scaler(Scaler):
    abstract = False

    def run(self):
        x = pd.read_pickle(self.input()["x"].path)
        scaler = RobustScaler()
        scaler.fit(x)
        scaled_x = pd.DataFrame(scaler.transform(x),
                                      columns=scaler.feature_names_in_,
                                      index=x.index)
        scaled_x.to_pickle(self.output()["x"].path)



class RegModel(luigi.Task, LuigiCombinator):
    abstract = True
    features = ClsParameter(tpe=Composable.return_type())
    target_values = ClsParameter(tpe=LoadDiabetesData.return_type())

    def requires(self):
        return {
            "features": self.features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "y_pred": luigi.LocalTarget(f"{output_dir}/y_{self.task_id}.pkl"),
            "mse": luigi.LocalTarget(f"{output_dir}/mse_{self.task_id}.txt"),
        }


class Linear_Reg(RegModel):
    abstract = False


    def run(self):
        x = pd.read_pickle(self.input()["features"]["x"].path)
        y = pd.read_pickle(self.input()["target_values"]["y"].path)

        reg = LinearRegression()
        reg.fit(x, y)

        y_pred = reg.predict(x)
        mse = mean_squared_error(y, y_pred)

        with open(self.output()["y_pred"].path, "wb") as f:
            pickle.dump(y_pred, f)

        with open(self.output()["mse"].path, "w") as f:
            f.write(str(mse))





class Lasso_Reg(RegModel):
    abstract = False

    def run(self):
        x = pd.read_pickle(self.input()["features"]["x"].path)
        y = pd.read_pickle(self.input()["target_values"]["y"].path)

        reg = LassoLars()
        reg.fit(x, y)

        y_pred = reg.predict(x)
        mse = mean_squared_error(y, y_pred)

        with open(self.output()["y_pred"].path, "wb") as f:
            pickle.dump(y_pred, f)

        with open(self.output()["mse"].path, "w") as f:
            f.write(str(mse))


if __name__ == '__main__':
    import os

    os.mkdir(output_dir)
    target_class = RegModel
    target = target_class.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    results = [t() for t in inhabitation_result.evaluated[0:max_results]]


    rtg = inhabitation_result.rules
    with open(pjoin(output_dir, "applicative_regular_tree_grammar.txt"), "w") as f:
        f.write(deep_str(rtg))

    tree_grammar = ApplicativeTreeGrammarEncoder(rtg, target_class.__name__).encode_into_tree_grammar()
    with open(pjoin(output_dir, "regular_tree_grammar.json"), "w") as f:
        json.dump(tree_grammar, f, indent=4)

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    with open(pjoin(output_dir, "grammar_nx_hypergraph.pkl"), "wb") as f:
        pickle.dump(hypergraph, f)

    nx.write_graphml(hypergraph, pjoin(output_dir, "grammar_nx_hypergraph.graphml"))
    render_hypergraph_components(hypergraph, pjoin(output_dir, "grammar_hypergraph.png"), node_size=9000,
                                 node_font_size=11, show=True)
    for r in results:
        print(print_tree(r))

    if results:
        print("Number of pipelines", len(results))
        print("Running Pipelines...")
        luigi.build(results, local_scheduler=True)
    else:
        print("No results!")