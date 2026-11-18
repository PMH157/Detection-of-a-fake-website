import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import penalized  # noqa: E402
from plots import pareto_front  # noqa: E402
from ofat_search import coordinate_search  # noqa: E402
from step_summary import norm, to_kwargs  # noqa: E402
import pandas as pd  # noqa: E402


def test_default_objective_is_plain_f1():
    assert penalized(0.95, 30.0) == 0.95                    # default lambda = 0: size is not penalised


def test_penalty_only_when_constraint_violated():  # archived 1 MB study (lambda = 1)
    assert penalized(0.95, 0.5, s_max=1.0, lam=1.0) == 0.95          # feasible: J = F1
    assert penalized(0.95, 1.0, s_max=1.0, lam=1.0) == 0.95          # on the boundary: still feasible
    assert abs(penalized(0.95, 1.5, s_max=1.0, lam=1.0) - 0.45) < 1e-12  # 50% over budget -> -0.5
    assert penalized(0.99, 3.0, s_max=1.0, lam=1.0) < penalized(0.90, 0.9, s_max=1.0, lam=1.0)


def test_pareto_front_keeps_only_non_dominated():
    d = pd.DataFrame({"size_mb": [0.1, 0.2, 0.3, 0.4], "cv_f1": [0.90, 0.89, 0.95, 0.94]})
    assert pareto_front(d)["size_mb"].tolist() == [0.1, 0.3]


def test_coordinate_search_changes_each_param_once_and_keeps_better_current():
    target = {"a": 3, "b": 7}
    calls = []
    def evaluate(p):
        calls.append(dict(p))
        return -sum((p[k] - target[k]) ** 2 for k in target), {}
    versions, log = coordinate_search({"a": 0, "b": 7}, {"a": [1, 3, 5], "b": [2, 4]}, ["a", "b"], evaluate)
    assert [v["changed"] for v in versions] == ["-", "a", "b"]
    assert (versions[1]["a"], versions[1]["b"]) == (3, 7)      # step 1 tunes only a
    assert (versions[2]["a"], versions[2]["b"]) == (3, 7)      # b=7 already optimal: no grid value beats it, kept
    assert len(log) == 5 and len(calls) == 6                    # 3 + 2 grid evaluations + 1 start point


def test_norm_gives_one_form_per_value():
    assert [norm(v) for v in (None, float("nan"), 11.0, 11, 0.05, "sqrt")] == ["None", "None", "11", "11", "0.05", "sqrt"]
    assert norm(True) == "True" and norm(False) == "False" and norm("False") == "False"
    kw = to_kwargs({"n_estimators": "300", "max_depth": "None", "max_features": "0.05", "bootstrap": "False",
                    "class_weight": "balanced", "ccp_alpha": "0"})
    assert kw == {"n_estimators": 300, "max_depth": None, "max_features": 0.05, "bootstrap": False,
                  "class_weight": "balanced", "ccp_alpha": 0.0}


if __name__ == "__main__":
    test_default_objective_is_plain_f1()
    test_penalty_only_when_constraint_violated()
    test_pareto_front_keeps_only_non_dominated()
    test_coordinate_search_changes_each_param_once_and_keeps_better_current()
    test_norm_gives_one_form_per_value()
    print("ok")
