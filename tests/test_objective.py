import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import penalized  # noqa: E402
from plots import pareto_front  # noqa: E402
import pandas as pd  # noqa: E402


def test_penalty_only_when_constraint_violated():
    assert penalized(0.95, 0.5, s_max=1.0) == 0.95          # feasible: J = F1
    assert penalized(0.95, 1.0, s_max=1.0) == 0.95          # on the boundary: still feasible
    assert abs(penalized(0.95, 1.5, s_max=1.0) - 0.45) < 1e-12  # 50% over budget -> -0.5
    assert penalized(0.99, 3.0, s_max=1.0) < penalized(0.90, 0.9, s_max=1.0)  # big infeasible loses


def test_pareto_front_keeps_only_non_dominated():
    d = pd.DataFrame({"size_mb": [0.1, 0.2, 0.3, 0.4], "cv_f1": [0.90, 0.89, 0.95, 0.94]})
    assert pareto_front(d)["size_mb"].tolist() == [0.1, 0.3]


if __name__ == "__main__":
    test_penalty_only_when_constraint_violated()
    test_pareto_front_keeps_only_non_dominated()
    print("ok")
