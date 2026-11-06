# evaluate_model.py
# ==============================================================================
# "A Comprehensive Analysis on the Skill-Set of the Students to Improve Campus
#  Drive Using Machine Learning Approaches"
#
# EVALUATION INTEGRITY & METHODOLOGY TAXONOMY (PHASE 4, 5 & 6):
#   Every metric reported by this script is categorized into one of four
#   explicit evaluation classes and saved to output/reports/model_metrics.csv
#   and output/reports/evaluation_report.txt:
#     1. [UNSUPERVISED_CLUSTERING]:
#        Silhouette Score, Davies-Bouldin Index (DBI), Calinski-Harabasz Index
#        (CHI) across K in {2, 3, 4, 5}, salary-weight sensitivity analysis,
#        and Bootstrap Resampling Stability (ARI, NMI, Exact Agreement).
#     2. [RULE_BASED_BENCHMARK_AGREEMENT]:
#        Held-out 3x3 Student Skill-Band comparison against an independent
#        domain-weighted composite readiness index (reporting Confusion Matrix,
#        ARI, NMI, Quadratic Weighted Cohen's Kappa, Spearman's rho, Precision,
#        Recall, Specificity, and F1-score).
#     3. [GROUND_TRUTH_PLACEMENT_CLASSIFICATION]:
#        Leak-free held-out (80/20 stratified split) and 5-fold stratified CV
#        evaluation against actual placement_status labels from Dataset 2
#        (strictly excluding post-placement salary_lpa and offer_count),
#        reporting Confusion Matrix, Accuracy, Precision, Recall, Specificity,
#        Balanced Accuracy, F1-Score, ROC-AUC, PR-AUC, and Feature Importance.
#     4. [CONTROLLED_SYNTHETIC_EXPERIMENT] & [CAPACITY_CONSTRAINED_ALLOCATION]:
#        - Controlled synthetic role-recovery robustness test under skill
#          dropout and distractor noise (Hit-Rate@1, Hit-Rate@3, Hit-Rate@5, MRR).
#        - Actual invocation of student_company_mapping() comparing First-Come-
#          First-Served (FCFS), Random Eligible Assignment, Constrained-First
#          Greedy Heuristic, and Proposed Optimal MILP/Min-Cost Flow Allocation
#          under identical student batches, eligibility gates, and capacities.
# ==============================================================================

import contextlib
import importlib
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import comb
from scipy.stats import spearmanr

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        average_precision_score,
        balanced_accuracy_score,
        calinski_harabasz_score,
        cohen_kappa_score,
        confusion_matrix,
        davies_bouldin_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        silhouette_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.nanmean(X, axis=0)
            self.scale_ = np.nanstd(X, axis=0, ddof=0)
            self.scale_ = np.where((self.scale_ == 0) | np.isnan(self.scale_), 1.0, self.scale_)
            return self

        def transform(self, X):
            return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class KMeans:
        def __init__(self, n_clusters=3, n_init=10, random_state=42, max_iter=300, tol=1e-4):
            self.n_clusters = n_clusters
            self.n_init = n_init
            self.random_state = random_state
            self.max_iter = max_iter
            self.tol = tol

        def _init_centroids(self, X, rng):
            n_samples, n_features = X.shape
            centers = np.empty((self.n_clusters, n_features), dtype=float)
            centers[0] = X[rng.randint(0, n_samples)]
            closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
            for c in range(1, self.n_clusters):
                total = closest_dist_sq.sum()
                probs = closest_dist_sq / total if total > 1e-12 else np.full(n_samples, 1.0 / n_samples)
                centers[c] = X[rng.choice(n_samples, p=probs)]
                closest_dist_sq = np.minimum(closest_dist_sq, np.sum((X - centers[c]) ** 2, axis=1))
            return centers

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            rng = np.random.RandomState(self.random_state)
            best_inertia, best_centers, best_labels = np.inf, None, None
            for _ in range(self.n_init):
                centers = self._init_centroids(X, rng)
                for _ in range(self.max_iter):
                    dists = cdist(X, centers, metric="euclidean")
                    labels = np.argmin(dists, axis=1)
                    new_centers = np.array([
                        X[labels == k].mean(axis=0) if np.any(labels == k)
                        else X[rng.randint(0, len(X))]
                        for k in range(self.n_clusters)
                    ])
                    if np.sum((new_centers - centers) ** 2) <= self.tol:
                        centers = new_centers
                        break
                    centers = new_centers
                dists = cdist(X, centers, metric="euclidean")
                labels = np.argmin(dists, axis=1)
                inertia = float(np.sum(np.min(dists, axis=1) ** 2))
                if inertia < best_inertia:
                    best_inertia, best_centers, best_labels = inertia, centers, labels
            self.cluster_centers_ = best_centers
            self.labels_ = best_labels
            self.inertia_ = best_inertia
            return self

        def predict(self, X):
            return np.argmin(self.transform(X), axis=1)

        def fit_predict(self, X):
            return self.fit(X).labels_

        def transform(self, X):
            return cdist(np.asarray(X, dtype=float), self.cluster_centers_, metric="euclidean")

    def silhouette_score(X, labels):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(X):
            return 0.0
        dists = cdist(X, X, metric="euclidean")
        s = np.zeros(len(X), dtype=float)
        for i in range(len(X)):
            own_mask = (labels == labels[i])
            own_mask[i] = False
            a_i = dists[i, own_mask].mean() if np.any(own_mask) else 0.0
            b_i = min(
                dists[i, labels == c].mean()
                for c in unique_labels if c != labels[i] and np.any(labels == c)
            )
            s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        return float(np.mean(s))

    def davies_bouldin_score(X, labels):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        k = len(unique_labels)
        centers = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
        s = np.array([
            np.mean(np.linalg.norm(X[labels == c] - centers[idx], axis=1))
            for idx, c in enumerate(unique_labels)
        ])
        center_dists = cdist(centers, centers, metric="euclidean")
        db_vals = []
        for i in range(k):
            ratios = [
                (s[i] + s[j]) / center_dists[i, j]
                for j in range(k) if i != j and center_dists[i, j] > 0
            ]
            db_vals.append(max(ratios) if ratios else 0.0)
        return float(np.mean(db_vals))

    def calinski_harabasz_score(X, labels):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_samples, k = len(X), len(unique_labels)
        overall_mean = X.mean(axis=0)
        extra_disp, intra_disp = 0.0, 0.0
        for c in unique_labels:
            cluster_k = X[labels == c]
            mean_k = cluster_k.mean(axis=0)
            extra_disp += len(cluster_k) * np.sum((mean_k - overall_mean) ** 2)
            intra_disp += np.sum((cluster_k - mean_k) ** 2)
        return float((extra_disp * (n_samples - k)) / (intra_disp * (k - 1))) if intra_disp > 0 else 1.0

    def confusion_matrix(y_true, y_pred, labels=None):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        mat = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                mat[label_to_idx[t], label_to_idx[p]] += 1
        return mat

    def adjusted_rand_score(labels_true, labels_pred):
        labels_true, labels_pred = np.asarray(labels_true), np.asarray(labels_pred)
        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        contingency = np.zeros((len(classes), len(clusters)), dtype=int)
        for i, j in zip(class_idx, cluster_idx):
            contingency[i, j] += 1
        sum_comb_c = sum(comb(n_c, 2) for n_c in contingency.sum(axis=1))
        sum_comb_k = sum(comb(n_k, 2) for n_k in contingency.sum(axis=0))
        sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.ravel())
        n = len(labels_true)
        prod_comb = (sum_comb_c * sum_comb_k) / max(comb(n, 2), 1.0)
        mean_comb = (sum_comb_c + sum_comb_k) / 2.0
        return float((sum_comb - prod_comb) / (mean_comb - prod_comb)) if mean_comb != prod_comb else 1.0

    def cohen_kappa_score(y1, y2, weights=None):
        mat = confusion_matrix(y1, y2)
        n_classes = mat.shape[0]
        sum0, sum1 = mat.sum(axis=0), mat.sum(axis=1)
        expected = np.outer(sum1, sum0) / max(mat.sum(), 1.0)
        if weights is None:
            w = np.ones((n_classes, n_classes), dtype=float)
            np.fill_diagonal(w, 0.0)
        elif weights == "linear":
            i, j = np.indices((n_classes, n_classes))
            w = np.abs(i - j) / max(n_classes - 1, 1)
        elif weights == "quadratic":
            i, j = np.indices((n_classes, n_classes))
            w = ((i - j) ** 2) / max((n_classes - 1) ** 2, 1)
        else:
            raise ValueError("Unsupported weights")
        return float(1.0 - np.sum(w * mat) / max(np.sum(w * expected), 1e-12))

    def precision_score(y_true, y_pred, zero_division=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        return tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)

    def recall_score(y_true, y_pred, zero_division=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        return tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)

    def f1_score(y_true, y_pred, zero_division=0):
        p = precision_score(y_true, y_pred, zero_division=zero_division)
        r = recall_score(y_true, y_pred, zero_division=zero_division)
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)

    def balanced_accuracy_score(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        rec_pos = recall_score(y_true, y_pred, zero_division=0)
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.5 * (rec_pos + spec)

    def roc_curve(y_true, y_score):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        thresholds = np.sort(np.unique(y_score))[::-1]
        thresholds = np.concatenate([[thresholds[0] + 1.0], thresholds, [thresholds[-1] - 1.0]])
        pos = max(np.sum(y_true == 1), 1)
        neg = max(np.sum(y_true == 0), 1)
        tpr_list, fpr_list = [], []
        for thr in thresholds:
            pred = y_score >= thr
            tpr_list.append(float(np.sum((pred == 1) & (y_true == 1)) / pos))
            fpr_list.append(float(np.sum((pred == 1) & (y_true == 0)) / neg))
        return np.array(fpr_list), np.array(tpr_list), thresholds

    def roc_auc_score(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        order = np.argsort(fpr)
        return float(np.trapezoid(tpr[order], fpr[order]) if hasattr(np, "trapezoid") else np.trapz(tpr[order], fpr[order]))

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        thresholds = np.sort(np.unique(y_score))
        pos = max(np.sum(y_true == 1), 1)
        precisions, recalls = [], []
        for thr in thresholds:
            pred = y_score >= thr
            tp = float(np.sum((pred == 1) & (y_true == 1)))
            fp = float(np.sum((pred == 1) & (y_true == 0)))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
            recalls.append(tp / pos)
        precisions.append(1.0)
        recalls.append(0.0)
        return np.array(precisions), np.array(recalls), thresholds

    def average_precision_score(y_true, y_score):
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        # Sort by ascending recall
        order = np.argsort(rec)
        r_sorted = rec[order]
        p_sorted = prec[order]
        return float(np.sum(np.diff(r_sorted, prepend=0.0) * p_sorted))

    def train_test_split(*arrays, test_size=0.2, random_state=None, stratify=None):
        rng = np.random.RandomState(random_state)
        n = len(arrays[0])
        if stratify is not None:
            strat = np.asarray(stratify)
            train_idx, test_idx = [], []
            for cls in np.unique(strat):
                cls_idx = np.where(strat == cls)[0]
                cls_perm = rng.permutation(cls_idx)
                n_t = max(1, int(np.round(len(cls_idx) * test_size)))
                test_idx.extend(cls_perm[:n_t])
                train_idx.extend(cls_perm[n_t:])
            train_idx = rng.permutation(train_idx)
            test_idx = rng.permutation(test_idx)
        else:
            indices = rng.permutation(n)
            n_test = int(np.ceil(n * test_size))
            test_idx, train_idx = indices[:n_test], indices[n_test:]
        result = []
        for arr in arrays:
            if hasattr(arr, "iloc"):
                result.extend([arr.iloc[train_idx].reset_index(drop=True), arr.iloc[test_idx].reset_index(drop=True)])
            else:
                arr = np.asarray(arr)
                result.extend([arr[train_idx], arr[test_idx]])
        return result

    class StratifiedKFold:
        def __init__(self, n_splits=5, shuffle=True, random_state=42):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X, y):
            y = np.asarray(y)
            rng = np.random.RandomState(self.random_state)
            folds = [[] for _ in range(self.n_splits)]
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                if self.shuffle:
                    cls_idx = rng.permutation(cls_idx)
                for i, idx in enumerate(cls_idx):
                    folds[i % self.n_splits].append(idx)
            all_idx = np.arange(len(y))
            for k in range(self.n_splits):
                val_idx = np.array(folds[k], dtype=int)
                tr_mask = np.ones(len(y), dtype=bool)
                tr_mask[val_idx] = False
                yield all_idx[tr_mask], val_idx

    class LogisticRegression:
        """L2-Regularized Binary Logistic Regression solved via SciPy L-BFGS-B."""
        def __init__(self, random_state=42, max_iter=500, class_weight="balanced", C=1.0):
            self.random_state = random_state
            self.max_iter = max_iter
            self.class_weight = class_weight
            self.C = C

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            n_samples, n_features = X.shape
            if self.class_weight == "balanced":
                n_pos = max(np.sum(y == 1), 1.0)
                n_neg = max(np.sum(y == 0), 1.0)
                w_pos = n_samples / (2.0 * n_pos)
                w_neg = n_samples / (2.0 * n_neg)
                sample_w = np.where(y == 1, w_pos, w_neg)
            else:
                sample_w = np.ones(n_samples, dtype=float)

            def loss_and_grad(params):
                w = params[:-1]
                b = params[-1]
                z = np.clip(X @ w + b, -30.0, 30.0)
                p = 1.0 / (1.0 + np.exp(-z))
                eps = 1e-12
                loss = -np.sum(sample_w * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))) / n_samples
                loss += 0.5 * (1.0 / self.C) * np.sum(w ** 2) / n_samples
                diff = sample_w * (p - y) / n_samples
                grad_w = X.T @ diff + (1.0 / self.C) * w / n_samples
                grad_b = float(np.sum(diff))
                return loss, np.append(grad_w, grad_b)

            init_p = np.zeros(n_features + 1, dtype=float)
            res = minimize(loss_and_grad, init_p, method="L-BFGS-B", jac=True, options={"maxiter": self.max_iter})
            self.coef_ = res.x[:-1].reshape(1, -1)
            self.intercept_ = np.array([res.x[-1]])
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            z = np.clip(X @ self.coef_.ravel() + self.intercept_[0], -30.0, 30.0)
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    class _SimpleDecisionTree:
        """CART Decision Tree using exact weighted Gini impurity reduction."""
        def __init__(self, max_depth=5, min_samples_split=6, max_features="sqrt", rng=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.rng = rng or np.random.RandomState(42)

        def _gini(self, y, w):
            total_w = np.sum(w)
            if total_w <= 0:
                return 0.0
            p1 = np.sum(w[y == 1]) / total_w
            return 2.0 * p1 * (1.0 - p1)

        def _build(self, X, y, w, depth):
            total_w = float(np.sum(w))
            p1 = float(np.sum(w[y == 1]) / max(total_w, 1e-12))
            if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
                return {"leaf": True, "prob": p1}

            n_samples, n_features = X.shape
            if self.max_features == "sqrt":
                k_feat = max(1, int(np.sqrt(n_features)))
                feat_indices = self.rng.choice(n_features, size=k_feat, replace=False)
            else:
                feat_indices = np.arange(n_features)

            parent_gini = self._gini(y, w)
            best_gain, best_feat, best_thr = 0.0, None, None

            for f_idx in feat_indices:
                col = X[:, f_idx]
                uniq = np.unique(col)
                if len(uniq) <= 1:
                    continue
                thresholds = np.quantile(uniq, np.linspace(0.1, 0.9, min(9, len(uniq) - 1)))
                for thr in np.unique(thresholds):
                    left_mask = col <= thr
                    right_mask = ~left_mask
                    if not np.any(left_mask) or not np.any(right_mask):
                        continue
                    w_l, w_r = float(np.sum(w[left_mask])), float(np.sum(w[right_mask]))
                    gain = parent_gini - (w_l / total_w) * self._gini(y[left_mask], w[left_mask]) - (w_r / total_w) * self._gini(y[right_mask], w[right_mask])
                    if gain > best_gain:
                        best_gain, best_feat, best_thr = gain, int(f_idx), float(thr)

            if best_feat is None or best_gain <= 1e-7:
                return {"leaf": True, "prob": p1}

            self.importances_[best_feat] += best_gain * total_w
            left_m = X[:, best_feat] <= best_thr
            return {
                "leaf": False,
                "feat": best_feat,
                "thr": best_thr,
                "left": self._build(X[left_m], y[left_m], w[left_m], depth + 1),
                "right": self._build(X[~left_m], y[~left_m], w[~left_m], depth + 1),
            }

        def fit(self, X, y, sample_weight=None):
            n_samples, n_features = X.shape
            if sample_weight is None:
                sample_weight = np.ones(n_samples, dtype=float)
            self.importances_ = np.zeros(n_features, dtype=float)
            self.tree_ = self._build(X, y, sample_weight, 0)
            return self

        def _pred_one(self, node, row):
            while not node["leaf"]:
                node = node["left"] if row[node["feat"]] <= node["thr"] else node["right"]
            return node["prob"]

        def predict_proba(self, X):
            probs = np.array([self._pred_one(self.tree_, r) for r in X], dtype=float)
            return np.column_stack([1.0 - probs, probs])

    class RandomForestClassifier:
        """Bootstrap Random Forest Classifier with Gini feature importance."""
        def __init__(self, n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.random_state = random_state
            self.class_weight = class_weight

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=int)
            n_samples, n_features = X.shape
            rng = np.random.RandomState(self.random_state)
            if self.class_weight == "balanced":
                n_pos = max(np.sum(y == 1), 1)
                n_neg = max(np.sum(y == 0), 1)
                base_w = np.where(y == 1, n_samples / (2.0 * n_pos), n_samples / (2.0 * n_neg))
            else:
                base_w = np.ones(n_samples, dtype=float)

            self.trees_ = []
            imp_total = np.zeros(n_features, dtype=float)
            for _ in range(self.n_estimators):
                boot_idx = rng.randint(0, n_samples, size=n_samples)
                tree = _SimpleDecisionTree(max_depth=self.max_depth, max_features="sqrt", rng=rng)
                tree.fit(X[boot_idx], y[boot_idx], sample_weight=base_w[boot_idx])
                self.trees_.append(tree)
                imp_total += tree.importances_
            self.feature_importances_ = imp_total / max(np.sum(imp_total), 1e-12)
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            all_p = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
            return np.column_stack([1.0 - all_p, all_p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    class GradientBoostingClassifier:
        """Gradient Boosted Decision Stumps/Trees for binary log-loss."""
        def __init__(self, n_estimators=80, max_depth=2, learning_rate=0.08, random_state=42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.random_state = random_state

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            n_samples, n_features = X.shape
            rng = np.random.RandomState(self.random_state)
            p_init = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
            self.init_logit_ = float(np.log(p_init / (1.0 - p_init)))
            f_val = np.full(n_samples, self.init_logit_, dtype=float)
            self.stages_ = []

            for _ in range(self.n_estimators):
                prob = 1.0 / (1.0 + np.exp(-np.clip(f_val, -20.0, 20.0)))
                residual = y - prob
                # Fit a shallow regression split on residuals
                best_gain, best_f, best_thr, best_l_val, best_r_val = 0.0, 0, 0.0, 0.0, 0.0
                base_var = float(np.sum(residual ** 2))
                feat_sub = rng.choice(n_features, size=max(1, int(np.ceil(n_features * 0.7))), replace=False)
                for f_idx in feat_sub:
                    col = X[:, f_idx]
                    uniq = np.unique(col)
                    if len(uniq) <= 1:
                        continue
                    thrs = np.quantile(uniq, np.linspace(0.15, 0.85, min(7, len(uniq) - 1)))
                    for thr in np.unique(thrs):
                        l_m = col <= thr
                        r_m = ~l_m
                        if not np.any(l_m) or not np.any(r_m):
                            continue
                        l_mean = float(np.mean(residual[l_m]))
                        r_mean = float(np.mean(residual[r_m]))
                        sse = float(np.sum((residual[l_m] - l_mean) ** 2) + np.sum((residual[r_m] - r_mean) ** 2))
                        gain = base_var - sse
                        if gain > best_gain:
                            # Newton step adjustment for log-loss
                            denom_l = float(np.sum(prob[l_m] * (1.0 - prob[l_m]))) + 1e-4
                            denom_r = float(np.sum(prob[r_m] * (1.0 - prob[r_m]))) + 1e-4
                            best_gain = gain
                            best_f = int(f_idx)
                            best_thr = float(thr)
                            best_l_val = float(np.sum(residual[l_m]) / denom_l)
                            best_r_val = float(np.sum(residual[r_m]) / denom_r)
                l_m = X[:, best_f] <= best_thr
                f_val += self.learning_rate * np.where(l_m, best_l_val, best_r_val)
                self.stages_.append((best_f, best_thr, best_l_val, best_r_val))
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            f_val = np.full(len(X), self.init_logit_, dtype=float)
            for f_idx, thr, l_v, r_v in self.stages_:
                f_val += self.learning_rate * np.where(X[:, f_idx] <= thr, l_v, r_v)
            p1 = 1.0 / (1.0 + np.exp(-np.clip(f_val, -20.0, 20.0)))
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

module_name = "career_matcher"

# Import career_matcher silently while preserving its logging and initialized artifacts
with contextlib.redirect_stdout(io.StringIO()):
    cm = importlib.import_module(module_name)

random_seed = cm.random_seed
bootstrap_rounds = 30
role_test_repeats = 10
skill_keep_chance = 0.80
distractor_chance = 0.50

# Expert domain weights for the composite readiness benchmark (sum = 1.00)
benchmark_weights = {
    "skill_count": 0.30,
    "coding_questions_solved": 0.25,
    "projects_count": 0.20,
    "experience_months": 0.15,
    "certifications_count": 0.10,
}

OUTPUT_ROOT = cm.OUTPUT_ROOT
OUTPUT_EVAL_DIR = cm.OUTPUT_EVAL_DIR
OUTPUT_CLUSTER_DIR = cm.OUTPUT_CLUSTER_DIR
OUTPUT_REPORT_DIR = cm.OUTPUT_REPORT_DIR
OUTPUT_TABLE_DIR = cm.OUTPUT_TABLE_DIR
OUTPUT_LOG_DIR = cm.OUTPUT_LOG_DIR

EVAL_REPORT_PATH = os.path.join(OUTPUT_REPORT_DIR, "evaluation_report.txt")
MODEL_METRICS_CSV_PATH = os.path.join(OUTPUT_REPORT_DIR, "model_metrics.csv")


# ==============================================================================
# HELPER FUNCTIONS (BOXED PRINTING, NMI, METRIC COLLECTOR)
# ==============================================================================
class EvaluationRecorder:
    """Collects console text for evaluation_report.txt and structured rows for model_metrics.csv."""
    def __init__(self):
        self.report_lines = []
        self.metric_rows = []

    def log_line(self, text=""):
        print(text)
        self.report_lines.append(str(text))

    def add_metric(self, category, component, metric_name, value, unit="", notes=""):
        self.metric_rows.append({
            "evaluation_category": category,
            "component": component,
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "calculation_and_source_notes": notes,
        })


def print_boxed_df(df, title=None, recorder=None):
    """Prints a DataFrame inside a clean Unicode box and records it for evaluation_report.txt."""
    out_fn = recorder.log_line if recorder is not None else print
    if title:
        out_fn(title)
    if df is None or len(df) == 0:
        out_fn("  (no rows)")
        return
    text = df.astype(str).copy()
    widths = {c: max(len(str(c)), int(text[c].str.len().max())) for c in text.columns}
    cols = list(text.columns)
    out_fn("┌─" + "─┬─".join("─" * widths[c] for c in cols) + "─┐")
    out_fn("│ " + " │ ".join(str(c).ljust(widths[c]) for c in cols) + " │")
    out_fn("├─" + "─┼─".join("─" * widths[c] for c in cols) + "─┤")
    for _, row in text.iterrows():
        out_fn("│ " + " │ ".join(row[c].ljust(widths[c]) for c in cols) + " │")
    out_fn("└─" + "─┴─".join("─" * widths[c] for c in cols) + "─┘")


def normalized_mutual_info(labels_true, labels_pred):
    """Computes Normalized Mutual Information NMI(U, V) = 2*I(U;V) / (H(U)+H(V))."""
    u, u_idx = np.unique(labels_true, return_inverse=True)
    v, v_idx = np.unique(labels_pred, return_inverse=True)
    n = len(labels_true)
    if n == 0:
        return 0.0
    contingency = np.zeros((len(u), len(v)), dtype=float)
    for i, j in zip(u_idx, v_idx):
        contingency[i, j] += 1.0
    p_uv = contingency / n
    p_u = p_uv.sum(axis=1, keepdims=True)
    p_v = p_uv.sum(axis=0, keepdims=True)
    nz = p_uv > 0
    mi = float(np.sum(p_uv[nz] * np.log(p_uv[nz] / (p_u @ p_v)[nz])))
    h_u = -float(np.sum(p_u[p_u > 0] * np.log(p_u[p_u > 0])))
    h_v = -float(np.sum(p_v[p_v > 0] * np.log(p_v[p_v > 0])))
    return (2.0 * mi / (h_u + h_v)) if (h_u + h_v) > 0 else 1.0


def compute_classification_report_df(mat, class_labels):
    """Computes per-class Precision, Recall, Specificity, F1-Score, and Support from an N x N matrix."""
    total = mat.sum()
    rows = []
    precisions, recalls, specs, f1s, supports = [], [], [], [], []
    for i, label in enumerate(class_labels):
        tp = float(mat[i, i])
        fp = float(mat[:, i].sum() - tp)
        fn = float(mat[i, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        supp = int(mat[i, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        specs.append(spec)
        f1s.append(f1)
        supports.append(supp)
        rows.append([
            label, f"{prec * 100:.1f}%", f"{rec * 100:.1f}%",
            f"{spec * 100:.1f}%", f"{f1 * 100:.1f}%", supp,
        ])

    supp_arr = np.array(supports, dtype=float)
    weights = supp_arr / max(supp_arr.sum(), 1.0)
    rows.append([
        "macro avg",
        f"{np.mean(precisions) * 100:.1f}%",
        f"{np.mean(recalls) * 100:.1f}%",
        f"{np.mean(specs) * 100:.1f}%",
        f"{np.mean(f1s) * 100:.1f}%",
        int(supp_arr.sum()),
    ])
    rows.append([
        "weighted avg",
        f"{np.sum(precisions * weights) * 100:.1f}%",
        f"{np.sum(recalls * weights) * 100:.1f}%",
        f"{np.sum(specs * weights) * 100:.1f}%",
        f"{np.sum(f1s * weights) * 100:.1f}%",
        int(supp_arr.sum()),
    ])
    return pd.DataFrame(rows, columns=[
        "class / tier", "precision", "recall (sens.)", "specificity", "f1_score", "support",
    ])


def rank_labels(model, scaler, data, feat_cols):
    """Projects rows into standardized K-Means clusters ordered by ascending mean centroid strength."""
    scaled = scaler.transform(data[feat_cols].values)
    strength = model.cluster_centers_.mean(axis=1)
    order = np.argsort(strength)
    to_rank = {int(cid): rank for rank, cid in enumerate(order)}
    return np.array([to_rank[int(c)] for c in model.predict(scaled)])


# ==============================================================================
# CHECK 1: UNSUPERVISED CLUSTERING EVALUATION & K-ABLATION (PHASE 4A)
# ==============================================================================
def evaluate_clustering_and_ablation(recorder):
    recorder.log_line("=" * 84)
    recorder.log_line("CHECK 1: UNSUPERVISED CLUSTERING VALIDITY & K-ABLATION [UNSUPERVISED_CLUSTERING]")
    recorder.log_line("=" * 84)

    train = cm.train_students
    features = cm.skill_feature_columns
    bands = ["low", "mid", "top"]

    rows = []
    for band in bands:
        trained = cm.band_models.get(band)
        part = train[train["cgpa_band"] == band]
        if trained is None or trained.get("model") is None or len(part) <= cm.skill_subclusters:
            rows.append([band, len(part), "-", "-", "-"])
            continue
        scaled = trained["scaler"].transform(part[features].values)
        labels = trained["model"].predict(scaled)
        sil = float(silhouette_score(scaled, labels))
        dbi = float(davies_bouldin_score(scaled, labels))
        chi = float(calinski_harabasz_score(scaled, labels))
        rows.append([band, len(part), f"{sil:.3f}", f"{dbi:.3f}", f"{chi:.1f}"])

        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Student_KMeans_{band}_CGPA", "Silhouette_Score", round(sil, 4), "[-1, 1]", "K=3 within CGPA band on standardized training features")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Student_KMeans_{band}_CGPA", "Davies_Bouldin_Index", round(dbi, 4), "[0, inf)", "Lower is better; K=3 within CGPA band")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Student_KMeans_{band}_CGPA", "Calinski_Harabasz_Index", round(chi, 2), "[0, inf)", "Higher is better; K=3 within CGPA band")

    sep_df = pd.DataFrame(rows, columns=[
        "cgpa_band", "students", "silhouette (↑)", "davies_bouldin (↓)", "calinski_harabasz (↑)",
    ])
    print_boxed_df(sep_df, "\nTable 1A: Student Skill Sub-Cluster Validity Inside Each CGPA Band (Design K = 3)", recorder)

    job_sil = float(silhouette_score(cm.job_scaled, cm.jobs["job_cluster"]))
    job_dbi = float(davies_bouldin_score(cm.job_scaled, cm.jobs["job_cluster"]))
    job_chi = float(calinski_harabasz_score(cm.job_scaled, cm.jobs["job_cluster"]))
    recorder.log_line(
        f"\nJob-Tier Clustering (K = 3, Salary Weight = 3.0): "
        f"Silhouette = {job_sil:.3f}, Davies-Bouldin = {job_dbi:.3f}, Calinski-Harabasz = {job_chi:.1f}"
    )
    recorder.add_metric("UNSUPERVISED_CLUSTERING", "Job_Tier_KMeans_K3", "Silhouette_Score", round(job_sil, 4), "[-1, 1]", "Salary-weighted K-Means (K=3) on brands_data.csv")
    recorder.add_metric("UNSUPERVISED_CLUSTERING", "Job_Tier_KMeans_K3", "Davies_Bouldin_Index", round(job_dbi, 4), "[0, inf)", "Salary-weighted K-Means (K=3) on brands_data.csv")
    recorder.add_metric("UNSUPERVISED_CLUSTERING", "Job_Tier_KMeans_K3", "Calinski_Harabasz_Index", round(job_chi, 2), "[0, inf)", "Salary-weighted K-Means (K=3) on brands_data.csv")

    k_values = [2, 3, 4, 5]
    k_rows = []
    ablation_records = []
    for k_val in k_values:
        if len(cm.jobs) > k_val:
            jm = KMeans(n_clusters=k_val, n_init=10, random_state=random_seed).fit(cm.job_scaled)
            j_s = float(silhouette_score(cm.job_scaled, jm.labels_))
            j_d = float(davies_bouldin_score(cm.job_scaled, jm.labels_))
            j_c = float(calinski_harabasz_score(cm.job_scaled, jm.labels_))
        else:
            j_s, j_d, j_c = np.nan, np.nan, np.nan

        s_sils, s_dbis, s_chis = [], [], []
        band_sils = {}
        for band in bands:
            part = train[train["cgpa_band"] == band]
            if len(part) > k_val:
                sc = StandardScaler().fit_transform(part[features].values)
                sm = KMeans(n_clusters=k_val, n_init=10, random_state=random_seed).fit(sc)
                bs = float(silhouette_score(sc, sm.labels_))
                bd = float(davies_bouldin_score(sc, sm.labels_))
                bc = float(calinski_harabasz_score(sc, sm.labels_))
                s_sils.append(bs)
                s_dbis.append(bd)
                s_chis.append(bc)
                band_sils[band] = bs
            else:
                band_sils[band] = np.nan

        mean_s_sil = float(np.mean(s_sils)) if s_sils else np.nan
        mean_s_dbi = float(np.mean(s_dbis)) if s_dbis else np.nan
        mean_s_chi = float(np.mean(s_chis)) if s_chis else np.nan

        k_rows.append([
            f"K = {k_val}" + (" (Design Choice)" if k_val == 3 else ""),
            f"{j_s:.3f}", f"{j_d:.3f}", f"{j_c:.1f}",
            f"{mean_s_sil:.3f}", f"{mean_s_dbi:.3f}", f"{mean_s_chi:.1f}",
        ])
        ablation_records.append({
            "k": k_val,
            "job_sil": j_s, "job_dbi": j_d, "job_chi": j_c,
            "student_sil": mean_s_sil, "student_dbi": mean_s_dbi, "student_chi": mean_s_chi,
            "low_sil": band_sils.get("low", np.nan),
            "mid_sil": band_sils.get("mid", np.nan),
            "top_sil": band_sils.get("top", np.nan),
        })
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"K_Ablation_K{k_val}", "Job_Silhouette", round(j_s, 4), "[-1, 1]", f"Job clustering with K={k_val}")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"K_Ablation_K{k_val}", "Student_Mean_Silhouette", round(mean_s_sil, 4), "[-1, 1]", f"Mean student skill clustering with K={k_val}")

    k_df = pd.DataFrame(k_rows, columns=[
        "clusters (k)", "job_sil (↑)", "job_dbi (↓)", "job_chi (↑)",
        "student_sil (↑)", "student_dbi (↓)", "student_chi (↑)",
    ])
    print_boxed_df(k_df, "\nTable 1B: K-Selection Sensitivity Analysis Across Job Tiers & Student Skill Sub-Clusters", recorder)

    best_job_k = max(ablation_records, key=lambda r: r["job_sil"])["k"]
    best_stu_k = max(ablation_records, key=lambda r: r["student_sil"])["k"]
    recorder.log_line(
        f"\n[Scientific Note on K Selection]: Empirical highest Silhouette occurs at K={best_job_k} for jobs "
        f"and K={best_stu_k} for students. K=3 is retained as the intended domain-interpretable design "
        f"(Low / Mid / Top remuneration tiers and Low / Mid / Good skill bands) to preserve the 3x3 "
        f"hierarchical cohort structure from the Springer methodology."
    )

    weight_rows = []
    job_feat_df = cm.jobs[cm.job_feature_columns].copy()
    for col in cm.job_feature_columns:
        job_feat_df[col] = job_feat_df[col].fillna(job_feat_df[col].median())
    base_scaled = StandardScaler().fit_transform(job_feat_df.values)

    for w_sal in [1.0, 2.0, 3.0, 4.0, 5.0]:
        w_vec = np.ones(len(cm.job_feature_columns), dtype=float)
        w_vec[0] = w_sal
        w_scaled = base_scaled * w_vec
        jm_w = KMeans(n_clusters=3, n_init=10, random_state=random_seed).fit(w_scaled)
        sil_w = float(silhouette_score(w_scaled, jm_w.labels_))
        c_sals = cm.jobs.groupby(jm_w.labels_)["salary_avg_lpa"].agg(["min", "max", "mean"]).sort_values("mean")
        non_overlap = all(
            c_sals["max"].iloc[idx] <= c_sals["min"].iloc[idx + 1] + 1e-6
            for idx in range(len(c_sals) - 1)
        )
        weight_rows.append([
            f"w_salary = {w_sal:.1f}" + (" (Selected)" if w_sal == 3.0 else ""),
            f"{sil_w:.3f}",
            "Yes (Strict Monotonic Tiers)" if non_overlap else "Partial Overlap",
        ])
    w_df = pd.DataFrame(weight_rows, columns=["salary_feature_weight", "silhouette_score", "strict_salary_tier_separation"])
    print_boxed_df(w_df, "\nTable 1C: Job-Clustering Salary Weight Sensitivity Justification", recorder)

    return sep_df, k_df, pd.DataFrame(ablation_records)


# ==============================================================================
# CHECK 2: BOOTSTRAP RESAMPLING STABILITY (PHASE 4A)
# ==============================================================================
def evaluate_bootstrap_stability(recorder):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line(f"CHECK 2: CLUSTERING STABILITY OVER {bootstrap_rounds} BOOTSTRAP RESAMPLES [UNSUPERVISED_CLUSTERING]")
    recorder.log_line("=" * 84)

    train = cm.train_students
    features = cm.skill_feature_columns
    bands = ["low", "mid", "top"]
    rng = np.random.RandomState(random_seed)

    rows = []
    for band in bands:
        trained = cm.band_models.get(band)
        part = train[train["cgpa_band"] == band].reset_index(drop=True)
        if trained is None or trained.get("model") is None or len(part) < cm.min_students_to_cluster:
            continue
        reference = rank_labels(trained["model"], trained["scaler"], part, features)
        aris, nmis, agreements = [], [], []
        for round_number in range(bootstrap_rounds):
            sample = part.iloc[rng.randint(0, len(part), len(part))]
            scaler = StandardScaler().fit(sample[features].values)
            model = KMeans(
                n_clusters=cm.skill_subclusters, n_init=10, random_state=round_number
            ).fit(scaler.transform(sample[features].values))
            labels = rank_labels(model, scaler, part, features)
            aris.append(float(adjusted_rand_score(reference, labels)))
            nmis.append(float(normalized_mutual_info(reference, labels)))
            agreements.append(float(np.mean(labels == reference)))

        mean_ari, std_ari = float(np.mean(aris)), float(np.std(aris))
        mean_nmi, std_nmi = float(np.mean(nmis)), float(np.std(nmis))
        mean_agr, std_agr = float(np.mean(agreements)), float(np.std(agreements))

        rows.append([
            band,
            f"{mean_ari:.3f} +- {std_ari:.3f}",
            f"{mean_nmi:.3f} +- {std_nmi:.3f}",
            f"{mean_agr * 100:.1f}% +- {std_agr * 100:.1f}%",
        ])
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Bootstrap_Stability_{band}", "Adjusted_Rand_Index_Mean", round(mean_ari, 4), "[-1, 1]", f"{bootstrap_rounds} bootstrap resamples vs full training fit")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Bootstrap_Stability_{band}", "Normalized_Mutual_Info_Mean", round(mean_nmi, 4), "[0, 1]", f"{bootstrap_rounds} bootstrap resamples vs full training fit")

    stab_df = pd.DataFrame(rows, columns=[
        "cgpa_band", "adjusted_rand_index (ari)", "norm_mutual_info (nmi)", "exact_band_agreement",
    ])
    print_boxed_df(stab_df, "\nTable 2: Bootstrap Resampling Stability by CGPA Band", recorder)
    return stab_df


# ==============================================================================
# CHECK 3: HELD-OUT SKILL-BAND BENCHMARK AGREEMENT (PHASE 4A & 4F)
# ==============================================================================
def evaluate_skill_band_benchmark_agreement(recorder):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("CHECK 3: HELD-OUT 3x3 SKILL-BAND AGREEMENT [RULE_BASED_BENCHMARK_AGREEMENT]")
    recorder.log_line("=" * 84)
    recorder.log_line(
        "Note: Because K-Means is unsupervised, this check evaluates out-of-sample ordinal agreement "
        "between learned K-Means skill bands and an independent domain-weighted readiness index "
        "(fitted strictly on the 80% training split). It is NOT ground-truth placement classification."
    )

    train = cm.train_students
    features = cm.skill_feature_columns
    bands = ["low", "mid", "top"]
    skill_names = cm.skill_names

    active_weights = {f: benchmark_weights.get(f, 1.0 / len(features)) for f in features}
    w_sum = sum(active_weights.values())
    active_weights = {f: w / w_sum for f, w in active_weights.items()}

    actual_all, predicted_all, score_all = [], [], []

    for band in bands:
        part = train[train["cgpa_band"] == band].reset_index(drop=True)
        if len(part) < cm.min_students_to_cluster * 2:
            continue
        fit_part, test_part = train_test_split(part, test_size=0.2, random_state=random_seed)
        scaler = StandardScaler().fit(fit_part[features].values)
        model = KMeans(
            n_clusters=cm.skill_subclusters, n_init=10, random_state=random_seed
        ).fit(scaler.transform(fit_part[features].values))
        predicted = rank_labels(model, scaler, test_part, features)

        means = fit_part[features].mean()
        stds = fit_part[features].std().replace(0, 1)

        def readiness(data):
            z = (data[features] - means) / stds
            return sum(z[column] * weight for column, weight in active_weights.items())

        cuts = np.quantile(readiness(fit_part), [1 / 3, 2 / 3])
        test_scores = readiness(test_part)
        actual = np.digitize(test_scores, cuts)

        actual_all.extend(list(actual))
        predicted_all.extend(list(predicted))
        score_all.extend(list(test_scores))

    matrix_3x3 = confusion_matrix(actual_all, predicted_all, labels=[0, 1, 2])
    accuracy = float(np.trace(matrix_3x3) / max(matrix_3x3.sum(), 1))
    adjacent = float(np.mean(np.abs(np.array(actual_all) - np.array(predicted_all)) <= 1))
    kappa = float(cohen_kappa_score(actual_all, predicted_all, weights="quadratic"))
    ari_bench = float(adjusted_rand_score(actual_all, predicted_all))
    nmi_bench = float(normalized_mutual_info(actual_all, predicted_all))
    rho_res = spearmanr(score_all, predicted_all)
    rho = float(rho_res.correlation if hasattr(rho_res, "correlation") else rho_res[0])

    cm_table = pd.DataFrame(
        matrix_3x3,
        index=[f"benchmark {n}" for n in skill_names],
        columns=[f"model {n}" for n in skill_names],
    ).reset_index().rename(columns={"index": "benchmark \\ model"})
    print_boxed_df(cm_table, f"\nTable 3A: Held-Out 3x3 Skill-Band Agreement Matrix (n = {matrix_3x3.sum()})", recorder)

    report_3x3 = compute_classification_report_df(matrix_3x3, [f"skill: {n}" for n in skill_names])
    print_boxed_df(report_3x3, "\nTable 3B: Per-Band Precision, Recall, Specificity, and F1 vs. Readiness Benchmark", recorder)

    recorder.log_line(
        f"\nBenchmark Agreement Summary: Exact Agreement = {accuracy * 100:.1f}% | "
        f"Adjacent Agreement = {adjacent * 100:.1f}% | ARI = {ari_bench:.3f} | "
        f"NMI = {nmi_bench:.3f} | Quadratic Kappa = {kappa:.3f} | Spearman rho = {rho:.3f}"
    )

    recorder.add_metric("RULE_BASED_BENCHMARK_AGREEMENT", "HeldOut_3x3_Skill_Band", "Exact_Agreement_Accuracy", round(accuracy, 4), "[0, 1]", "20% held-out K-Means skill band vs composite readiness tertiles")
    recorder.add_metric("RULE_BASED_BENCHMARK_AGREEMENT", "HeldOut_3x3_Skill_Band", "Adjusted_Rand_Index", round(ari_bench, 4), "[-1, 1]", "ARI against domain readiness tertiles on held-out test split")
    recorder.add_metric("RULE_BASED_BENCHMARK_AGREEMENT", "HeldOut_3x3_Skill_Band", "Normalized_Mutual_Info", round(nmi_bench, 4), "[0, 1]", "NMI against domain readiness tertiles on held-out test split")
    recorder.add_metric("RULE_BASED_BENCHMARK_AGREEMENT", "HeldOut_3x3_Skill_Band", "Quadratic_Weighted_Kappa", round(kappa, 4), "[-1, 1]", "Ordinal agreement vs readiness tertiles")
    recorder.add_metric("RULE_BASED_BENCHMARK_AGREEMENT", "HeldOut_3x3_Skill_Band", "Spearman_Rank_Correlation", round(rho, 4), "[-1, 1]", "Rank correlation between predicted skill band and continuous readiness score")

    return matrix_3x3, accuracy, kappa, rho, ari_bench, nmi_bench


# ==============================================================================
# CHECK 4: GROUND-TRUTH PLACEMENT CLASSIFICATION EVALUATION (PHASE 4B & 4F)
# ==============================================================================
def evaluate_placement_classification(recorder):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("CHECK 4: SUPERVISED PLACEMENT CLASSIFICATION [GROUND_TRUTH_PLACEMENT_CLASSIFICATION]")
    recorder.log_line("=" * 84)

    train_df = cm.train_students.copy()
    valid_label_mask = train_df["placement_label"].notna()

    if valid_label_mask.sum() < 20 or train_df.loc[valid_label_mask, "placement_label"].nunique() < 2:
        recorder.log_line(
            "Independent binary placement_status labels are unavailable in student_data.csv. "
            "Skipping supervised placement classification evaluation to avoid fabricating ground-truth labels."
        )
        return None

    labeled_df = train_df[valid_label_mask].copy().reset_index(drop=True)
    y = labeled_df["placement_label"].astype(int).to_numpy()

    candidate_predictors = [
        "cgpa", "skill_count", "coding_questions_solved", "projects_count",
        "experience_months", "certifications_count", "comm_continuous",
        "github_repos", "hackathons_participated", "aptitude_score",
        "resume_score", "mock_interview_score", "backlog_history", "attendance_percentage"
    ]
    predictor_cols = [
        c for c in candidate_predictors
        if c in labeled_df.columns and labeled_df[c].notna().all() and labeled_df[c].nunique() > 1
    ]

    recorder.log_line(f"Ground-Truth Source   : Actual 'placement_status' column from {cm.train_student_file_name}")
    recorder.log_line(f"Labeled Records       : {len(labeled_df)} (Placed = {int((y == 1).sum())}, Not Placed = {int((y == 0).sum())})")
    recorder.log_line(f"Pre-Placement Features: {predictor_cols}")
    recorder.log_line("Excluded Leakage Cols : ['salary_lpa', 'observed_salary_lpa', 'offer_count', 'company_type', 'work_mode']")

    X = labeled_df[predictor_cols].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, labeled_df, test_size=0.20, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=random_seed, max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=5, random_state=random_seed, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.08, random_state=random_seed),
    }

    model_results = {}
    summary_rows = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    for name, clf in models.items():
        cv_aucs, cv_f1s = [], []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            sc_fold = StandardScaler().fit(X_train[tr_idx])
            clf.fit(sc_fold.transform(X_train[tr_idx]), y_train[tr_idx])
            val_probs = clf.predict_proba(sc_fold.transform(X_train[val_idx]))[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            cv_aucs.append(float(roc_auc_score(y_train[val_idx], val_probs)))
            cv_f1s.append(float(f1_score(y_train[val_idx], val_preds, zero_division=0)))

        clf.fit(X_train_sc, y_train)
        y_prob = clf.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        cmat = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cmat.ravel()
        acc = float((tp + tn) / max(cmat.sum(), 1))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        spec = float(tn / max(tn + fp, 1))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        f1_val = float(f1_score(y_test, y_pred, zero_division=0))
        auc_val = float(roc_auc_score(y_test, y_prob))
        ap_val = float(average_precision_score(y_test, y_prob))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)

        model_results[name] = {
            "clf": clf,
            "confusion_matrix": cmat,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "balanced_accuracy": bal_acc,
            "f1": f1_val,
            "roc_auc": auc_val,
            "pr_auc": ap_val,
            "cv_auc_mean": float(np.mean(cv_aucs)),
            "cv_auc_std": float(np.std(cv_aucs)),
            "fpr": fpr,
            "tpr": tpr,
            "pr_prec": pr_prec,
            "pr_rec": pr_rec,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

        summary_rows.append([
            name,
            f"{acc * 100:.1f}%",
            f"{bal_acc * 100:.1f}%",
            f"{prec * 100:.1f}%",
            f"{rec * 100:.1f}%",
            f"{spec * 100:.1f}%",
            f"{f1_val * 100:.1f}%",
            f"{auc_val:.3f}",
            f"{np.mean(cv_aucs):.3f} +- {np.std(cv_aucs):.3f}",
        ])

        comp_tag = f"Placement_{name.replace(' ', '_')}"
        for m_label, m_val in [
            ("Accuracy", acc), ("Balanced_Accuracy", bal_acc), ("Precision", prec),
            ("Recall", rec), ("Specificity", spec), ("F1_Score", f1_val),
            ("ROC_AUC", auc_val), ("PR_AUC", ap_val), ("CV5_ROC_AUC_Mean", float(np.mean(cv_aucs)))
        ]:
            recorder.add_metric("GROUND_TRUTH_PLACEMENT_CLASSIFICATION", comp_tag, m_label, round(m_val, 4), "[0, 1]", "20% stratified held-out test evaluation on actual placement_status")

    sub_models = cm.fit_hierarchical_student_models(df_train, cm.skill_feature_columns)
    hier_scores = []
    hier_preds = []
    for _, row in df_test.iterrows():
        b = row["cgpa_band"]
        b_rank = cm.cgpa_band_rank[b]
        tm = sub_models.get(b)
        if tm is not None and tm.get("model") is not None:
            sc_vec = tm["scaler"].transform([[float(row[f]) for f in cm.skill_feature_columns]])
            cid = int(tm["model"].predict(sc_vec)[0])
            s_rank = tm["cluster_to_rank"][cid]
        else:
            s_rank = 1
        comb = b_rank + s_rank
        hier_scores.append(comb / 4.0)
        hier_preds.append(1 if comb >= 2 else 0)

    hier_scores = np.array(hier_scores, dtype=float)
    hier_preds = np.array(hier_preds, dtype=int)
    cmat_h = confusion_matrix(y_test, hier_preds, labels=[0, 1])
    tn_h, fp_h, fn_h, tp_h = cmat_h.ravel()
    acc_h = float((tp_h + tn_h) / max(cmat_h.sum(), 1))
    prec_h = float(precision_score(y_test, hier_preds, zero_division=0))
    rec_h = float(recall_score(y_test, hier_preds, zero_division=0))
    spec_h = float(tn_h / max(tn_h + fp_h, 1))
    bal_h = float(balanced_accuracy_score(y_test, hier_preds))
    f1_h = float(f1_score(y_test, hier_preds, zero_division=0))
    auc_h = float(roc_auc_score(y_test, hier_scores))
    ap_h = float(average_precision_score(y_test, hier_scores))
    fpr_h, tpr_h, _ = roc_curve(y_test, hier_scores)
    pr_prec_h, pr_rec_h, _ = precision_recall_curve(y_test, hier_scores)

    model_results["Unsupervised Hierarchy Tier"] = {
        "clf": None,
        "confusion_matrix": cmat_h,
        "accuracy": acc_h,
        "precision": prec_h,
        "recall": rec_h,
        "specificity": spec_h,
        "balanced_accuracy": bal_h,
        "f1": f1_h,
        "roc_auc": auc_h,
        "pr_auc": ap_h,
        "cv_auc_mean": auc_h,
        "cv_auc_std": 0.0,
        "fpr": fpr_h,
        "tpr": tpr_h,
        "pr_prec": pr_prec_h,
        "pr_rec": pr_rec_h,
        "y_prob": hier_scores,
        "y_pred": hier_preds,
    }
    summary_rows.append([
        "Unsupervised Hierarchy Tier (Comb >= 2)",
        f"{acc_h * 100:.1f}%",
        f"{bal_h * 100:.1f}%",
        f"{prec_h * 100:.1f}%",
        f"{rec_h * 100:.1f}%",
        f"{spec_h * 100:.1f}%",
        f"{f1_h * 100:.1f}%",
        f"{auc_h:.3f}",
        f"{auc_h:.3f} (Unsup.)",
    ])

    models_df = pd.DataFrame(summary_rows, columns=[
        "model", "accuracy", "balanced_acc", "precision", "recall",
        "specificity", "f1_score", "test_roc_auc", "5fold_cv_roc_auc",
    ])
    print_boxed_df(models_df, f"\nTable 4A: Held-Out Placement Classification Comparison (n_test = {len(y_test)})", recorder)

    rf_mat = model_results["Random Forest"]["confusion_matrix"]
    cm_2x2_df = pd.DataFrame(
        rf_mat,
        index=["Actual 0 (Not Placed)", "Actual 1 (Placed)"],
        columns=["Predicted 0 (Not Placed)", "Predicted 1 (Placed)"],
    ).reset_index().rename(columns={"index": "Ground Truth \\ Predicted"})
    print_boxed_df(
        cm_2x2_df,
        f"\nTable 4B: Held-Out 2x2 Placement Confusion Matrix (Random Forest; Acc = {model_results['Random Forest']['accuracy']*100:.1f}%, AUC = {model_results['Random Forest']['roc_auc']:.3f})",
        recorder,
    )

    rf_clf = model_results["Random Forest"]["clf"]
    importances = rf_clf.feature_importances_
    imp_order = np.argsort(importances)[::-1]
    imp_df = pd.DataFrame([
        [predictor_cols[i], f"{importances[i] * 100:.2f}%"]
        for i in imp_order
    ], columns=["pre_placement_feature", "rf_gini_importance"])
    print_boxed_df(imp_df, "\nTable 4C: Supervised Placement Feature Importance (Random Forest, Pre-Placement Features Only)", recorder)

    if cm.checked_students["placement_label"].notna().sum() == 0:
        recorder.log_line(
            f"\n[External Testing Dataset Note]: {cm.check_student_file_name} does not contain ground-truth "
            f"'placement_status' labels in this batch; therefore, external placement classification accuracy "
            f"is intentionally not calculated on {cm.check_student_file_name} (Phase 4B compliance)."
        )

    return {
        "model_results": model_results,
        "models_df": models_df,
        "predictor_cols": predictor_cols,
        "importances": importances,
        "y_test": y_test,
        "df_test": df_test,
    }


# ==============================================================================
# CHECK 5: RELATIVE CLUSTER-MARGIN SCORE ANALYSIS (PHASE 4C)
# ==============================================================================
def evaluate_cluster_margin_score(recorder, placement_eval_res=None):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("CHECK 5: RELATIVE CLUSTER-MARGIN SCORE DISTRIBUTION & CALIBRATION [UNSUPERVISED_CLUSTERING]")
    recorder.log_line("=" * 84)
    recorder.log_line(
        "Mathematical Formula: M(x) = 1 - (d_1(x) / d_2(x)), where d_1(x) <= d_2(x) are Euclidean distances "
        "to the nearest and second-nearest K-Means centroids in standardized feature space.\n"
        "Edge-Case Handling  : Returns 0.0 when d_2(x) == 0 or d_1(x) == d_2(x); returns 1.0 when d_1(x) == 0 < d_2(x).\n"
        "Interpretation      : Geometric relative cluster-boundary margin in [0, 1] (NOT a probability)."
    )

    rows = []
    cohort_margins = {}
    for name, data in [("Training Students", cm.train_students), ("Testing Students", cm.checked_students)]:
        values = data["margin_score"].dropna().to_numpy(dtype=float)
        cohort_margins[name] = values
        if len(values) == 0:
            continue
        q25, q75 = np.percentile(values, [25, 75])
        rows.append([
            name,
            len(values),
            f"{values.mean():.3f}",
            f"{values.std():.3f}",
            f"{np.median(values):.3f}",
            f"{q75 - q25:.3f}",
            f"{np.mean(values >= 0.50) * 100:.1f}%",
            f"{np.mean(values < 0.20) * 100:.1f}%",
        ])
        tag = name.replace(" ", "_")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Relative_Margin_{tag}", "Mean_Margin_Score", round(float(values.mean()), 4), "[0, 1]", "M(x) = 1 - d1/d2")
        recorder.add_metric("UNSUPERVISED_CLUSTERING", f"Relative_Margin_{tag}", "Median_Margin_Score", round(float(np.median(values)), 4), "[0, 1]", "M(x) = 1 - d1/d2")

    conf_df = pd.DataFrame(rows, columns=[
        "cohort", "count", "mean_margin", "std_margin", "median_margin",
        "iqr_margin", "strong_margin (>=0.50)", "borderline (<0.20)",
    ])
    print_boxed_df(conf_df, "\nTable 5A: Relative Cluster-Margin Score Distribution", recorder)

    train = cm.train_students
    margins = train["margin_score"].to_numpy(dtype=float)
    cal_rows = []
    for tier_label, mask in [
        ("Borderline Margin (< 0.20)", margins < 0.20),
        ("Moderate Margin (0.20 to < 0.50)", (margins >= 0.20) & (margins < 0.50)),
        ("Strong Interior Margin (>= 0.50)", margins >= 0.50),
    ]:
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sub = train[mask]
        pl_rate = (
            f"{sub['placement_label'].mean() * 100:.1f}%"
            if sub["placement_label"].notna().any() else "-"
        )
        cal_rows.append([
            tier_label,
            cnt,
            f"{cnt / len(train) * 100:.1f}%",
            f"{sub['margin_score'].mean():.3f}",
            pl_rate,
        ])
    cal_df = pd.DataFrame(cal_rows, columns=[
        "margin_stratum", "students", "share_of_cohort", "mean_margin_score", "observed_placement_rate",
    ])
    print_boxed_df(cal_df, "\nTable 5B: Stratification by Relative Cluster-Margin Score (Training Set)", recorder)

    return conf_df, cohort_margins


# ==============================================================================
# CHECK 6: JOB RECOMMENDATION EVALUATION — CONTROLLED ROLE-RECOVERY TEST (PHASE 4D)
# ==============================================================================
def evaluate_job_recommendation_robustness(recorder):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("CHECK 6: CONTROLLED SYNTHETIC ROLE-RECOVERY ROBUSTNESS TEST [CONTROLLED_SYNTHETIC_EXPERIMENT]")
    recorder.log_line("=" * 84)
    recorder.log_line(
        "Scientific Disclosure: Because historical student-to-job click/application relevance logs are "
        "not present in the datasets, this experiment is a CONTROLLED SYNTHETIC ROBUSTNESS TEST measuring "
        "whether the skill-matching function recovers a target role under 20% random required-skill "
        "dropout and 50% distractor skill injection."
    )

    jobs = cm.jobs
    all_roles = jobs["job_role"].nunique()
    all_skills = sorted(cm.known_skills - cm.soft_skills)
    if not all_skills:
        all_skills = ["python", "java", "sql", "javascript"]

    methods = {
        "Proposed Weighted Matcher (0.8 Req + 0.2 Pref)": {"req_only": False},
        "Required-Skills-Only Matcher (1.0 Req)": {"req_only": True},
    }

    results_by_method = {}
    for m_name, cfg in methods.items():
        hits = {1: [], 3: [], 5: []}
        mrrs = []
        rng = np.random.RandomState(random_seed)
        for _ in range(role_test_repeats):
            repeat_hits = {1: 0, 3: 0, 5: 0}
            reciprocal_ranks = []
            for _, job in jobs.iterrows():
                required = [s for s in job["required_list"] if s not in cm.soft_skills]
                kept = [s for s in required if rng.rand() < skill_keep_chance] or required[:1]
                if rng.rand() < distractor_chance:
                    kept = kept + [str(rng.choice(all_skills))]
                deg = job["degree_list"][0] if len(job["degree_list"]) > 0 else "btech"
                candidate = pd.DataFrame([{
                    "cgpa": 9.5, "experience_months": 24.0, "comm_score": 3,
                    "degree": deg, "skill_list": sorted(set(kept)),
                }])
                scored = cm.score_jobs_for_group(candidate, required_only=cfg["req_only"])
                ranked = scored.sort_values(["skill_match", "salary_avg_lpa"], ascending=False)["job_role"]
                ordered_roles = list(dict.fromkeys(ranked))
                for k in (1, 3, 5):
                    repeat_hits[k] += int(job["job_role"] in ordered_roles[:k])
                if job["job_role"] in ordered_roles:
                    rank_pos = ordered_roles.index(job["job_role"]) + 1
                    reciprocal_ranks.append(1.0 / rank_pos)
                else:
                    reciprocal_ranks.append(0.0)
            for k in (1, 3, 5):
                hits[k].append(repeat_hits[k] / len(jobs))
            mrrs.append(float(np.mean(reciprocal_ranks)))
        results_by_method[m_name] = {"hits": hits, "mrrs": mrrs}

    rand_hits = {k: min(k / max(all_roles, 1), 1.0) for k in (1, 3, 5)}
    rand_mrr = sum(1.0 / r for r in range(1, all_roles + 1)) / max(all_roles, 1)

    rows = []
    prop_res = results_by_method["Proposed Weighted Matcher (0.8 Req + 0.2 Pref)"]
    req_res = results_by_method["Required-Skills-Only Matcher (1.0 Req)"]

    for k in (1, 3, 5):
        p_mean, p_std = float(np.mean(prop_res["hits"][k])), float(np.std(prop_res["hits"][k]))
        r_mean, r_std = float(np.mean(req_res["hits"][k])), float(np.std(req_res["hits"][k]))
        rb = rand_hits[k]
        rows.append([
            f"Hit-Rate@{k}",
            f"{p_mean * 100:.1f}% +- {p_std * 100:.1f}%",
            f"{r_mean * 100:.1f}% +- {r_std * 100:.1f}%",
            f"{rb * 100:.1f}%",
            f"{(p_mean / max(rb, 1e-6)):.2f}x",
        ])
        recorder.add_metric("CONTROLLED_SYNTHETIC_EXPERIMENT", "Role_Recovery_Proposed", f"Hit_Rate_at_{k}", round(p_mean, 4), "[0, 1]", f"Controlled 20% skill dropout + distractor test ({role_test_repeats} repeats)")
        recorder.add_metric("CONTROLLED_SYNTHETIC_EXPERIMENT", "Role_Recovery_RequiredOnly", f"Hit_Rate_at_{k}", round(r_mean, 4), "[0, 1]", f"Required-skills-only matcher ({role_test_repeats} repeats)")

    p_mrr, p_mrr_std = float(np.mean(prop_res["mrrs"])), float(np.std(prop_res["mrrs"]))
    r_mrr, r_mrr_std = float(np.mean(req_res["mrrs"])), float(np.std(req_res["mrrs"]))
    rows.append([
        "Mean Reciprocal Rank (MRR)",
        f"{p_mrr:.3f} +- {p_mrr_std:.3f}",
        f"{r_mrr:.3f} +- {r_mrr_std:.3f}",
        f"{rand_mrr:.3f}",
        f"{(p_mrr / max(rand_mrr, 1e-6)):.2f}x",
    ])
    recorder.add_metric("CONTROLLED_SYNTHETIC_EXPERIMENT", "Role_Recovery_Proposed", "MRR", round(p_mrr, 4), "[0, 1]", "Controlled synthetic role-recovery Mean Reciprocal Rank")
    recorder.add_metric("CONTROLLED_SYNTHETIC_EXPERIMENT", "Role_Recovery_RequiredOnly", "MRR", round(r_mrr, 4), "[0, 1]", "Required-only matcher Mean Reciprocal Rank")

    rec_df = pd.DataFrame(rows, columns=[
        "metric", "proposed_weighted_matcher", "required_only_ablation", "random_baseline", "lift_over_random"
    ])
    print_boxed_df(rec_df, f"\nTable 6: Controlled Role-Recovery Robustness ({len(jobs)} jobs, {all_roles} distinct roles, {role_test_repeats} repeats)", recorder)

    return results_by_method, rand_hits, rand_mrr, rec_df


# ==============================================================================
# CHECK 7: CAPACITY-CONSTRAINED ALLOCATION EVALUATION (PHASE 4E)
# ==============================================================================
def evaluate_allocation_methods(recorder):
    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("CHECK 7: CAMPUS ALLOCATION EVALUATION [CAPACITY_CONSTRAINED_ALLOCATION]")
    recorder.log_line("=" * 84)
    recorder.log_line(
        "All methods are evaluated on the EXACT SAME student testing batch, eligibility matrix, "
        "offer quota (j_offer), and company slot capacities by invoking cm.student_company_mapping()."
    )

    students = cm.checked_students
    j_offer = cm.default_j_offer

    _, company_rows, _, qual_matrix, _, shared_capacities = cm.build_eligibility_and_utility_matrices(
        students, j_offer=j_offer
    )

    method_specs = [
        ("First-Come-First-Served (FCFS) Baseline", "fcfs"),
        ("Random Eligible Assignment Baseline", "random"),
        ("Greedy Constrained-First Heuristic (Fixed Capacity)", "greedy_heuristic"),
        ("Proposed Optimal Capacity-Constrained MILP/Flow", "optimal"),
    ]

    alloc_rows = []
    method_metrics = {}
    for label, m_code in method_specs:
        _, _, _, _, metrics = cm.student_company_mapping(
            students, j_offer=j_offer, capacities=shared_capacities, method=m_code
        )
        method_metrics[label] = metrics
        alloc_rows.append([
            label,
            f"{metrics['placed_students']}/{metrics['eligible_students']}",
            f"{metrics['placement_coverage_all_pct']:.1f}% ({metrics['placement_coverage_eligible_pct']:.1f}% elig)",
            f"{metrics['mean_assigned_salary_lpa']:.2f} LPA",
            f"{metrics['total_salary_objective_lpa']:.1f} LPA",
            metrics["capacity_violations"],
            metrics["unassigned_students"],
            metrics["unmet_quota_students"],
        ])

        tag = f"Allocation_{m_code}"
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Placed_Students", metrics["placed_students"], "count", f"Out of {metrics['eligible_students']} eligible and {metrics['total_students']} total students")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Placement_Coverage_All_Pct", round(metrics["placement_coverage_all_pct"], 2), "%", "Percentage of all tested students receiving >= 1 offer")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Mean_Assigned_Salary_LPA", round(metrics["mean_assigned_salary_lpa"], 4), "LPA", "Average salary across all assigned offers")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Total_Salary_Objective_LPA", round(metrics["total_salary_objective_lpa"], 2), "LPA", "Sum of salaries across all assigned offers")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Capacity_Violations", metrics["capacity_violations"], "slots", "Number of assigned slots exceeding company capacity")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Unassigned_Students", metrics["unassigned_students"], "count", "Students receiving 0 offers")
        recorder.add_metric("CAPACITY_CONSTRAINED_ALLOCATION", tag, "Unmet_Quota_Students", metrics["unmet_quota_students"], "count", f"Eligible students receiving < {j_offer} offers")

    opt_df = pd.DataFrame(alloc_rows, columns=[
        "allocation_method", "eligible_placed", "placement_coverage",
        "avg_assigned_salary", "total_salary_obj", "cap_violations",
        "unassigned", "unmet_quotas",
    ])
    print_boxed_df(
        opt_df,
        f"\nTable 7: Capacity-Constrained Allocation Comparison (n = {len(students)}, j_offer = {j_offer}, Total Company Slots = {int(np.sum(shared_capacities))})",
        recorder,
    )
    return opt_df, method_metrics


# ==============================================================================
# MODULAR INDIVIDUAL 300-DPI FIGURE GENERATORS (PHASE 5 & PHASE 6)
# ==============================================================================
def generate_all_individual_evaluation_figures(
    ablation_df,
    skill_cm_3x3,
    skill_acc,
    skill_kappa,
    skill_rho,
    placement_eval,
    cohort_margins,
    role_rec_results,
    rand_hits,
    rand_mrr,
    alloc_metrics,
):
    """
    Generates each evaluation figure as a standalone, 300-DPI publication-ready
    PNG file inside output/evaluation/ (never combining panels into a dashboard).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    cm.ensure_output_directories()
    saved_files = []
    ks = ablation_df["k"].to_numpy(dtype=int)

    # 1. Job-Tier Silhouette Comparison (job_tier_silhouette_comparison.png)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(ks, ablation_df["job_sil"], marker="o", linewidth=2.4, markersize=7, color="#1d4ed8", label="Job-Tier Silhouette Score")
    ax.axvline(3, color="#dc2626", linestyle="--", linewidth=1.4, label="Selected Design (K = 3)")
    for k_v, s_v in zip(ks, ablation_df["job_sil"]):
        ax.annotate(f"{s_v:.3f}", (k_v, s_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(ablation_df["job_sil"].max() * 1.22, 0.5))
    ax.set_xlabel("Number of Job Tiers (K)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Silhouette Score (Higher is Better ↑)", fontweight="bold", fontsize=10)
    ax.set_title("Job-Tier Clustering Silhouette Score Across K", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p1 = os.path.join(OUTPUT_EVAL_DIR, "job_tier_silhouette_comparison.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_files.append(p1)

    # 2. Student-Clustering Silhouette Comparison (student_silhouette_comparison.png)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ks, ablation_df["low_sil"], marker="^", linestyle="--", linewidth=1.6, color="#dc2626", label="Low CGPA Band (< 6.0)")
    ax.plot(ks, ablation_df["mid_sil"], marker="s", linestyle="--", linewidth=1.6, color="#d97706", label="Mid CGPA Band (6.0–8.0)")
    ax.plot(ks, ablation_df["top_sil"], marker="d", linestyle="--", linewidth=1.6, color="#0284c7", label="Top CGPA Band (>= 8.0)")
    ax.plot(ks, ablation_df["student_sil"], marker="o", linewidth=2.6, color="#0f766e", label="Mean Student Silhouette")
    ax.axvline(3, color="#1e293b", linestyle=":", linewidth=1.4, label="Selected Design (K = 3)")
    for k_v, s_v in zip(ks, ablation_df["student_sil"]):
        ax.annotate(f"{s_v:.3f}", (k_v, s_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#0f766e")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(ablation_df["student_sil"].max() * 1.25, 0.5))
    ax.set_xlabel("Number of Skill Sub-Clusters (K)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Silhouette Score (Higher is Better ↑)", fontweight="bold", fontsize=10)
    ax.set_title("Student Skill-Clustering Silhouette Comparison Across K", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p2 = os.path.join(OUTPUT_EVAL_DIR, "student_silhouette_comparison.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_files.append(p2)

    # 3. Combined Silhouette Comparison (silhouette_comparison.png)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ks, ablation_df["job_sil"], marker="o", linewidth=2.3, markersize=7, color="#1d4ed8", label="Job-Tier Clustering Silhouette")
    ax.plot(ks, ablation_df["student_sil"], marker="s", linewidth=2.3, markersize=7, color="#0f766e", label="Student Skill-Clustering Mean Silhouette")
    ax.axvline(3, color="#dc2626", linestyle="--", linewidth=1.3, label="Intended Hierarchy Design (K = 3)")
    for k_v, j_v, s_v in zip(ks, ablation_df["job_sil"], ablation_df["student_sil"]):
        ax.annotate(f"{j_v:.2f}", (k_v, j_v), textcoords="offset points", xytext=(-10, 7), fontsize=8, fontweight="bold", color="#1d4ed8")
        ax.annotate(f"{s_v:.2f}", (k_v, s_v), textcoords="offset points", xytext=(6, -12), fontsize=8, fontweight="bold", color="#0f766e")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(float(ablation_df["job_sil"].max()), float(ablation_df["student_sil"].max())) * 1.25)
    ax.set_xlabel("Number of Clusters (K)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Silhouette Coefficient (↑)", fontweight="bold", fontsize=10)
    ax.set_title("Silhouette Score Comparison: Job Tiers & Student Skill Clusters", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p3 = os.path.join(OUTPUT_EVAL_DIR, "silhouette_comparison.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_files.append(p3)

    # 4. Davies-Bouldin Comparison (davies_bouldin_comparison.png)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ks, ablation_df["job_dbi"], marker="o", linewidth=2.3, markersize=7, color="#b45309", label="Job-Tier Davies-Bouldin Index")
    ax.plot(ks, ablation_df["student_dbi"], marker="s", linewidth=2.3, markersize=7, color="#7c3aed", label="Student Skill Mean Davies-Bouldin Index")
    ax.axvline(3, color="#dc2626", linestyle="--", linewidth=1.3, label="Intended Hierarchy Design (K = 3)")
    for k_v, j_v, s_v in zip(ks, ablation_df["job_dbi"], ablation_df["student_dbi"]):
        ax.annotate(f"{j_v:.2f}", (k_v, j_v), textcoords="offset points", xytext=(-10, 7), fontsize=8, fontweight="bold", color="#b45309")
        ax.annotate(f"{s_v:.2f}", (k_v, s_v), textcoords="offset points", xytext=(6, -12), fontsize=8, fontweight="bold", color="#7c3aed")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of Clusters (K)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Davies-Bouldin Index (Lower is Better ↓)", fontweight="bold", fontsize=10)
    ax.set_title("Davies-Bouldin Index Comparison Across K", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p4 = os.path.join(OUTPUT_EVAL_DIR, "davies_bouldin_comparison.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_files.append(p4)

    # 5. Calinski-Harabasz Comparison (calinski_harabasz_comparison.png)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ks, ablation_df["job_chi"], marker="o", linewidth=2.3, markersize=7, color="#1d4ed8", label="Job-Tier Calinski-Harabasz Index")
    ax.plot(ks, ablation_df["student_chi"], marker="s", linewidth=2.3, markersize=7, color="#15803d", label="Student Skill Mean Calinski-Harabasz Index")
    ax.axvline(3, color="#dc2626", linestyle="--", linewidth=1.3, label="Intended Hierarchy Design (K = 3)")
    for k_v, j_v, s_v in zip(ks, ablation_df["job_chi"], ablation_df["student_chi"]):
        ax.annotate(f"{j_v:.0f}", (k_v, j_v), textcoords="offset points", xytext=(-8, 7), fontsize=8, fontweight="bold", color="#1d4ed8")
        ax.annotate(f"{s_v:.0f}", (k_v, s_v), textcoords="offset points", xytext=(6, -12), fontsize=8, fontweight="bold", color="#15803d")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of Clusters (K)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Calinski-Harabasz Index (Higher is Better ↑)", fontweight="bold", fontsize=10)
    ax.set_title("Calinski-Harabasz Index Comparison Across K", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p5 = os.path.join(OUTPUT_EVAL_DIR, "calinski_harabasz_comparison.png")
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    saved_files.append(p5)

    # 6. Student Skill-Band Confusion Matrix (confusion_matrix_skill.png)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(skill_cm_3x3, cmap="Blues")
    ax.set_xticks(range(3), [f"K-Means\n{n.capitalize()}" for n in cm.skill_names], fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(3), [f"Benchmark\n{n.capitalize()}" for n in cm.skill_names], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Out-of-Sample K-Means Skill Sub-Cluster", fontsize=10, fontweight="bold")
    ax.set_ylabel("Domain-Weighted Readiness Benchmark Tertile", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Held-Out 3x3 Student Skill-Band Agreement Matrix\n"
        f"(Exact Acc = {skill_acc * 100:.1f}%, Quadratic κ = {skill_kappa:.2f}, Spearman ρ = {skill_rho:.2f})",
        fontsize=10.5, fontweight="bold", pad=10,
    )
    for i in range(3):
        r_sum = max(skill_cm_3x3[i, :].sum(), 1)
        for j in range(3):
            val = skill_cm_3x3[i, j]
            pct = val / r_sum * 100.0
            color = "white" if val > skill_cm_3x3.max() / 2 else "#0f172a"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Student Count")
    fig.tight_layout()
    p6 = os.path.join(OUTPUT_EVAL_DIR, "confusion_matrix_skill.png")
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    saved_files.append(p6)

    # 7–11. Supervised Placement Figures
    if placement_eval is not None:
        m_res = placement_eval["model_results"]
        rf_info = m_res["Random Forest"]
        rf_mat = rf_info["confusion_matrix"]

        # 7. Placement Confusion Matrix (confusion_matrix_placement.png)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(rf_mat, cmap="Greens")
        ax.set_xticks([0, 1], ["Predicted 0\n(Not Placed)", "Predicted 1\n(Placed)"], fontsize=9.5, fontweight="bold")
        ax.set_yticks([0, 1], ["Actual 0\n(Not Placed)", "Actual 1\n(Placed)"], fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Predicted Placement Status (Held-Out Test Split)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Ground-Truth Placement Status (Dataset 2)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Held-Out Placement Classification Confusion Matrix\n"
            f"(Random Forest: Acc = {rf_info['accuracy']*100:.1f}%, Bal Acc = {rf_info['balanced_accuracy']*100:.1f}%, F1 = {rf_info['f1']*100:.1f}%)",
            fontsize=10, fontweight="bold", pad=10,
        )
        for i in range(2):
            r_sum = max(rf_mat[i, :].sum(), 1)
            for j in range(2):
                val = rf_mat[i, j]
                pct = val / r_sum * 100.0
                color = "white" if val > rf_mat.max() / 2 else "#0f172a"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10.5, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Student Count")
        fig.tight_layout()
        p7 = os.path.join(OUTPUT_EVAL_DIR, "confusion_matrix_placement.png")
        fig.savefig(p7, dpi=300)
        plt.close(fig)
        saved_files.append(p7)

        # 8. ROC Curve (roc_curve.png)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        palette = {
            "Logistic Regression": "#0284c7",
            "Random Forest": "#15803d",
            "Gradient Boosting": "#7c3aed",
            "Unsupervised Hierarchy Tier": "#d97706",
        }
        for name, info in m_res.items():
            ax.plot(
                info["fpr"], info["tpr"], linewidth=2.2,
                color=palette.get(name, "#1e293b"),
                label=f"{name} (AUC = {info['roc_auc']:.3f})"
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="Random Classifier (AUC = 0.500)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", fontsize=10)
        ax.set_title("Held-Out Placement Classification ROC Curves", fontweight="bold", fontsize=11, pad=12)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        p8 = os.path.join(OUTPUT_EVAL_DIR, "roc_curve.png")
        fig.savefig(p8, dpi=300)
        plt.close(fig)
        saved_files.append(p8)

        # 9. Precision-Recall Curve (precision_recall_curve.png)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        base_pos_rate = float(np.mean(placement_eval["y_test"]))
        for name, info in m_res.items():
            order = np.argsort(info["pr_rec"])
            ax.plot(
                info["pr_rec"][order], info["pr_prec"][order], linewidth=2.2,
                color=palette.get(name, "#1e293b"),
                label=f"{name} (AP = {info['pr_auc']:.3f})"
            )
        ax.axhline(base_pos_rate, linestyle="--", color="#64748b", linewidth=1.2, label=f"No-Skill Baseline (Pos Rate = {base_pos_rate:.2f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall (Sensitivity)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold", fontsize=10)
        ax.set_title("Held-Out Placement Precision-Recall Curves", fontweight="bold", fontsize=11, pad=12)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        p9 = os.path.join(OUTPUT_EVAL_DIR, "precision_recall_curve.png")
        fig.savefig(p9, dpi=300)
        plt.close(fig)
        saved_files.append(p9)

        # 10. Model Comparison Chart (model_comparison.png)
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Balanced\nAcc", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
        x = np.arange(len(metric_keys))
        w = 0.20
        m_names = list(m_res.keys())
        for idx_m, name in enumerate(m_names):
            vals = [m_res[name][k] * 100.0 for k in metric_keys]
            ax.bar(
                x + (idx_m - 1.5) * w, vals, w,
                label=name, color=palette.get(name, "#334155"), edgecolor="#0f172a", alpha=0.92
            )
        ax.set_xticks(x, metric_labels, fontweight="bold", fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score / Percentage (%)", fontweight="bold", fontsize=10)
        ax.set_xlabel("Evaluation Metric (Held-Out Test Split)", fontweight="bold", fontsize=10)
        ax.set_title("Placement Classification Model Comparison (Leak-Free Pre-Placement Features)", fontweight="bold", fontsize=11, pad=12)
        ax.legend(loc="upper right", fontsize=8.5, ncol=2)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        p10 = os.path.join(OUTPUT_EVAL_DIR, "model_comparison.png")
        fig.savefig(p10, dpi=300)
        plt.close(fig)
        saved_files.append(p10)

        # 11. Feature Importance Chart (feature_importance.png)
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        p_cols = placement_eval["predictor_cols"]
        imps = placement_eval["importances"] * 100.0
        order = np.argsort(imps)
        y_pos = np.arange(len(order))
        bars = ax.barh(y_pos, imps[order], color="#1d4ed8", edgecolor="#1e3a8a", height=0.62)
        ax.set_yticks(y_pos, [p_cols[i] for i in order], fontweight="bold", fontsize=9)
        ax.set_xlim(0, max(imps) * 1.22)
        ax.set_xlabel("Gini Feature Importance (%)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Pre-Placement Student Attribute", fontweight="bold", fontsize=10)
        ax.set_title("Random Forest Feature Importance for Placement Prediction", fontweight="bold", fontsize=11, pad=12)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        for bar in bars:
            w_val = bar.get_width()
            ax.text(w_val + max(imps) * 0.015, bar.get_y() + bar.get_height() / 2, f"{w_val:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")
        fig.tight_layout()
        p11 = os.path.join(OUTPUT_EVAL_DIR, "feature_importance.png")
        fig.savefig(p11, dpi=300)
        plt.close(fig)
        saved_files.append(p11)

    # 12. Relative Cluster-Margin Score Distribution (confidence_distribution.png)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    tr_m = cohort_margins.get("Training Students", np.array([]))
    ch_m = cohort_margins.get("Testing Students", np.array([]))
    bins = np.linspace(0.0, 1.0, 16)
    if len(tr_m) > 0:
        ax.hist(tr_m, bins=bins, density=True, alpha=0.60, color="#1d4ed8", edgecolor="#1e3a8a", label=f"Training Set (μ={tr_m.mean():.2f}, med={np.median(tr_m):.2f})")
    if len(ch_m) > 0:
        ax.hist(ch_m, bins=bins, density=True, alpha=0.60, color="#0f766e", edgecolor="#115e59", label=f"Testing Set (μ={ch_m.mean():.2f}, med={np.median(ch_m):.2f})")
    ax.axvline(0.20, color="#dc2626", linestyle="--", linewidth=1.4, label="Borderline Threshold (0.20)")
    ax.axvline(0.50, color="#15803d", linestyle="--", linewidth=1.4, label="Strong Interior Margin (0.50)")
    ax.set_xlabel("Relative Cluster-Margin Score M(x) = 1 - d1(x)/d2(x)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Density", fontweight="bold", fontsize=10)
    ax.set_title("Distribution of Relative Cluster-Margin Scores (Not Probabilities)", fontweight="bold", fontsize=11, pad=12)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p12 = os.path.join(OUTPUT_EVAL_DIR, "confidence_distribution.png")
    fig.savefig(p12, dpi=300)
    plt.close(fig)
    saved_files.append(p12)

    # 13. Role Hit-Rate@K Comparison (hit_rate_comparison.png)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    k_labels = ["Hit-Rate@1", "Hit-Rate@3", "Hit-Rate@5"]
    prop_hits = [np.mean(role_rec_results["Proposed Weighted Matcher (0.8 Req + 0.2 Pref)"]["hits"][k]) * 100.0 for k in (1, 3, 5)]
    req_hits = [np.mean(role_rec_results["Required-Skills-Only Matcher (1.0 Req)"]["hits"][k]) * 100.0 for k in (1, 3, 5)]
    r_base = [rand_hits[k] * 100.0 for k in (1, 3, 5)]
    x = np.arange(len(k_labels))
    w = 0.26
    b1 = ax.bar(x - w, prop_hits, w, label="Proposed Matcher (0.8 Req + 0.2 Pref)", color="#1d4ed8", edgecolor="#0f172a")
    b2 = ax.bar(x, req_hits, w, label="Required-Skills-Only Matcher", color="#0f766e", edgecolor="#0f172a")
    b3 = ax.bar(x + w, r_base, w, label="Random Ranking Baseline", color="#94a3b8", edgecolor="#0f172a")
    ax.set_xticks(x, k_labels, fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Top-K Role Recommendation Cutoff", fontweight="bold", fontsize=10)
    ax.set_ylabel("Role Recovery Hit-Rate (%)", fontweight="bold", fontsize=10)
    ax.set_title("Controlled Synthetic Role-Recovery Hit-Rate@K (20% Skill Dropout + Noise)", fontweight="bold", fontsize=10.5, pad=12)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    p13 = os.path.join(OUTPUT_EVAL_DIR, "hit_rate_comparison.png")
    fig.savefig(p13, dpi=300)
    plt.close(fig)
    saved_files.append(p13)

    # 14. MRR Comparison (mrr_comparison.png)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    mrr_labels = ["Random\nBaseline", "Required-Only\nMatcher", "Proposed Weighted\nMatcher (0.8/0.2)"]
    mrr_vals = [
        rand_mrr,
        float(np.mean(role_rec_results["Required-Skills-Only Matcher (1.0 Req)"]["mrrs"])),
        float(np.mean(role_rec_results["Proposed Weighted Matcher (0.8 Req + 0.2 Pref)"]["mrrs"])),
    ]
    mrr_errs = [
        0.0,
        float(np.std(role_rec_results["Required-Skills-Only Matcher (1.0 Req)"]["mrrs"])),
        float(np.std(role_rec_results["Proposed Weighted Matcher (0.8 Req + 0.2 Pref)"]["mrrs"])),
    ]
    bars = ax.bar(mrr_labels, mrr_vals, yerr=mrr_errs, capsize=5, color=["#94a3b8", "#0f766e", "#1d4ed8"], width=0.46, edgecolor="#0f172a")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Recommendation Ranking Method", fontweight="bold", fontsize=10)
    ax.set_ylabel("Mean Reciprocal Rank (MRR ↑)", fontweight="bold", fontsize=10)
    ax.set_title("Controlled Synthetic Role-Recovery Mean Reciprocal Rank (MRR)", fontweight="bold", fontsize=10.5, pad=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, err in zip(bars, mrr_errs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.025, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p14 = os.path.join(OUTPUT_EVAL_DIR, "mrr_comparison.png")
    fig.savefig(p14, dpi=300)
    plt.close(fig)
    saved_files.append(p14)

    # 15. Salary Optimization & Allocation Comparison (salary_optimization.png)
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    strat_keys = list(alloc_metrics.keys())
    short_labels = ["FCFS\nBaseline", "Random\nEligible", "Greedy\nHeuristic", "Proposed Optimal\n(MILP / Flow)"]
    sal_means = [alloc_metrics[k]["mean_assigned_salary_lpa"] for k in strat_keys]
    cov_pcts = [alloc_metrics[k]["placement_coverage_eligible_pct"] for k in strat_keys]

    x = np.arange(len(short_labels))
    w = 0.36
    b1 = ax1.bar(x - w / 2, sal_means, w, color="#15803d", edgecolor="#0f172a", label="Mean Assigned Offer Salary (LPA)")
    ax1.set_ylabel("Mean Assigned Salary (LPA)", fontweight="bold", fontsize=10, color="#15803d")
    ax1.set_ylim(0, max(sal_means + [5.0]) * 1.28)
    ax1.set_xticks(x, short_labels, fontweight="bold", fontsize=9.5)
    ax1.set_xlabel("Campus Drive Allocation Strategy (Identical Capacities & Quotas)", fontweight="bold", fontsize=10)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, cov_pcts, w, color="#1d4ed8", edgecolor="#0f172a", alpha=0.88, label="Eligible Placement Coverage (%)")
    ax2.set_ylabel("Eligible Student Placement Coverage (%)", fontweight="bold", fontsize=10, color="#1d4ed8")
    ax2.set_ylim(0, 118)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max(sal_means + [1]) * 0.02, f"{h:.2f}\nLPA", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#14532d")
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e3a8a")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)
    ax1.set_title("Capacity-Constrained Campus Allocation: Mean Salary & Eligible Coverage", fontweight="bold", fontsize=10.5, pad=12)
    fig.tight_layout()
    p15 = os.path.join(OUTPUT_EVAL_DIR, "salary_optimization.png")
    fig.savefig(p15, dpi=300)
    plt.close(fig)
    saved_files.append(p15)

    cm.save_clustering_distribution_figures()

    return saved_files


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    cm.ensure_output_directories()
    cm.log_info("Starting evaluate_model.main() execution.")
    recorder = EvaluationRecorder()

    recorder.log_line("=" * 84)
    recorder.log_line("SCIENTIFIC MODEL EVALUATION REPORT — STUDENT PROFILING & JOB MATCHING SYSTEM")
    recorder.log_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    recorder.log_line("=" * 84)

    sep_df, k_df, ablation_df = evaluate_clustering_and_ablation(recorder)
    stab_df = evaluate_bootstrap_stability(recorder)
    skill_cm_3x3, skill_acc, skill_kappa, skill_rho, ari_bench, nmi_bench = evaluate_skill_band_benchmark_agreement(recorder)
    placement_eval = evaluate_placement_classification(recorder)
    conf_df, cohort_margins = evaluate_cluster_margin_score(recorder, placement_eval)
    role_rec_results, rand_hits, rand_mrr, rec_df = evaluate_job_recommendation_robustness(recorder)
    opt_df, alloc_metrics = evaluate_allocation_methods(recorder)

    saved_eval_figs = generate_all_individual_evaluation_figures(
        ablation_df,
        skill_cm_3x3,
        skill_acc,
        skill_kappa,
        skill_rho,
        placement_eval,
        cohort_margins,
        role_rec_results,
        rand_hits,
        rand_mrr,
        alloc_metrics,
    )

    recorder.log_line("\n" + "=" * 84)
    recorder.log_line("GENERATED INDIVIDUAL 300-DPI FIGURES & REPORTS")
    recorder.log_line("=" * 84)
    for fig_p in saved_eval_figs:
        recorder.log_line(f"  - {fig_p}")
    for cl_fig in ["cgpa_distribution.png", "skill_cluster_distribution.png", "job_tier_distribution.png"]:
        recorder.log_line(f"  - {os.path.join(OUTPUT_CLUSTER_DIR, cl_fig)}")

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(recorder.report_lines) + "\n")

    metrics_df = pd.DataFrame(recorder.metric_rows)
    metrics_df.to_csv(MODEL_METRICS_CSV_PATH, index=False)

    print(f"\nSaved evaluation report to : {EVAL_REPORT_PATH}")
    print(f"Saved model metrics CSV to : {MODEL_METRICS_CSV_PATH}")
    cm.log_info("Completed evaluate_model.main() successfully.")
    print("\ndone")


if __name__ == "__main__":
    main()