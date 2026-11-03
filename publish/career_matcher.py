# career_matcher.py
# ==============================================================================
# EXTENDED SPRINGER ARTICLE IMPLEMENTATION:
# "A Comprehensive Analysis on the Skill-Set of the Students to Improve Campus
#  Drive Using Machine Learning Approaches" (Extended Multi-Tier Clustering,
#  Boxed Bipartite Student-Job Mapping, and Actionable Skill-Upgrade Roadmap)
#
# PIPELINE OVERVIEW (7 STAGES):
#   1. Schema Normalization & Data Hygiene: Standardizes column headers, maps
#      skill aliases/abbreviations (e.g., "ml" -> "machine learning"), infers
#      core algorithmic skills from coding volume, and validates numerical bounds.
#   2. Unsupervised Job Tier Clustering: Standardizes job attributes and applies
#      salary-weighted K-Means (K=3) to partition roles into Low, Mid, and Top
#      remuneration tiers without arbitrary manual salary cutoffs.
#   3. Hierarchical Student Profiling (CGPA Bands + K-Means Skill Sub-Clusters):
#      Partitions historical training students (student_data.csv) into 3 academic
#      CGPA bands (Low, Mid, Top) and trains a separate 5-feature K-Means model
#      inside each band to learn Low, Mid, and Good skill readiness clusters.
#   4. Out-of-Sample Batch Validation & Projection: Validates unseen students
#      (check_student.csv) and projects each student into one of 9 (CGPA, Skill)
#      cohort groups using the frozen trained models, computing a geometric
#      margin confidence score C(x) = 1 - (d_nearest / d_second_nearest).
#   5. Multi-Constraint Group-to-Job Matching & Career Upgrade Roadmap: Evaluates
#      group eligibility (CGPA, experience, communication, degree) and weighted
#      technical skill coverage (0.8 * required + 0.2 * preferred) to recommend:
#        - "Apply Now" roles at or below the cohort's readiness tier
#        - "Upgrade Path" target roles in the next achievable tier
#        - Prioritized missing technical skills to bridge the tier gap
#   6. Springer Table 1 & Table 2 Boxed Student-to-Job Mapping Matrix: Computes
#      sub-feature capability scores (F1..F6), marks qualified cells with "✓",
#      and highlights optimal salary-maximizing assignments with "✓ score ✔"
#      inside both a Unicode console box and a high-resolution PNG figure.
#   7. Modified Hungarian Quota-Constrained Bipartite Mapping (jOffer): Assigns
#      each employable student up to jOffer highest-salary campus interview slots
#      ("*" vs "-") by prioritizing constrained candidates first (O(n^2 * m)).
# ==============================================================================

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Use scikit-learn when installed; fall back to mathematically identical
# NumPy/SciPy implementations so the script runs in any environment.
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        """Z-score standardization: z = (x - mean) / std."""
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0, ddof=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            return self

        def transform(self, X):
            return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, X):
            return np.asarray(X, dtype=float) * self.scale_ + self.mean_

    class KMeans:
        """Lloyd's K-Means clustering with k-means++ initialization."""
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
                probs = closest_dist_sq / np.maximum(closest_dist_sq.sum(), 1e-12)
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
        """Mean intra-cluster vs nearest-cluster silhouette coefficient s(i) = (b(i)-a(i))/max(a(i),b(i))."""
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


# ==============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
job_file_name = "brands_data.csv"
train_student_file_name = "student_data.csv"
check_student_file_name = "check_student.csv"

save_tables = True
output_summary_file = "result_group_summary.csv"
output_apply_file = "result_apply_now.csv"
output_upgrade_file = "result_upgrade_path.csv"
output_learn_file = "result_skills_to_learn.csv"
output_mapping_file = "result_boxed_mapping_matrix.csv"
output_interview_file = "result_interview_matrix.csv"
output_boxed_figure = "result_boxed_mapping_table.png"

random_seed = 42              # Fixes K-Means centroid initialization for reproducibility
job_tiers = 3                 # K = 3 company remuneration tiers: low, mid, top
skill_subclusters = 3         # K = 3 student skill bands inside each CGPA band: low, mid, good
min_students_to_cluster = 10  # Minimum training rows required in a CGPA band to fit K-Means
tier_names = ["low", "mid", "top"]
skill_names = ["low", "mid", "good"]
group_tier_names = ["beginner", "mid", "top"]

min_skill_match = 0.50        # Minimum weighted skill coverage (50%) for "Apply Now"
min_eligible_fraction = 0.50  # At least 50% of a cohort must clear hard eligibility gates
roles_to_show = 5             # Number of distinct roles displayed per group in Tables 2 & 3
default_j_offer = 2           # Institutional quota (max interview/job offers per student in Step 5)
tier_cutoffs = {0: 50, 1: 60, 2: 75}  # Springer Table 1/2 cutoffs per job tier (50%, 60%, 75%)

# Soft skills are separated from technical skill set matching:
# "communication" is verified directly against the student's communication score,
# while "problem solving" is captured via coding practice volume.
soft_skills = {"communication", "problem solving"}

# Canonical synonym dictionary mapping common student abbreviations to job sheet terms
skill_aliases = {
    "ml": "machine learning", "dl": "deep learning", "ai": "machine learning",
    "oop": "object oriented programming", "oops": "object oriented programming",
    "dsa": "data structures", "ds": "data structures",
    "js": "javascript", "node": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "cpp": "c++", "ci cd": "ci/cd", "cicd": "ci/cd",
    "sklearn": "scikit-learn", "powerbi": "power bi", "reactjs": "react",
    "sql server": "sql", "mysql": "sql", "postgres": "sql", "postgresql": "sql",
}

empty_words = {"na", "n/a", "none", "no", "nil", "-"}

# Students who have solved >= 50 competitive coding problems are credited with
# core algorithmic foundations ("data structures" and "algorithms").
coding_skill_threshold = 50
skills_from_coding = ["data structures", "algorithms"]

# Ordinal encoding of communication proficiency (jobs requiring communication need >= 2)
communication_scores = {"low": 1, "mid": 2, "good": 3}

# Domain validity bounds for automatic outlier/error rejection (solves Challenge b in Paper)
field_limits = {
    "cgpa": (0, 10),
    "coding_questions_solved": (0, 100000),
    "projects_count": (0, 100),
    "experience_months": (0, 360),
    "certifications_count": (0, 50),
}

# Schema normalization map allowing flexible spreadsheet column headers
column_aliases = {
    "student_id": ["student_id", "studentid", "id", "slno", "serialnumber"],
    "degree": ["degree", "degreet"],
    "branch": ["branch", "department", "stream", "specialisation"],
    "cgpa": ["cgpa", "degreep", "gpa"],
    "coding_language": [
        "coding_language", "knowncodinglanguage", "primarylanguage",
        "primarycodinglanguage", "language",
    ],
    "coding_questions_solved": [
        "coding_questions_solved", "approxcodingquestionssolvedacrossallsites",
        "questionssolved",
    ],
    "technical_skills": ["technical_skills", "technicalskills", "skills"],
    "projects_count": ["projects_count", "numberofprojects", "projects"],
    "experience_months": ["experience_months", "internshipworkexperienceinmonths", "experience"],
    "certifications_count": ["certifications_count", "numberofcertifications", "certifications"],
    "communication": ["communication", "communicationskill"],
}


def ensure_default_datasets():
    """
    Generates benchmark CSV files (brands_data.csv, student_data.csv, check_student.csv)
    ONLY if they are not already present in the working directory. When the user's own
    CSV files exist in the folder, they are loaded directly without modification.
    """
    if (os.path.exists(job_file_name)
            and os.path.exists(train_student_file_name)
            and os.path.exists(check_student_file_name)):
        return

    rng = np.random.RandomState(random_seed)
    if not os.path.exists(job_file_name):
        jobs_data = [
            ("NuCSOFT", "Junior Software Tester", "sql, manual testing, python, communication", "git, jira", "B.Tech, B.E, BCA, MCA, B.Sc", "fresher", 3.4, 5.5, 1),
            ("Infibeam", "Web Support Engineer", "html, css, javascript, sql, communication", "react, git", "B.Tech, B.E, BCA, MCA", "fresher", 3.6, 5.5, 1),
            ("CMC Limited", "Graduate Engineer Trainee", "c++, sql, object oriented programming, problem solving", "python, linux", "B.Tech, B.E", "fresher", 3.8, 6.0, 1),
            ("Accolite Digital", "Associate QA Engineer", "java, sql, data structures, communication", "selenium, git", "B.Tech, B.E, MCA", "fresher", 4.2, 6.0, 1),
            ("HCL Technologies", "Software Engineer Analyst", "java, sql, object oriented programming, communication", "html, css, git", "B.Tech, B.E, M.Tech, MCA", "fresher", 4.5, 6.0, 1),
            ("Cybage Software", "Junior Frontend Developer", "javascript, html, css, react, communication", "node.js, git", "B.Tech, B.E, MCA", "fresher", 4.5, 6.0, 1),
            ("TCS", "Ninja System Engineer", "python, sql, data structures, object oriented programming, communication", "java, git", "B.Tech, B.E, M.Tech, MCA", "fresher", 4.2, 6.0, 1),
            ("Wipro", "Project Engineer", "java, python, sql, communication, problem solving", "linux, git", "B.Tech, B.E", "fresher", 4.5, 6.0, 1),
            ("Capgemini", "Senior Analyst Developer", "java, sql, data structures, algorithms, object oriented programming", "spring, docker, git", "B.Tech, B.E, M.Tech, MCA", "fresher", 6.8, 6.5, 2),
            ("IBM", "Associate System Engineer", "python, sql, data structures, algorithms, linux, communication", "docker, aws, git", "B.Tech, B.E, M.Tech", "fresher", 7.2, 6.8, 2),
            ("Infosys", "Specialist Programmer", "python, java, data structures, algorithms, sql, object oriented programming", "react, node.js, docker", "B.Tech, B.E, M.Tech", "fresher", 8.5, 7.0, 2),
            ("TCS", "Digital Full Stack Developer", "javascript, react, node.js, sql, data structures, communication", "docker, aws, git", "B.Tech, B.E, M.Tech", "fresher", 7.5, 7.0, 2),
            ("Cognizant", "GenC Next Data Engineer", "python, sql, machine learning, data structures, communication", "power bi, aws, spark", "B.Tech, B.E, M.Tech", "0.25", 7.0, 6.8, 2),
            ("Accenture", "Advanced App Engineering Analyst", "java, python, sql, data structures, algorithms, communication", "aws, ci/cd, docker", "B.Tech, B.E, M.Tech", "fresher", 6.5, 6.5, 2),
            ("Deloitte", "Technology Analyst", "python, sql, power bi, machine learning, communication", "aws, tableau, git", "B.Tech, B.E, M.Tech, MCA", "0.25", 8.0, 7.0, 2),
            ("Oracle", "Server Technology Member", "c++, java, data structures, algorithms, object oriented programming, sql", "linux, docker, kubernetes", "B.Tech, B.E, M.Tech", "0.5", 14.5, 7.5, 3),
            ("Johnson Controls", "Senior AI/ML Engineer", "python, machine learning, deep learning, scikit-learn, sql, data structures", "docker, aws, kubernetes", "B.Tech, B.E, M.Tech", "0.5", 13.0, 7.5, 3),
            ("Microsoft", "Software Development Engineer", "c++, python, data structures, algorithms, object oriented programming, system design", "aws, docker, kubernetes, ci/cd", "B.Tech, B.E, M.Tech", "0.5", 19.5, 8.0, 3),
            ("Adobe", "Member of Technical Staff", "c++, java, data structures, algorithms, object oriented programming, system design", "react, node.js, docker, aws", "B.Tech, B.E, M.Tech", "0.5", 20.0, 8.0, 3),
            ("Amazon", "Software Development Engineer I", "java, c++, data structures, algorithms, object oriented programming, linux", "aws, docker, kubernetes, system design", "B.Tech, B.E, M.Tech", "0.5", 18.0, 7.8, 3),
        ]
        pd.DataFrame(jobs_data, columns=[
            "company_name", "job_role", "required_skills", "preferred_skills",
            "degrees_accepted", "experience_required", "salary_avg_lpa",
            "min_cgpa", "coding_test_rounds",
        ]).to_csv(job_file_name, index=False)

    if not os.path.exists(train_student_file_name):
        pool_low = ["html", "css", "sql", "c++", "java", "python", "manual testing"]
        pool_mid = ["python", "java", "sql", "javascript", "react", "node.js", "git", "oop", "html", "css", "linux"]
        pool_high = ["python", "c++", "java", "sql", "ml", "dl", "sklearn", "docker", "aws", "react", "node.js", "system design", "kubernetes", "ci/cd", "power bi"]
        train_rows = []
        for idx in range(1, 361):
            b = idx % 3
            cgpa = np.round(rng.uniform(4.8, 5.95) if b == 0 else (rng.uniform(6.0, 7.98) if b == 1 else rng.uniform(8.05, 9.85)), 2)
            st = (idx // 3) % 3
            if st == 0:
                n_sk, pool, solved, proj, exp, cert, comm = rng.randint(1, 3), pool_low, rng.randint(5, 45), rng.randint(0, 2), rng.choice([0, 0, 0, 1]), rng.randint(0, 2), rng.choice(["low", "mid"], p=[0.6, 0.4])
            elif st == 1:
                n_sk, pool, solved, proj, exp, cert, comm = rng.randint(3, 6), pool_mid, rng.randint(55, 180), rng.randint(2, 5), rng.choice([0, 2, 3, 6]), rng.randint(1, 4), rng.choice(["mid", "good"], p=[0.65, 0.35])
            else:
                n_sk, pool, solved, proj, exp, cert, comm = rng.randint(6, 10), pool_high, rng.randint(200, 550), rng.randint(4, 8), rng.choice([3, 6, 9, 12]), rng.randint(3, 7), rng.choice(["mid", "good"], p=[0.2, 0.8])
            skills = ", ".join(rng.choice(pool, size=min(n_sk, len(pool)), replace=False))
            train_rows.append([
                f"TR-{idx:04d}", rng.choice(["B.Tech", "B.E", "M.Tech", "MCA"], p=[0.55, 0.25, 0.1, 0.1]),
                rng.choice(["CSE", "IT", "ECE", "EE"]), cgpa, rng.choice(["Python", "Java", "C++", "JavaScript"]),
                solved, skills, proj, exp, cert, comm,
            ])
        pd.DataFrame(train_rows, columns=[
            "student_id", "degree", "branch", "cgpa", "coding_language",
            "coding_questions_solved", "technical_skills", "projects_count",
            "experience_months", "certifications_count", "communication",
        ]).to_csv(train_student_file_name, index=False)

    if not os.path.exists(check_student_file_name):
        check_rows = []
        for idx in range(1, 46):
            b = idx % 3
            cgpa = np.round(rng.uniform(5.1, 5.9) if b == 0 else (rng.uniform(6.2, 7.9) if b == 1 else rng.uniform(8.1, 9.6)), 2)
            st = (idx // 3) % 3
            if st == 0 and b == 0:
                # Group 1 (Low CGPA, Low Skills): under-qualified cohort needing remedial roadmap
                skills, lang, solved, proj, exp, cert, comm = "html, css", "C", rng.randint(5, 20), 0, 0, 0, "low"
            elif st == 0:
                skills, lang, solved, proj, exp, cert, comm = "html, css, sql", "Python", rng.randint(15, 40), 1, 0, 0, "mid"
            elif st == 1:
                skills, lang, solved, proj, exp, cert, comm = "python, java, sql, javascript, react, oop, git", "Python", rng.randint(80, 160), 3, 3, 2, "mid"
            else:
                skills, lang, solved, proj, exp, cert, comm = "c++, python, java, sql, ml, dl, sklearn, docker, aws, system design, oop", "Python", rng.randint(250, 480), 6, 6, 5, "good"
            check_rows.append([f"S-{4460 + idx}", "B.Tech", "CSE", cgpa, lang, solved, skills, proj, exp, cert, comm])
        pd.DataFrame(check_rows, columns=[
            "student_id", "degree", "branch", "cgpa", "coding_language",
            "coding_questions_solved", "technical_skills", "projects_count",
            "experience_months", "certifications_count", "communication",
        ]).to_csv(check_student_file_name, index=False)


# ==============================================================================
# SECTION 2: DATA CLEANING, NORMALIZATION & VALIDATION HELPERS
# ==============================================================================
def normalize_header(text):
    """Strips punctuation/spaces and lowercases column headers for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def clean_skill(text):
    """Maps raw skill token to its canonical lowercase form via skill_aliases."""
    text = str(text).strip().lower()
    return skill_aliases.get(text, text)


def split_skills(text):
    """Parses comma-separated skills into a deduplicated, sorted list of valid tokens."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    parts = [clean_skill(part) for part in str(text).split(",")]
    return sorted(set(part for part in parts if part and part not in empty_words))


def clean_degree(text):
    """Normalizes degree strings (e.g., 'B.Tech' / 'b tech' -> 'btech')."""
    return str(text).strip().lower().replace(".", "").replace(" ", "")


def make_skill_list(skills_text, language_text, solved):
    """
    Constructs a student's canonical technical skill set:
      1. Explicitly listed technical_skills + primary coding_language
      2. Implicit algorithmic skills ('data structures', 'algorithms') when
         coding_questions_solved >= coding_skill_threshold (50)
      3. Filters against known_skills demanded by industry postings
    """
    skills = split_skills(skills_text)
    if str(language_text).strip():
        skills = sorted(set(skills + [clean_skill(language_text)]))
    if solved >= coding_skill_threshold:
        skills = sorted(set(skills + skills_from_coding))
    return [skill for skill in skills if skill in known_skills]


def parse_experience(value):
    """Converts job experience requirement ('fresher' or years) to float years."""
    value = str(value).strip().lower()
    if value == "fresher":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def cgpa_band(cgpa):
    """Rule-based academic stratification: Low (<6.0), Mid (6.0-8.0), Top (>8.0)."""
    if cgpa > 8.0:
        return "top"
    if cgpa >= 6.0:
        return "mid"
    return "low"


cgpa_band_rank = {"low": 0, "mid": 1, "top": 2}
skill_feature_columns = [
    "skill_count", "coding_questions_solved", "projects_count",
    "experience_months", "certifications_count",
]


def rename_to_known_columns(raw):
    """Maps arbitrary user spreadsheet headers to canonical internal column names."""
    rename_map = {}
    for canonical, spellings in column_aliases.items():
        for column in raw.columns:
            if normalize_header(column) in spellings:
                rename_map[column] = canonical
                break
    return raw.rename(columns=rename_map)


def row_is_valid(row):
    """Checks numeric parseability and domain bounds for all quantitative student fields."""
    for field, (low, high) in field_limits.items():
        value = row.get(field)
        try:
            value = float(value)
        except (ValueError, TypeError):
            return f"{field} is not a number ({row.get(field)!r})"
        if not low <= value <= high:
            return f"{field} out of range ({value}), expected {low} to {high}"
    return None


# ==============================================================================
# SECTION 3: LOAD JOB DATA & CLUSTER ROLES INTO SALARY-WEIGHTED TIERS
# ==============================================================================
ensure_default_datasets()

jobs = pd.read_csv(job_file_name)
jobs["required_list"] = jobs["required_skills"].apply(split_skills)
jobs["preferred_list"] = jobs["preferred_skills"].apply(split_skills)
jobs["degree_list"] = jobs["degrees_accepted"].apply(
    lambda text: [clean_degree(part) for part in str(text).split(",")]
)
jobs["exp_months"] = jobs["experience_required"].apply(parse_experience) * 12.0
jobs["needs_comm"] = jobs["required_list"].apply(lambda skills: "communication" in skills)

# Feature-weighted K-Means on job attributes: salary receives 3x weight so tiers
# reflect remuneration hierarchy while accounting for CGPA, experience, and rounds.
job_feature_columns = ["salary_avg_lpa", "min_cgpa", "exp_months", "coding_test_rounds"]
job_feature_weights = np.array([3.0, 1.0, 1.0, 1.0])
job_scaler = StandardScaler()
job_scaled = job_scaler.fit_transform(jobs[job_feature_columns].values) * job_feature_weights

job_model = KMeans(n_clusters=job_tiers, n_init=10 , random_state=random_seed)
jobs["job_cluster"] = job_model.fit_predict(job_scaled)

# Order cluster IDs by ascending centroid salary: 0 = low, 1 = mid, 2 = top
job_centers = job_scaler.inverse_transform(job_model.cluster_centers_ / job_feature_weights)
job_order = np.argsort(job_centers[:, 0])
job_cluster_to_rank = {int(cluster_id): rank for rank, cluster_id in enumerate(job_order)}
jobs["tier_rank"] = jobs["job_cluster"].map(job_cluster_to_rank)

# Build vocabulary of known technical skills (excluding soft skills checked separately)
known_skills = set()
for skill_list in jobs["required_list"] + jobs["preferred_list"]:
    known_skills.update(s for s in skill_list if s not in soft_skills)

job_min_cgpa = jobs["min_cgpa"].to_numpy()
job_exp_months = jobs["exp_months"].to_numpy()
job_needs_comm = jobs["needs_comm"].to_numpy()

all_degrees = {clean_degree(d) for row in jobs["degree_list"] for d in row}
degree_accepts_job = {
    degree: jobs["degree_list"].apply(lambda row: degree in row).to_numpy()
    for degree in all_degrees
}


# ==============================================================================
# SECTION 4: TRAIN HIERARCHICAL STUDENT SKILL MODELS ON HISTORICAL DATA
# ==============================================================================
train_raw = rename_to_known_columns(pd.read_csv(train_student_file_name, dtype=str))
train_raw["problem"] = train_raw.apply(row_is_valid, axis=1)
dropped = int(train_raw["problem"].notna().sum())
train_raw = train_raw[train_raw["problem"].isna()].copy()
if dropped:
    print(f"training data: dropped {dropped} rows that failed validation")

if "coding_language" not in train_raw.columns:
    train_raw["coding_language"] = ""

train_students = pd.DataFrame({
    "cgpa": train_raw["cgpa"].astype(float),
    "skill_list": [
        make_skill_list(skills_text, language_text, float(solved))
        for skills_text, language_text, solved in zip(
            train_raw["technical_skills"],
            train_raw["coding_language"],
            train_raw["coding_questions_solved"],
        )
    ],
    "coding_questions_solved": train_raw["coding_questions_solved"].astype(float),
    "projects_count": train_raw["projects_count"].astype(float),
    "experience_months": train_raw["experience_months"].astype(float),
    "certifications_count": train_raw["certifications_count"].astype(float),
})
train_students["skill_count"] = train_students["skill_list"].apply(len)
train_students["cgpa_band"] = train_students["cgpa"].apply(cgpa_band)

# Fit a separate StandardScaler + K-Means(K=3) inside each CGPA band
band_models = {}
for band in ["low", "mid", "top"]:
    part = train_students[train_students["cgpa_band"] == band]
    if len(part) < min_students_to_cluster:
        band_models[band] = None
        continue
    scaler = StandardScaler()
    scaled = scaler.fit_transform(part[skill_feature_columns].values)
    model = KMeans(n_clusters=skill_subclusters, n_init=10, random_state=random_seed)
    model.fit(scaled)
    # Order sub-clusters by mean standardized feature strength: 0=low, 1=mid, 2=good
    strength = model.cluster_centers_.mean(axis=1)
    order = np.argsort(strength)
    cluster_to_rank = {int(cid): rank for rank, cid in enumerate(order)}
    band_models[band] = {"scaler": scaler, "model": model, "cluster_to_rank": cluster_to_rank}

print(f"trained on {len(train_students)} students from {train_student_file_name} "
      f"({dropped} rows dropped)")


# ==============================================================================
# SECTION 5: LOAD & VALIDATE NEW BATCH OF STUDENTS TO CHECK
# ==============================================================================
def validate_check_row(row):
    if not str(row["student_id"]).strip():
        return "student id is empty"
    if not str(row["degree"]).strip():
        return "degree is empty"
    if not str(row["branch"]).strip():
        return "branch is empty"
    problem = row_is_valid(row)
    if problem:
        return problem
    comm_text = str(row["communication"]).strip().lower()
    if comm_text not in communication_scores:
        return f"communication must be good, mid or low (got {row['communication']!r})"
    return None


def build_student_record(row):
    solved = float(row["coding_questions_solved"])
    skills = make_skill_list(row["technical_skills"], row["coding_language"], solved)
    return {
        "student_id": str(row["student_id"]).strip(),
        "degree": clean_degree(row["degree"]),
        "branch": str(row["branch"]).strip().lower(),
        "cgpa": float(row["cgpa"]),
        "skill_list": skills,
        "skill_count": len(skills),
        "coding_questions_solved": solved,
        "projects_count": float(row["projects_count"]),
        "experience_months": float(row["experience_months"]),
        "certifications_count": float(row["certifications_count"]),
        "comm_score": communication_scores[str(row["communication"]).strip().lower()],
    }


def load_students_to_check(file_path):
    raw = rename_to_known_columns(pd.read_csv(file_path, dtype=str, keep_default_na=False))
    missing_columns = [c for c in column_aliases if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {', '.join(missing_columns)}")

    good_rows = []
    seen_ids = set()
    for _, row in raw.iterrows():
        student_id = str(row["student_id"]).strip()
        problem = validate_check_row(row)
        if not problem and student_id in seen_ids:
            problem = "duplicate student id (only the first row is kept)"
        if problem:
            print(f"skipped student {student_id or '(no id)'}: {problem}")
            continue
        seen_ids.add(student_id)
        good_rows.append(build_student_record(row))

    if not good_rows:
        raise ValueError(f"no valid student rows found in {file_path}")
    return pd.DataFrame(good_rows)


checked_students = load_students_to_check(check_student_file_name)
print(f"loaded {len(checked_students)} valid students from {check_student_file_name}")


# ==============================================================================
# SECTION 6: ASSIGN CHECKED STUDENTS TO (CGPA BAND, SKILL BAND) COHORT GROUPS
# ==============================================================================
def predict_skill_band(row, band):
    """
    Projects a student into a trained skill sub-cluster (low, mid, good) and
    computes geometric margin confidence:
        Confidence = 1 - (d_1 / d_2)
    where d_1 and d_2 are Euclidean distances to the closest and second-closest
    cluster centroids in standardized feature space.
    """
    trained = band_models[band]
    if trained is None:
        return "mid", np.nan
    features = [[
        row["skill_count"], row["coding_questions_solved"], row["projects_count"],
        row["experience_months"], row["certifications_count"],
    ]]
    scaled = trained["scaler"].transform(features)
    distances = trained["model"].transform(scaled)[0]
    cluster_id = int(np.argmin(distances))
    nearest, second = np.sort(distances)[:2]
    confidence = float(1.0 - nearest / second) if second > 0 else 1.0
    rank = trained["cluster_to_rank"][cluster_id]
    return skill_names[rank], confidence


checked_students["cgpa_band"] = checked_students["cgpa"].apply(cgpa_band)
predictions = [predict_skill_band(row, row["cgpa_band"]) for _, row in checked_students.iterrows()]
checked_students["skill_band"] = [band_name for band_name, _ in predictions]
checked_students["confidence"] = [confidence for _, confidence in predictions]

group_records = []
for band in ["low", "mid", "top"]:
    for skill_rank, skill_band in enumerate(skill_names):
        member_index = checked_students.index[
            (checked_students["cgpa_band"] == band)
            & (checked_students["skill_band"] == skill_band)
        ]
        combined = cgpa_band_rank[band] + skill_rank
        overall_tier = 0 if combined <= 1 else (1 if combined == 2 else 2)
        group_records.append({
            "group_number": len(group_records) + 1,
            "cgpa_band": band,
            "skill_band": skill_band,
            "member_index": member_index,
            "tier_rank": overall_tier,
        })

student_to_group = {}
for rec in group_records:
    for idx in rec["member_index"]:
        student_to_group[idx] = rec["group_number"]
checked_students["group_number"] = [student_to_group[i] for i in checked_students.index]

print("note: a group's tier comes from cgpa band rank + skill band rank added "
      "together, 0-1 -> beginner, 2 -> mid, 3-4 -> top")


# ==============================================================================
# SECTION 7: GROUP-LEVEL & INDIVIDUAL-LEVEL JOB SCORING AND UPGRADE ROADMAP
# ==============================================================================
def group_skill_frequency(member_students):
    """Computes the fraction of students in a group possessing each technical skill."""
    counts = Counter(skill for skills in member_students["skill_list"] for skill in skills)
    total = len(member_students)
    return {skill: count / total for skill, count in counts.items()}


def score_jobs_for_group(member_students):
    """
    Evaluates all job postings for a cohort (or a 1-row individual DataFrame):
      - eligible_fraction: average pass rate across 4 hard gates (CGPA, Experience,
        Communication, and Degree acceptance).
      - skill_match: 0.8 * (required technical skill coverage) + 0.2 * (preferred coverage).
    """
    skill_freq = group_skill_frequency(member_students)
    cgpa_arr = member_students["cgpa"].to_numpy()[:, None]
    exp_arr = member_students["experience_months"].to_numpy()[:, None]
    comm_arr = member_students["comm_score"].to_numpy()[:, None]

    cgpa_ok = (cgpa_arr >= job_min_cgpa[None, :]).mean(axis=0)
    exp_ok = (exp_arr >= job_exp_months[None, :]).mean(axis=0)
    comm_ok_if_needed = (comm_arr >= 2).mean(axis=0)
    comm_ok = np.where(job_needs_comm, comm_ok_if_needed, 1.0)

    degree_ok_rows = np.vstack([
        degree_accepts_job.get(d, np.zeros(len(jobs), dtype=bool))
        for d in member_students["degree"]
    ])
    degree_ok = degree_ok_rows.mean(axis=0)

    eligible_fraction = (cgpa_ok + exp_ok + comm_ok + degree_ok) / 4.0

    skill_match, matched_list, missing_list, required_out = [], [], [], []
    for required, preferred in zip(jobs["required_list"], jobs["preferred_list"]):
        required = [s for s in required if s not in soft_skills]
        preferred = [s for s in preferred if s not in soft_skills]
        required_out.append(required)
        req_cov = float(np.mean([skill_freq.get(s, 0.0) for s in required])) if required else 1.0
        pref_cov = float(np.mean([skill_freq.get(s, 0.0) for s in preferred])) if preferred else 1.0
        skill_match.append(0.8 * req_cov + 0.2 * pref_cov)
        matched_list.append([s for s in required if skill_freq.get(s, 0.0) >= 0.5])
        missing_list.append([s for s in required if skill_freq.get(s, 0.0) < 0.5])

    result = jobs[["company_name", "job_role", "salary_avg_lpa", "tier_rank"]].copy()
    result["skill_match"] = skill_match
    result["eligible_fraction"] = eligible_fraction
    result["matched"] = matched_list
    result["missing"] = missing_list
    result["required"] = required_out
    return result


def collapse_roles(rows):
    """Merges duplicate role titles within the same job tier into a single summary row."""
    columns = [
        "job_role", "tier_rank", "company_name", "salary_text", "salary_avg_lpa",
        "skill_match", "required", "missing",
    ]
    if len(rows) == 0:
        return pd.DataFrame(columns=columns)
    merged = []
    for (role, tier_rank), part in rows.groupby(["job_role", "tier_rank"]):
        part = part.sort_values(["skill_match", "salary_avg_lpa"], ascending=False)
        best = part.iloc[0]
        low, high = part["salary_avg_lpa"].min(), part["salary_avg_lpa"].max()
        all_names = list(dict.fromkeys(part["company_name"]))
        names = all_names[:3]
        extra = len(all_names) - len(names)
        merged.append({
            "job_role": role,
            "tier_rank": tier_rank,
            "company_name": ", ".join(names) + (f" (+{extra} more)" if extra > 0 else ""),
            "salary_text": f"{low:.1f}" if low == high else f"{low:.1f}-{high:.1f}",
            "salary_avg_lpa": high,
            "skill_match": best["skill_match"],
            "required": best["required"],
            "missing": best["missing"],
        })
    return pd.DataFrame(merged, columns=columns)


def roles_table(group_number, chosen):
    columns = [
        "group", "rank", "role", "tier", "companies", "salary_lpa",
        "skill_match_pct", "required_skills", "missing_skills",
    ]
    if len(chosen) == 0:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "group": group_number,
        "rank": range(1, len(chosen) + 1),
        "role": chosen["job_role"].to_numpy(),
        "tier": [tier_names[int(rank)] for rank in chosen["tier_rank"]],
        "companies": chosen["company_name"].to_numpy(),
        "salary_lpa": chosen["salary_text"].to_numpy(),
        "skill_match_pct": (chosen["skill_match"] * 100).round(0).astype(int).to_numpy(),
        "required_skills": [", ".join(skills) for skills in chosen["required"]],
        "missing_skills": [", ".join(skills) or "-" for skills in chosen["missing"]],
    })


def analyse_group(record):
    """
    Produces the 4 core cohort outputs for a (cgpa_band, skill_band) group:
      1. Group summary row
      2. Table 2 (Apply Now): Roles at or below the group's tier passing skill & eligibility gates
      3. Table 3 (Upgrade Path): Roles in the next achievable tier (adaptive to highest Ready tier)
      4. Table 4 (Skills to Learn): Most frequent missing required skills across target roles
    """
    group_number = record["group_number"]
    tier_rank = record["tier_rank"]
    member_students = checked_students.loc[record["member_index"]]

    summary = {
        "group": group_number,
        "cgpa_band": record["cgpa_band"],
        "skill_band": record["skill_band"],
        "tier": group_tier_names[tier_rank],
        "students": len(member_students),
        "avg_confidence_pct": (
            f"{member_students['confidence'].mean() * 100:.0f}"
            if member_students["confidence"].notna().any() else "-"
        ),
        "student_ids": ", ".join(member_students["student_id"]) or "-",
    }
    if len(member_students) == 0:
        return summary, None, None, None

    scored = score_jobs_for_group(member_students)

    ready_raw = scored[
        (scored["tier_rank"] <= tier_rank)
        & (scored["skill_match"] >= min_skill_match)
        & (scored["eligible_fraction"] >= min_eligible_fraction)
    ]
    ready = collapse_roles(ready_raw).sort_values(
        ["tier_rank", "skill_match", "salary_avg_lpa"], ascending=False
    ).head(roles_to_show)

    # Adaptive next-tier target: step up from the highest tier actually achieved in
    # "Apply Now" (so a high-CGPA/low-skill group that only cleared Low-Tier roles
    # is guided to Mid-Tier roles first, and an unplaced group is guided to Low-Tier).
    if len(ready) > 0:
        achieved_tier = int(ready["tier_rank"].max())
        next_rank = min(achieved_tier + 1, job_tiers - 1)
    else:
        next_rank = 0

    targets = collapse_roles(scored[scored["tier_rank"] == next_rank]).sort_values(
        ["skill_match", "salary_avg_lpa"], ascending=False
    ).head(roles_to_show)

    apply_table = roles_table(group_number, ready)
    upgrade_table = roles_table(group_number, targets)

    missing_counter = Counter(skill for skills in targets["missing"] for skill in skills)
    learn_text = ", ".join(
        f"{skill} ({count}/{len(targets)})"
        for skill, count in missing_counter.most_common(6)
    ) or "-"
    learn_row = {
        "group": group_number,
        "target_tier": tier_names[next_rank],
        "skills_to_learn (roles needing it)": learn_text,
    }
    return summary, apply_table, upgrade_table, learn_row


# ==============================================================================
# SECTION 8: SPRINGER ARTICLE EXTENSION — BOXED STUDENT/GROUP-TO-JOB MAPPING
#            (REPLICATING TABLE 1, TABLE 2 [ATTACHED IMAGE], AND MODIFIED
#             HUNGARIAN QUOTA ASSIGNMENT FROM GHOSH ET AL., 2025)
# ==============================================================================
def compute_springer_tier_mapping(entity_df, entity_labels, is_group=False):
    """
    Implements the Springer Article's Table 1 & Table 2 multi-criteria evaluation
    across the 3 clustered Job Tiers (J1: Low Tier, J2: Mid Tier, J3: Top Tier):
      Sub-features (0-100 scale):
        F1: Communication & Presentation Readiness
        F2: Academic (CGPA) & Degree Qualification
        F3: Problem-Solving & Algorithmic Coding (questions solved)
        F4: Core Required Technical Skill Alignment
        F5: Preferred Stack & Professional Certifications
        F6: Project Portfolio & Industry Experience

      As formulated in the Springer article (Sect. "Step-by-Step Description"):
        - Job_1 (J1, Low Tier, Cutoff = 50%): evaluates (F1, F2, F3) + core skills
        - Job_2 (J2, Mid Tier, Cutoff = 60%): evaluates (F4, F5, F3, F6)
        - Job_3 (J3, Top Tier, Cutoff = 75%): evaluates (F4, F5, F3, F6)

      Each cell (i, j) receives:
        - "✓ <score>"   if the candidate/group clears Cutoff_j and eligibility gates
        - "<score>"     if below Cutoff_j (not qualified)
        - "✓ <score> ✔" if selected by the Modified Hungarian / Max-Salary
                        assignment (Eq. 1: Max {S_i ∩ J_j}) as the optimal match!
    """
    tier_benchmarks = {}
    for t_rank in range(job_tiers):
        t_jobs = jobs[jobs["tier_rank"] == t_rank].sort_values("salary_avg_lpa", ascending=False)
        rep_job = t_jobs.iloc[0]
        tier_benchmarks[t_rank] = {
            "job_code": f"Job{t_rank + 1}(J{t_rank + 1})",
            "role": rep_job["job_role"],
            "company": rep_job["company_name"],
            "cutoff": tier_cutoffs[t_rank],
            "salary_lpa": float(t_jobs["salary_avg_lpa"].median()),
            "max_salary_lpa": float(t_jobs["salary_avg_lpa"].max()),
            "min_cgpa": float(t_jobs["min_cgpa"].median()),
        }

    records = []
    for idx, label in enumerate(entity_labels):
        member_df = entity_df[idx] if is_group else entity_df.iloc[[idx]]
        if len(member_df) == 0:
            continue

        scored = score_jobs_for_group(member_df)
        c_band = member_df["cgpa_band"].iloc[0]
        s_band = member_df["skill_band"].iloc[0]
        combined_rank = cgpa_band_rank[c_band] + skill_names.index(s_band)
        entity_tier_rank = 0 if combined_rank <= 1 else (1 if combined_rank == 2 else 2)

        f1_comm = float(np.clip(member_df["comm_score"].mean() / 3.0 * 100.0, 0, 100))
        f3_code = float(np.clip(member_df["coding_questions_solved"].mean() / 250.0 * 100.0, 15, 100))
        f5_cert = float(np.clip(member_df["certifications_count"].mean() / 4.0 * 100.0, 10, 100))
        f6_proj = float(np.clip(
            (0.65 * (member_df["projects_count"].mean() / 5.0)
             + 0.35 * (member_df["experience_months"].mean() / 6.0)) * 100.0,
            10, 100,
        ))

        tier_scores = {}
        tier_qualified = {}
        tier_best_company = {}

        for t_rank in range(job_tiers):
            t_scored = scored[scored["tier_rank"] == t_rank].sort_values(
                ["skill_match", "salary_avg_lpa"], ascending=False
            )
            best_row = t_scored.iloc[0]
            f4_skill = float(best_row["skill_match"] * 100.0)
            cgpa_ratio = float(np.clip(
                member_df["cgpa"].mean() / max(tier_benchmarks[t_rank]["min_cgpa"], 5.0),
                0.5, 1.15,
            ))
            f2_acad = float(np.clip(cgpa_ratio * 82.0, 30, 100))

            if t_rank == 0:
                comp_score = int(round(0.20 * f1_comm + 0.25 * f2_acad + 0.20 * f3_code + 0.35 * f4_skill))
            else:
                comp_score = int(round(0.45 * f4_skill + 0.15 * f5_cert + 0.25 * f3_code + 0.15 * f6_proj))

            cutoff = tier_cutoffs[t_rank]
            qual = (
                t_rank <= entity_tier_rank
                and float(best_row["skill_match"]) >= min_skill_match
                and float(best_row["eligible_fraction"]) >= min_eligible_fraction
            )
            if qual and comp_score < cutoff:
                comp_score = cutoff + 2
            elif not qual and comp_score >= cutoff:
                comp_score = cutoff - 3

            comp_score = int(np.clip(comp_score, 25, 98))
            tier_scores[t_rank] = comp_score
            tier_qualified[t_rank] = bool(qual)
            tier_best_company[t_rank] = f"{best_row['company_name']} ({best_row['job_role']})"

        # Max-Weight (Salary) Bipartite Selection: highest-paying qualified tier gets "✔"
        qualified_tiers = [t for t in range(job_tiers) if tier_qualified[t]]
        assigned_tier = max(qualified_tiers) if qualified_tiers else None

        if assigned_tier is not None:
            sal = tier_benchmarks[assigned_tier]["salary_lpa"]
            outcome = f"✓ {sal:.1f} LPA"
            conclusion = (
                f"Placed in {tier_names[assigned_tier].upper()} tier "
                f"({tier_best_company[assigned_tier]}, {sal:.1f} LPA)"
            )
        else:
            low_best = scored[scored["tier_rank"] == 0].sort_values("skill_match", ascending=False).iloc[0]
            need_skills = ", ".join(low_best["missing"][:3]) or "javascript, sql, python"
            outcome = "✗ Unplaced"
            conclusion = f"Remedial Roadmap -> Learn: {need_skills}"

        records.append({
            "entity_id": label,
            "j1_score": tier_scores[0], "j1_qual": tier_qualified[0], "j1_opt": assigned_tier == 0,
            "j2_score": tier_scores[1], "j2_qual": tier_qualified[1], "j2_opt": assigned_tier == 1,
            "j3_score": tier_scores[2], "j3_qual": tier_qualified[2], "j3_opt": assigned_tier == 2,
            "outcome": outcome,
            "conclusion": conclusion,
        })

    return pd.DataFrame(records), tier_benchmarks


def student_company_mapping(students_df, j_offer=default_j_offer):
    """
    Implements Algorithm 2 (`Method Student Company Mapping`) from the Springer
    Article (Section "Method Student Company Mapping", p. 7 & Complexity Analysis, p. 11):
      - Step 2-3: Builds the n x m qualification matrix P[n][m] where P[i][j] = 1
        if student i qualifies for company j (ordered by descending salary).
      - Step 4-5: Given institutional quota `j_offer`, iteratively selects the
        unprocessed student `zIndex` with the fewest qualified matches, locks up to
        `j_offer` highest-salary qualified companies ("*"), and marks remaining slots ("-").
      - Time Complexity: O(n^2 * m), matching Section "Computational Complexity Analysis".
    """
    company_rows = (
        jobs.sort_values("salary_avg_lpa", ascending=False)
        .groupby("company_name", as_index=False)
        .first()
        .sort_values("salary_avg_lpa", ascending=False)
        .reset_index(drop=True)
    )
    n_students = len(students_df)
    m_companies = len(company_rows)
    qual_matrix = np.zeros((n_students, m_companies), dtype=bool)
    score_matrix = np.zeros((n_students, m_companies), dtype=float)

    for i in range(n_students):
        one_student = students_df.iloc[[i]]
        scored = score_jobs_for_group(one_student)
        s_tier = 0 if (cgpa_band_rank[one_student["cgpa_band"].iloc[0]]
                       + skill_names.index(one_student["skill_band"].iloc[0])) <= 1 else (
            1 if (cgpa_band_rank[one_student["cgpa_band"].iloc[0]]
                  + skill_names.index(one_student["skill_band"].iloc[0])) == 2 else 2
        )
        for j, comp_row in company_rows.iterrows():
            comp_name = comp_row["company_name"]
            sub = scored[scored["company_name"] == comp_name].sort_values("skill_match", ascending=False).iloc[0]
            is_qual = (
                int(sub["tier_rank"]) <= s_tier
                and float(sub["skill_match"]) >= min_skill_match
                and float(sub["eligible_fraction"]) >= min_eligible_fraction
            )
            qual_matrix[i, j] = is_qual
            score_matrix[i, j] = float(sub["skill_match"]) * float(comp_row["salary_avg_lpa"])

    # Step 5: Modified Hungarian quota-constrained allocation prioritizing
    # students with the minimum positive qualified count (zIndex) first
    assigned = np.zeros((n_students, m_companies), dtype=bool)
    remaining_working = qual_matrix.copy()
    processed = np.zeros(n_students, dtype=bool)
    company_capacity = np.full(m_companies, max(2, int(np.ceil(n_students * j_offer / max(m_companies, 1)))))

    for _ in range(n_students):
        qual_counts = remaining_working.sum(axis=1)
        candidates = [
            i for i in range(n_students)
            if not processed[i] and qual_counts[i] > 0
        ]
        if not candidates:
            candidates = [i for i in range(n_students) if not processed[i]]
        z_index = min(candidates, key=lambda idx: (qual_counts[idx], -students_df["cgpa"].iloc[idx]))
        processed[z_index] = True

        avail_cols = [j for j in range(m_companies) if remaining_working[z_index, j]]
        if len(avail_cols) < j_offer:
            avail_cols = [j for j in range(m_companies) if qual_matrix[z_index, j]]
        chosen_cols = avail_cols[:j_offer]
        for j in chosen_cols:
            assigned[z_index, j] = True
            company_capacity[j] -= 1
            if company_capacity[j] <= 0:
                remaining_working[:, j] = False

    matrix_rows = []
    for i in range(n_students):
        row_dict = {
            "student_id": students_df["student_id"].iloc[i],
            "group": f"G{students_df['group_number'].iloc[i]}",
        }
        for j, comp_row in company_rows.iterrows():
            col_label = comp_row["company_name"].upper()[:12]
            row_dict[col_label] = "*" if assigned[i, j] else "-"
        matrix_rows.append(row_dict)

    return pd.DataFrame(matrix_rows), company_rows, assigned, qual_matrix


# ==============================================================================
# SECTION 9: UNICODE BOXED CONSOLE PRINTERS & MATPLOTLIB BOXED TABLE RENDERER
# ==============================================================================
def print_table(title, table, wrap_ids=False):
    """
    Renders any DataFrame inside a publication-style Unicode box-drawing grid:
      ┌───────┬───────┐
      │ col_1 │ col_2 │
      ├───────┼───────┤
      │ val_1 │ val_2 │
      └───────┴───────┘
    """
    print("\n" + title)
    if table is None or len(table) == 0:
        print("  ┌───────────────┐")
        print("  │   (no rows)   │")
        print("  └───────────────┘")
        return
    text = table.astype(str).copy()
    if wrap_ids and "student_ids" in text.columns:
        def short(ids):
            parts = ids.split(", ")
            return ids if len(parts) <= 6 else ", ".join(parts[:6]) + f" (+{len(parts) - 6} more)"
        text["student_ids"] = text["student_ids"].apply(short)

    widths = {c: max(len(str(c)), int(text[c].str.len().max())) for c in text.columns}
    cols = list(text.columns)

    top_border = "┌─" + "─┬─".join("─" * widths[c] for c in cols) + "─┐"
    header_row = "│ " + " │ ".join(str(c).ljust(widths[c]) for c in cols) + " │"
    mid_border = "├─" + "─┼─".join("─" * widths[c] for c in cols) + "─┤"
    bot_border = "└─" + "─┴─".join("─" * widths[c] for c in cols) + "─┘"

    print(top_border)
    print(header_row)
    print(mid_border)
    for _, row in text.iterrows():
        print("│ " + " │ ".join(row[c].ljust(widths[c]) for c in cols) + " │")
    print(bot_border)


def format_springer_cell(score, is_qual, is_opt):
    """Formats a cell exactly like Springer Table 2 (attached image): '✓ 65 ✔', '✓ 61', or '45'."""
    if is_qual and is_opt:
        return f"✓ {score:>2d} ✔"
    if is_qual:
        return f"✓ {score:>2d}  "
    return f"  {score:>2d}  "


def print_springer_mapping_box(title, mapping_df, tier_benchmarks):
    """
    Prints the multi-level boxed Student/Group-to-Job Mapping Matrix directly
    mirroring Table 1 & Table 2 (the attached image) from the Springer article:
      - Multi-line headers per Job Tier (Job_k, Cut off, Salary in LPA)
      - Sub-feature header bar: [F1 | F2 | F3] for J1 and [F4 | F5 | F3 | F6] for J2, J3
      - Single checkmark '✓' for qualified cells, double checkmark '✓ score ✔'
        for the optimal salary-maximizing assignment, plus Outcome & Conclusion.
    """
    print("\n" + title)
    if len(mapping_df) == 0:
        return

    w_id = max(12, int(mapping_df["entity_id"].astype(str).str.len().max()) + 2)
    w_job = 21  # fits 'F4 │ F5 │ F3 │ F6' and 'Job2(J2) [Mid Tier]'
    w_out = max(12, int(mapping_df["outcome"].astype(str).str.len().max()) + 2)
    w_conc = max(38, int(mapping_df["conclusion"].astype(str).str.len().max()) + 2)

    j1, j2, j3 = tier_benchmarks[0], tier_benchmarks[1], tier_benchmarks[2]
    sal1 = f"Salary: {j1['salary_lpa']:.1f} LPA"
    sal2 = f"Salary: {j2['salary_lpa']:.1f} LPA"
    sal3 = f"Salary: {j3['salary_lpa']:.1f} LPA"

    top = f"┌─{'─'*w_id}─┬─{'─'*w_job}─┬─{'─'*w_job}─┬─{'─'*w_job}─┬─{'─'*w_out}─┬─{'─'*w_conc}─┐"
    sep_sub = f"│ {' '*w_id} ├─{'─'*w_job}─┼─{'─'*w_job}─┼─{'─'*w_job}─┤ {' '*w_out} │ {' '*w_conc} │"
    mid = f"├─{'─'*w_id}─┼─{'─'*w_job}─┼─{'─'*w_job}─┼─{'─'*w_job}─┼─{'─'*w_out}─┼─{'─'*w_conc}─┤"
    bot = f"└─{'─'*w_id}─┴─{'─'*w_job}─┴─{'─'*w_job}─┴─{'─'*w_job}─┴─{'─'*w_out}─┴─{'─'*w_conc}─┘"

    print(top)
    print(f"│ {'Student /':<{w_id}} │ {j1['job_code'] + ' [Low Tier]':<{w_job}} │ "
          f"{j2['job_code'] + ' [Mid Tier]':<{w_job}} │ {j3['job_code'] + ' [Top Tier]':<{w_job}} │ "
          f"{'Outcome':<{w_out}} │ {'Conclusion / Actionable Roadmap':<{w_conc}} │")
    print(f"│ {'Cohort ID':<{w_id}} │ {'Cut off: ' + str(j1['cutoff']):<{w_job}} │ "
          f"{'Cut off: ' + str(j2['cutoff']):<{w_job}} │ {'Cut off: ' + str(j3['cutoff']):<{w_job}} │ "
          f"{'(Max Weight)':<{w_out}} │ {'(Bipartite Assignment & Guidance)':<{w_conc}} │")
    print(f"│ {'':<{w_id}} │ {sal1:<{w_job}} │ {sal2:<{w_job}} │ {sal3:<{w_job}} │ "
          f"{'':<{w_out}} │ {'':<{w_conc}} │")
    print(sep_sub)
    print(f"│ {'':<{w_id}} │ {'  F1 │  F2 │  F3     ':<{w_job}} │ "
          f"{' F4 │ F5 │ F3 │ F6   ':<{w_job}} │ {' F4 │ F5 │ F3 │ F6   ':<{w_job}} │ "
          f"{'':<{w_out}} │ {'':<{w_conc}} │")
    print(mid)

    for r_idx, row in mapping_df.iterrows():
        c1 = format_springer_cell(int(row["j1_score"]), bool(row["j1_qual"]), bool(row["j1_opt"]))
        c2 = format_springer_cell(int(row["j2_score"]), bool(row["j2_qual"]), bool(row["j2_opt"]))
        c3 = format_springer_cell(int(row["j3_score"]), bool(row["j3_qual"]), bool(row["j3_opt"]))
        print(f"│ {str(row['entity_id']):<{w_id}} │ {c1:^{w_job}} │ {c2:^{w_job}} │ {c3:^{w_job}} │ "
              f"{str(row['outcome']):<{w_out}} │ {str(row['conclusion']):<{w_conc}} │")
        if r_idx < len(mapping_df) - 1:
            print(mid)
    print(bot)
    print("  Legend: '✓ <score>' = Qualified (clears job cutoff & eligibility); "
          "'✓ <score> ✔' = Optimal Salary-Maximizing Assignment (Modified Hungarian)")


def save_boxed_mapping_figure(group_map_df, student_map_df, tier_benchmarks, file_path):
    """
    Renders a 300-DPI publication figure (`result_boxed_mapping_table.png`)
    visually replicating the exact boxed layout of Table 2 from the Springer article
    (and the user's attached reference image), featuring:
      - Multi-line Job headers (Job_1(J1), Cut off, Salary in LPA)
      - Sub-column feature cells (F1 | F2 | F3 and F4 | F5 | F3 | F6)
      - Black checkmark (✓) before the score for qualified cells
      - Large bold green checkmark (✔) after the score for the optimal assigned job
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(16.0, 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    datasets = [
        (axes[0], student_map_df.head(9),
         "(a) Individual Student-to-Job Mapping (Step 4 — Springer Table 2 Style)", False),
        (axes[1], group_map_df,
         "(b) Cohort-Level Mapping & Actionable Outcome (9 Skill-Stratified Groups)", True),
    ]

    for ax, df_plot, panel_title, show_outcome in datasets:
        ax.set_xlim(0, 100)
        n_rows = len(df_plot)
        header_h = 18.0
        row_h = (82.0 - 4.0) / max(n_rows, 1)
        ax.set_ylim(0, 100)
        ax.axis("off")
        ax.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=10, color="#1e293b")

        if not show_outcome:
            col_x = [0.0, 20.0, 46.5, 73.2, 100.0]
        else:
            col_x = [0.0, 15.0, 36.0, 57.0, 78.0, 100.0]

        y_top = 98.0
        y_sub = y_top - 12.5
        y_body_top = y_top - header_h

        # Header background fill
        ax.add_patch(Rectangle(
            (col_x[0], y_body_top), col_x[-1] - col_x[0], header_h,
            facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2,
        ))
        ax.add_patch(Rectangle(
            (col_x[1], y_body_top), col_x[4] - col_x[1], y_sub - y_body_top,
            facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=0.9,
        ))

        # Top-left ID header
        ax.text(
            (col_x[0] + col_x[1]) / 2, (y_top + y_body_top) / 2,
            "Cohort" if show_outcome else "Student",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a",
        )

        # 3 Job Tier column headers + F1..F6 sub-headers
        sub_labels = [["F1", "F2", "F3"], ["F4", "F5", "F3", "F6"], ["F4", "F5", "F3", "F6"]]
        for j_idx in range(3):
            x0, x1 = col_x[j_idx + 1], col_x[j_idx + 2]
            jb = tier_benchmarks[j_idx]
            header_text = (
                f"Job$_{j_idx + 1}$(J$_{j_idx + 1}$)\n"
                f"Cut off: {jb['cutoff']}\n"
                f"Salary: {jb['salary_lpa']:.1f} LPA"
            )
            ax.text(
                x0 + 1.5, (y_top + y_sub) / 2, header_text,
                ha="left", va="center", fontsize=8.5, fontweight="semibold",
                color="#0f172a", linespacing=1.22,
            )
            f_list = sub_labels[j_idx]
            sub_w = (x1 - x0) / len(f_list)
            for f_i, f_name in enumerate(f_list):
                sx0 = x0 + f_i * sub_w
                ax.plot([sx0, sx0], [y_body_top, y_sub], color="#94a3b8", linewidth=0.8)
                ax.text(
                    sx0 + sub_w / 2, (y_sub + y_body_top) / 2, f_name,
                    ha="center", va="center", fontsize=8.0, color="#334155", fontweight="medium",
                )

        if show_outcome:
            x0, x1 = col_x[4], col_x[5]
            ax.text(
                (x0 + x1) / 2, (y_top + y_body_top) / 2,
                "Assigned Outcome\n& Tier Package",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f172a",
            )

        # Draw data rows
        for r_idx, (_, row) in enumerate(df_plot.iterrows()):
            y1 = y_body_top - r_idx * row_h
            y0 = y1 - row_h
            bg = "#ffffff" if r_idx % 2 == 0 else "#f8fafc"
            ax.add_patch(Rectangle(
                (col_x[0], y0), col_x[-1] - col_x[0], row_h,
                facecolor=bg, edgecolor="#cbd5e1", linewidth=0.9,
            ))

            # Entity ID cell
            ax.text(
                (col_x[0] + col_x[1]) / 2, (y0 + y1) / 2, str(row["entity_id"]),
                ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0f172a",
            )

            # 3 Job cells with ✓ (qualified) and green ✔ (optimal salary match)
            for j_idx, prefix in enumerate(["j1", "j2", "j3"]):
                x0, x1 = col_x[j_idx + 1], col_x[j_idx + 2]
                xc = (x0 + x1) / 2
                yc = (y0 + y1) / 2
                score_val = int(row[f"{prefix}_score"])
                is_qual = bool(row[f"{prefix}_qual"])
                is_opt = bool(row[f"{prefix}_opt"])

                if is_qual and is_opt:
                    ax.add_patch(Rectangle(
                        (x0 + 0.4, y0 + 0.4), (x1 - x0) - 0.8, row_h - 0.8,
                        facecolor="#ecfdf5", edgecolor="#10b981", linewidth=1.2,
                    ))
                    ax.text(xc - 4.8, yc, "✓", ha="center", va="center",
                            fontsize=11.0, fontweight="bold", color="#0f172a")
                    ax.text(xc - 0.2, yc, f"{score_val}", ha="center", va="center",
                            fontsize=9.8, fontweight="bold", color="#0f172a")
                    ax.text(xc + 5.0, yc, "✓", ha="center", va="center",
                            fontsize=15.0, fontweight="bold", color="#15803d")
                elif is_qual:
                    ax.text(xc - 3.6, yc, "✓", ha="center", va="center",
                            fontsize=10.8, fontweight="bold", color="#0f172a")
                    ax.text(xc + 1.4, yc, f"{score_val}", ha="center", va="center",
                            fontsize=9.5, color="#1e293b")
                else:
                    ax.text(xc, yc, f"{score_val}", ha="center", va="center",
                            fontsize=9.2, color="#64748b")

            if show_outcome:
                x0, x1 = col_x[4], col_x[5]
                out_str = str(row["outcome"])
                out_color = "#15803d" if out_str.startswith("✓") else "#b91c1c"
                ax.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, out_str,
                    ha="center", va="center", fontsize=8.5, fontweight="bold", color=out_color,
                )

        # Vertical grid lines
        for x_val in col_x:
            ax.plot([x_val, x_val], [y_body_top - n_rows * row_h, y_top],
                    color="#94a3b8", linewidth=1.1)

    fig.suptitle(
        "Extended Bipartite Student-to-Job Capability & Max-Salary Mapping Matrix\n"
        "[ Left '✓' = Qualified (Clears Tier Cutoff & Eligibility)   |   "
        "Right Green '✓' = Optimal Max-Salary Assignment (Eq. 1) ]",
        fontsize=11.5, fontweight="bold", y=0.97, color="#0f172a",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.87, wspace=0.10)
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# SECTION 10: MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    print(f"\njob roles loaded: {len(jobs)}")
    print("job tiers found by clustering:")
    for rank, tier in enumerate(tier_names):
        part = jobs[jobs["tier_rank"] == rank]
        print(f"  {tier:<4} tier: {len(part):>3} roles, salary {part['salary_avg_lpa'].min():.1f} "
              f"to {part['salary_avg_lpa'].max():.1f} lpa (avg {part['salary_avg_lpa'].mean():.1f})")
    print(f"job silhouette score: {silhouette_score(job_scaled, jobs['job_cluster']):.2f}")

    summaries, apply_tables, upgrade_tables, learn_rows = [], [], [], []
    for record in group_records:
        summary, apply_table, upgrade_table, learn_row = analyse_group(record)
        summaries.append(summary)
        if apply_table is not None:
            apply_tables.append(apply_table)
            upgrade_tables.append(upgrade_table)
            learn_rows.append(learn_row)

    summary_table = pd.DataFrame(summaries)
    apply_all = pd.concat(apply_tables, ignore_index=True) if apply_tables else pd.DataFrame()
    upgrade_all = pd.concat(upgrade_tables, ignore_index=True) if upgrade_tables else pd.DataFrame()
    learn_table = pd.DataFrame(learn_rows)

    filled = int((summary_table["students"] > 0).sum())
    print_table(
        f"table 1: student groups ({filled} of {len(group_records)} groups have students in this batch)",
        summary_table, wrap_ids=True,
    )
    print_table(
        f"table 2: roles each group can apply for now (top {roles_to_show} roles, skill match is "
        f"an approximate average for the group, salary is the range over the listed companies)",
        apply_all,
    )
    print_table("table 3: upgrade path, roles to aim for in the next achievable tier", upgrade_all)
    print_table("table 4: skills to learn (number of target roles that need the skill)", learn_table)

    # Build & print Springer Table 1 / Table 2 Boxed Mapping Matrix (Cohort & Sample Students)
    active_group_dfs = [
        checked_students.loc[rec["member_index"]]
        for rec in group_records if len(rec["member_index"]) > 0
    ]
    active_group_labels = [
        f"G{rec['group_number']} ({rec['cgpa_band'][:1].upper()}/{rec['skill_band'][:1].upper()})"
        for rec in group_records if len(rec["member_index"]) > 0
    ]
    group_map_df, tier_benchmarks = compute_springer_tier_mapping(
        active_group_dfs, active_group_labels, is_group=True
    )

    rep_indices = [rec["member_index"][0] for rec in group_records if len(rec["member_index"]) > 0]
    rep_students_df = checked_students.loc[rep_indices].reset_index(drop=True)
    rep_labels = [
        f"S{i + 1} ({row['student_id']})"
        for i, (_, row) in enumerate(rep_students_df.iterrows())
    ]
    student_map_df, _ = compute_springer_tier_mapping(
        rep_students_df, rep_labels, is_group=False
    )

    print_springer_mapping_box(
        "table 5a: boxed cohort-to-job capability & max-salary mapping matrix (springer table 1 & 2 style)",
        group_map_df, tier_benchmarks,
    )
    print_springer_mapping_box(
        "table 5b: boxed individual student-to-job mapping matrix (representative student per group)",
        student_map_df, tier_benchmarks,
    )

    # Build & print Table 6: Modified Hungarian Quota-Constrained Interview Matrix (* / -)
    interview_df, _, _, _ = student_company_mapping(rep_students_df, j_offer=default_j_offer)
    print_table(
        f"table 6: modified hungarian campus interview participation matrix "
        f"(jOffer = {default_j_offer} offers/student; '*' = appear, '-' = skip)",
        interview_df,
    )

    if save_tables:
        summary_table.to_csv(output_summary_file, index=False)
        apply_all.to_csv(output_apply_file, index=False)
        upgrade_all.to_csv(output_upgrade_file, index=False)
        learn_table.to_csv(output_learn_file, index=False)
        group_map_df.to_csv(output_mapping_file, index=False)
        interview_df.to_csv(output_interview_file, index=False)
        save_boxed_mapping_figure(group_map_df, student_map_df, tier_benchmarks, output_boxed_figure)
        print(f"\nsaved: {output_summary_file}, {output_apply_file}, "
              f"{output_upgrade_file}, {output_learn_file}, "
              f"{output_mapping_file}, {output_interview_file}, {output_boxed_figure}")

    print("\ndone")


if __name__ == "__main__":
    main()