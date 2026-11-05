# career_matcher.py
# ==============================================================================
# RESEARCH-GRADE PROTOTYPE (EXTENDED SPRINGER METHODOLOGY):
# "A Comprehensive Analysis on the Skill-Set of the Students to Improve Campus
#  Drive Using Machine Learning Approaches"
#
# CORE METHODOLOGY & SCIENTIFIC UPGRADES:
#   1. Multi-Schema Dataset Integration & Provenance-Aware Preprocessing:
#      Supports canonical project CSVs as well as auto-detection of the three
#      Kaggle datasets:
#        - Dataset 1 (brands_data.csv): ankushnarwade/indian-fresher-salary-and-skill-trends-2025
#        - Dataset 2 (student_data.csv): vishwajeetchandole/indian-engineering-placement-dataset-2026
#        - Dataset 3 (check_student.csv): shambhurajejagadale/student-placement-and-career-success-dataset-2026
#      Explicitly distinguishes actual source-provided requirements, missing
#      requirements, optional requirements, and derived features without
#      fabricating unavailable attributes (such as missing min_cgpa).
#   2. Leak-Free Common-Feature Alignment:
#      Fits all data-dependent transformations (imputation medians, communication
#      quantiles, StandardScaler, and K-Means centroids) strictly on the training
#      dataset (student_data.csv) and applies them to unseen testing records
#      (check_student.csv) after verifying zero student_id overlap.
#   3. Hierarchical Academic + Skill Profiling (2-Level Architecture):
#      - Level 1 (Rule-Based Academic Filtering): Low (<6.0), Mid ([6.0, 8.0)),
#        and Top (>=8.0) CGPA categories.
#      - Level 2 (Within-Band K-Means Clustering): Standardized K-Means (K=3,
#        fixed seed) inside each CGPA band to identify Low, Mid, and Good
#        skill-readiness sub-clusters ordered by centroid feature strength.
#   4. Relative Cluster-Margin Score:
#      Computes M(x) = 1 - d_1(x) / d_2(x) in standardized centroid space with
#      explicit handling for zero-distance and equal-distance boundary cases.
#   5. Salary-Weighted Job-Tier Clustering:
#      Partitions roles into Low, Mid, and Top remuneration tiers using
#      available job attributes without fabricating missing cutoffs.
#   6. Multi-Constraint Eligibility, Skill-Gap & Upgrade Recommendations:
#      Evaluates only genuinely available eligibility gates, adjusts skill-match
#      weights when preferred_skills is absent, and generates "Apply Now",
#      "Upgrade Path", and "Skills to Learn" tables.
#   7. Mathematically Valid Capacity-Constrained Student-Company Allocation:
#      Replaces the misnamed greedy heuristic with an exact Mixed-Integer Linear
#      Programming (MILP) / Min-Cost Flow b-matching assignment that maximizes
#      student placement coverage and salary-weighted utility subject to hard
#      eligibility, student offer quotas (j_offer), and company slot capacities.
# ==============================================================================

import glob
import logging
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp
from scipy.spatial.distance import cdist

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        """Z-score standardization: z = (x - mean) / std."""
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


# ==============================================================================
# SECTION 1: OUTPUT DIRECTORY STRUCTURE, LOGGING & CONFIGURATION
# ==============================================================================
OUTPUT_ROOT = "output"
OUTPUT_EVAL_DIR = os.path.join(OUTPUT_ROOT, "evaluation")
OUTPUT_CLUSTER_DIR = os.path.join(OUTPUT_ROOT, "clustering")
OUTPUT_REPORT_DIR = os.path.join(OUTPUT_ROOT, "reports")
OUTPUT_TABLE_DIR = os.path.join(OUTPUT_ROOT, "tables")
OUTPUT_LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")


def ensure_output_directories():
    """Creates the required output/ directory tree without writing to project root."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for sub in (OUTPUT_EVAL_DIR, OUTPUT_CLUSTER_DIR, OUTPUT_REPORT_DIR, OUTPUT_TABLE_DIR, OUTPUT_LOG_DIR):
        os.makedirs(sub, exist_ok=True)


ensure_output_directories()

LOG_FILE_PATH = os.path.join(OUTPUT_LOG_DIR, "execution.log")
logger = logging.getLogger("career_matcher")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)


def log_info(msg):
    logger.info(msg)


# Input dataset paths
job_file_name = "brands_data.csv"
train_student_file_name = "student_data.csv"
check_student_file_name = "check_student.csv"

# Output file paths (strictly inside output/ subdirectories)
save_tables = True
GENERATE_CLUSTERING_DISTRIBUTION_PLOTS = True
output_summary_file = os.path.join(OUTPUT_TABLE_DIR, "student_cluster_summary.csv")
output_matching_file = os.path.join(OUTPUT_TABLE_DIR, "job_matching_results.csv")
output_mapping_file = os.path.join(OUTPUT_TABLE_DIR, "student_company_mapping.csv")
output_preprocessing_report = os.path.join(OUTPUT_REPORT_DIR, "dataset_preprocessing_report.txt")

# Reproducibility & methodology constants
random_seed = 42
job_tiers = 3
skill_subclusters = 3
min_students_to_cluster = 10
tier_names = ["low", "mid", "top"]
skill_names = ["low", "mid", "good"]
group_tier_names = ["beginner", "mid", "top"]

min_skill_match = 0.50
min_eligible_fraction = 0.50
roles_to_show = 5
default_j_offer = 2
tier_cutoffs = {0: 50, 1: 60, 2: 75}

soft_skills = {"communication", "problem solving"}

skill_aliases = {
    "ml": "machine learning", "dl": "deep learning", "ai": "machine learning",
    "artificial intelligence": "machine learning",
    "oop": "object oriented programming", "oops": "object oriented programming",
    "dsa": "data structures", "ds": "data structures", "algo": "algorithms",
    "js": "javascript", "ts": "typescript",
    "node": "node.js", "nodejs": "node.js", "reactjs": "react", "react.js": "react",
    "k8s": "kubernetes", "cpp": "c++", "csharp": "c#", "c sharp": "c#",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "sklearn": "scikit-learn", "scikitlearn": "scikit-learn",
    "powerbi": "power bi", "ms excel": "excel",
    "sql server": "sql", "mysql": "sql", "postgres": "sql", "postgresql": "sql",
    "mongodb": "nosql", "aws cloud": "aws", "amazon web services": "aws",
}

empty_words = {"na", "n/a", "none", "no", "nil", "-", "null", "nan", ""}

coding_skill_threshold = 50
skills_from_coding = ["data structures", "algorithms"]

communication_scores = {
    "low": 1, "poor": 1, "weak": 1, "beginner": 1,
    "mid": 2, "medium": 2, "average": 2, "moderate": 2, "intermediate": 2,
    "good": 3, "high": 3, "strong": 3, "excellent": 3, "advanced": 3,
}

field_limits = {
    "cgpa": (0.0, 10.0),
    "coding_questions_solved": (0.0, 100000.0),
    "projects_count": (0.0, 100.0),
    "experience_months": (0.0, 360.0),
    "certifications_count": (0.0, 50.0),
}

student_column_aliases = {
    "student_id": [
        "student_id", "studentid", "id", "slno", "serialnumber", "rollno", "roll_no", "candidate_id"
    ],
    "degree": [
        "degree", "degreet", "course", "qualification", "program"
    ],
    "branch": [
        "branch", "department", "stream", "specialisation", "specialization", "major", "field_of_study"
    ],
    "cgpa": [
        "cgpa", "degreep", "gpa", "academic_cgpa", "cumulative_gpa", "ug_cgpa"
    ],
    "coding_language": [
        "coding_language", "knowncodinglanguage", "primarylanguage", "primary_language",
        "primarycodinglanguage", "language", "programming_language", "preferred_language"
    ],
    "coding_questions_solved": [
        "coding_questions_solved", "approxcodingquestionssolvedacrossallsites", "questionssolved",
        "dsa_problems_solved", "dsaproblemssolved", "coding_problems_solved", "codingproblemssolved",
        "leetcode_problems_solved", "leetcode_questions", "dsa_count", "dsa_questions_solved",
        "coding_skill_score"
    ],
    "technical_skills": [
        "technical_skills", "technicalskills", "skills", "key_skills", "tech_skills",
        "programming_skills", "skill_set", "tech_stack"
    ],
    "projects_count": [
        "projects_count", "numberofprojects", "projects", "project_count", "total_projects",
        "academic_projects", "major_projects"
    ],
    "development_projects_count": [
        "development_projects_count", "developmentprojectscount", "dev_projects", "web_projects"
    ],
    "ai_ml_projects": [
        "ai_ml_projects", "aimlprojects", "ml_projects", "ai_projects"
    ],
    "experience_months": [
        "experience_months", "internshipworkexperienceinmonths", "experience",
        "internship_months", "work_experience_months", "total_experience_months"
    ],
    "internships_completed": [
        "internships_completed", "internshipscompleted", "internships_count", "internshipscount",
        "internships", "number_of_internships", "internship_count", "internship_experience"
    ],
    "certifications_count": [
        "certifications_count", "numberofcertifications", "certifications",
        "certification_count", "courses_completed"
    ],
    "communication": [
        "communication", "communicationskill", "communication_skills", "communicationskills",
        "communication_score", "communicationscore", "communication_skill_score", "soft_skills_score"
    ],
    "placement_status": [
        "placement_status", "placementstatus", "placed", "status", "is_placed",
        "placement_outcome", "hiring_status"
    ],
    "salary_lpa": [
        "salary_lpa", "salarylpa", "salary_package_lpa", "salarypackagelpa",
        "annual_salary_lpa", "package_lpa", "ctc_lpa", "offered_salary_lpa", "salary"
    ],
    "github_repos": [
        "github_repos", "githubrepos", "github_contributions", "githubcontributions",
        "open_source_contributions", "github_projects"
    ],
    "hackathons_participated": [
        "hackathons_participated", "hackathonsparticipated", "hackathons", "hackathon_count"
    ],
    "aptitude_score": [
        "aptitude_score", "aptitudescore", "aptitude_test_score", "logical_reasoning_score"
    ],
    "resume_score": [
        "resume_score", "resumescore"
    ],
    "mock_interview_score": [
        "mock_interview_score", "mockinterviewscore", "interview_score"
    ],
    "backlog_history": [
        "backlog_history", "backloghistory", "backlogs", "active_backlogs", "num_backlogs"
    ],
    "attendance_percentage": [
        "attendance_percentage", "attendancepercentage", "attendance"
    ],
    "college_tier": [
        "college_tier", "collegetier"
    ],
    "ai_tool_usage_frequency": [
        "ai_tool_usage_frequency", "aitoolusagefrequency"
    ],
    "prompt_engineering_skill": [
        "prompt_engineering_skill", "promptengineeringskill"
    ],
    "adaptability_score": [
        "adaptability_score", "adaptabilityscore"
    ],
    "leadership_score": [
        "leadership_score", "leadershipscore", "leadership_experience", "extracurricular_score"
    ],
}

column_aliases = student_column_aliases

job_column_aliases = {
    "job_id": ["job_id", "jobid", "id", "posting_id"],
    "company_name": ["company_name", "companyname", "company", "employer", "brand", "organization"],
    "job_role": ["job_role", "jobrole", "role", "job_title", "title", "position", "designation"],
    "required_skills": [
        "required_skills", "requiredskills", "skills", "technical_skills", "mandatory_skills", "key_skills"
    ],
    "primary_skill": ["primary_skill", "primaryskill", "main_skill", "core_skill"],
    "preferred_skills": [
        "preferred_skills", "preferredskills", "optional_skills", "bonus_skills", "good_to_have"
    ],
    "degrees_accepted": [
        "degrees_accepted", "degreesaccepted", "degree", "degrees", "qualification", "eligible_degrees"
    ],
    "experience_required": [
        "experience_required", "experiencerequired", "experience", "min_experience", "exp_required"
    ],
    "internship_experience": [
        "internship_experience", "internshipexperience", "internship_required"
    ],
    "salary_avg_lpa": [
        "salary_avg_lpa", "salaryavglpa", "salary_lpa", "salarylpa", "salary",
        "package_lpa", "ctc_lpa", "annual_salary_lpa"
    ],
    "stipend_if_intern": ["stipend_if_intern", "stipendifintern", "stipend", "monthly_stipend"],
    "min_cgpa": ["min_cgpa", "mincgpa", "cgpa_cutoff", "cutoff_cgpa", "minimum_cgpa"],
    "coding_test_rounds": ["coding_test_rounds", "codingtestrounds", "coding_rounds"],
    "coding_test": ["coding_test", "codingtest"],
    "aptitude_test": ["aptitude_test", "aptitudetest"],
    "selection_rounds": ["selection_rounds", "selectionrounds", "total_rounds"],
    "interview_count": ["interview_count", "interviewcount", "interview_rounds"],
    "company_type": ["company_type", "companytype", "sector"],
    "city": ["city", "job_city", "location"],
    "state": ["state", "job_state"],
    "graduation_year": ["graduation_year", "graduationyear", "batch_year"],
    "remote": ["remote", "is_remote", "work_mode"],
    "work_type": ["work_type", "worktype", "employment_type", "job_type"],
    "offer_made": ["offer_made", "offermade", "offer_status"],
    "job_capacity": ["job_capacity", "capacity", "openings", "vacancies", "slots"],
}


# ==============================================================================
# SECTION 2: KAGGLE FILE AUTO-DETECTION & REALISTIC NON-UNIFORM FALLBACK DATA
# ==============================================================================
def _is_legacy_uniform_modulo_file(file_path):
    """
    Detects whether an existing student_data.csv or check_student.csv on disk was
    created by the legacy `b = idx % 3` / `st = (idx // 3) % 3` loop (which forced
    an artificial 120/120/120 [40/40/40] or 15/15/15 [5/5/5] uniform split).
    Never touches real user/Kaggle datasets.
    """
    if not os.path.exists(file_path):
        return False
    try:
        df = pd.read_csv(file_path)
        if "student_id" not in df.columns or "cgpa" not in df.columns:
            return False
        sids = df["student_id"].astype(str).tolist()
        cgpas = pd.to_numeric(df["cgpa"], errors="coerce")
        n_low = int((cgpas < 6.0).sum())
        n_mid = int(((cgpas >= 6.0) & (cgpas < 8.0)).sum())
        n_top = int((cgpas >= 8.0).sum())
        if len(df) == 360 and sids[0] == "TR-0001" and n_low == 120 and n_mid == 120 and n_top == 120:
            return True
        if len(df) == 45 and sids[0] == "S-4461" and n_low == 15 and n_mid == 15 and n_top == 15:
            return True
    except Exception:
        return False
    return False


def _auto_link_kaggle_csvs():
    """
    If the user downloaded the Kaggle CSVs under their native Kaggle filenames
    (e.g. Indian_Fresher_Salary_Skills_2025.csv) in the working directory,
    automatically copies them to brands_data.csv / student_data.csv / check_student.csv.
    """
    patterns = [
        (job_file_name, ["*Fresher*Salary*Skills*.csv", "*fresher*salary*skill*.csv"]),
        (train_student_file_name, ["*engineering*placement*2026*.csv", "*Indian_Engineering_Placement*.csv"]),
        (check_student_file_name, ["*career*success*2026*.csv", "*Student_Placement*Career_Success*.csv"]),
    ]
    for target_name, globs_list in patterns:
        if os.path.exists(target_name) and not _is_legacy_uniform_modulo_file(target_name):
            continue
        for pat in globs_list:
            matches = [m for m in glob.glob(pat) if os.path.basename(m) != target_name]
            if matches:
                shutil.copy(matches[0], target_name)
                log_info(f"Auto-linked Kaggle dataset '{matches[0]}' -> '{target_name}'")
                break


def ensure_default_datasets():
    """
    Ensures brands_data.csv, student_data.csv, and check_student.csv exist.
    - Never overwrites user-supplied or Kaggle CSV files.
    - If a legacy uniform modulo file (120/120/120 -> 40/40/40) from the older script
      is detected, replaces it with a realistic, non-uniform campus distribution
      (skewed toward Mid CGPA 6.0-8.0, with natural unequal cluster sizes).
    """
    _auto_link_kaggle_csvs()
    datasets_status = {}
    rng = np.random.RandomState(random_seed)

    if os.path.exists(job_file_name):
        datasets_status["jobs_source"] = f"Existing file ({job_file_name})"
    else:
        datasets_status["jobs_source"] = f"Generated synthetic fallback ({job_file_name})"
        log_info(f"{job_file_name} not found in root; generating fallback benchmark dataset.")
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

    if os.path.exists(train_student_file_name) and not _is_legacy_uniform_modulo_file(train_student_file_name):
        datasets_status["train_source"] = f"Existing file ({train_student_file_name})"
    else:
        datasets_status["train_source"] = f"Generated realistic non-uniform synthetic fallback ({train_student_file_name})"
        log_info(f"Generating realistic non-uniform training dataset ({train_student_file_name}).")
        pool_low = ["html", "css", "sql", "c++", "java", "python", "manual testing"]
        pool_mid = ["python", "java", "sql", "javascript", "react", "node.js", "git", "oop", "html", "css", "linux"]
        pool_high = ["python", "c++", "java", "sql", "ml", "dl", "sklearn", "docker", "aws", "react", "node.js", "system design", "kubernetes", "ci/cd", "power bi"]

        # Realistic Indian engineering campus distribution:
        #   - Academic CGPA bands are NOT 1/3 each: Mid CGPA [6.0, 8.0) is largest (~52%),
        #     Top CGPA (>= 8.0) is ~29%, and Low CGPA (< 6.0) is ~19%.
        #   - Within each CGPA band, skill readiness correlates with academic preparation
        #     so cluster sizes vary naturally rather than being flat 40/40/40 bars.
        band_probs = [0.19, 0.52, 0.29]
        skill_probs_by_band = {
            0: [0.52, 0.33, 0.15],  # Low CGPA: mostly low/mid skills, fewer top coders
            1: [0.28, 0.46, 0.26],  # Mid CGPA: bell-shaped around mid skill readiness
            2: [0.16, 0.38, 0.46],  # Top CGPA: skewed toward mid/good skill readiness
        }
        train_rows = []
        n_train_total = 420
        for idx in range(1, n_train_total + 1):
            b = int(rng.choice([0, 1, 2], p=band_probs))
            if b == 0:
                cgpa = float(np.round(np.clip(rng.normal(5.45, 0.32), 4.60, 5.98), 2))
            elif b == 1:
                cgpa = float(np.round(np.clip(rng.normal(7.05, 0.52), 6.00, 7.98), 2))
            else:
                cgpa = float(np.round(np.clip(rng.normal(8.68, 0.45), 8.00, 9.88), 2))

            st = int(rng.choice([0, 1, 2], p=skill_probs_by_band[b]))
            if st == 0:
                n_sk, pool = rng.randint(1, 4), pool_low
                solved = int(np.clip(rng.normal(24, 12), 4, 48))
                proj = int(rng.choice([0, 1, 2], p=[0.35, 0.50, 0.15]))
                exp = int(rng.choice([0, 0, 1, 2], p=[0.55, 0.25, 0.12, 0.08]))
                cert = int(rng.choice([0, 1, 2], p=[0.50, 0.38, 0.12]))
                comm = rng.choice(["low", "mid"], p=[0.62, 0.38])
                github = int(rng.randint(0, 3))
                apt = int(np.clip(rng.normal(52, 9), 35, 72))
                backlogs = int(rng.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.18, 0.07]))
            elif st == 1:
                n_sk, pool = rng.randint(4, 7), pool_mid
                solved = int(np.clip(rng.normal(125, 38), 55, 210))
                proj = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.40, 0.25, 0.10]))
                exp = int(rng.choice([0, 2, 3, 6], p=[0.20, 0.35, 0.30, 0.15]))
                cert = int(rng.choice([1, 2, 3, 4], p=[0.20, 0.42, 0.26, 0.12]))
                comm = rng.choice(["mid", "good"], p=[0.64, 0.36])
                github = int(rng.randint(2, 7))
                apt = int(np.clip(rng.normal(69, 8), 50, 88))
                backlogs = int(rng.choice([0, 1], p=[0.82, 0.18]))
            else:
                n_sk, pool = rng.randint(6, 11), pool_high
                solved = int(np.clip(rng.normal(340, 85), 195, 580))
                proj = int(rng.choice([4, 5, 6, 7, 8], p=[0.22, 0.32, 0.24, 0.14, 0.08]))
                exp = int(rng.choice([3, 6, 9, 12], p=[0.25, 0.42, 0.20, 0.13]))
                cert = int(rng.choice([3, 4, 5, 6, 7], p=[0.25, 0.32, 0.23, 0.12, 0.08]))
                comm = rng.choice(["mid", "good"], p=[0.22, 0.78])
                github = int(rng.randint(5, 14))
                apt = int(np.clip(rng.normal(84, 7), 66, 99))
                backlogs = 0

            skills = ", ".join(rng.choice(pool, size=min(n_sk, len(pool)), replace=False))
            comm_num = communication_scores[comm]
            logit = (
                -3.3
                + 0.42 * (cgpa - 6.0)
                + 0.0085 * min(solved, 320)
                + 0.34 * proj
                + 0.17 * exp
                + 0.24 * cert
                + 0.55 * (comm_num - 1)
                + 0.026 * (apt - 50)
                - 0.68 * backlogs
                + rng.normal(0, 0.80)
            )
            prob_placed = 1.0 / (1.0 + np.exp(-logit))
            is_placed = int(rng.rand() < prob_placed)
            placement_str = "Placed" if is_placed == 1 else "Not Placed"
            sal_lpa = np.round(
                max(3.2, 2.5 + 0.65 * cgpa + 0.015 * min(solved, 400) + 0.45 * proj + 0.22 * exp + rng.normal(0, 1.1)),
                2
            ) if is_placed == 1 else 0.0
            train_rows.append([
                f"TR-{idx:04d}", rng.choice(["B.Tech", "B.E", "M.Tech", "MCA"], p=[0.55, 0.25, 0.10, 0.10]),
                rng.choice(["CSE", "IT", "ECE", "EE"], p=[0.42, 0.28, 0.18, 0.12]), cgpa,
                rng.choice(["Python", "Java", "C++", "JavaScript"], p=[0.42, 0.30, 0.18, 0.10]),
                solved, skills, proj, exp, cert, comm, github, apt, backlogs, placement_str, sal_lpa
            ])
        pd.DataFrame(train_rows, columns=[
            "student_id", "degree", "branch", "cgpa", "coding_language",
            "coding_questions_solved", "technical_skills", "projects_count",
            "experience_months", "certifications_count", "communication",
            "github_repos", "aptitude_score", "backlog_history",
            "placement_status", "salary_lpa"
        ]).to_csv(train_student_file_name, index=False)

    if os.path.exists(check_student_file_name) and not _is_legacy_uniform_modulo_file(check_student_file_name):
        datasets_status["check_source"] = f"Existing file ({check_student_file_name})"
    else:
        datasets_status["check_source"] = f"Generated realistic non-uniform synthetic fallback ({check_student_file_name})"
        log_info(f"Generating realistic non-uniform testing dataset ({check_student_file_name}).")
        # Realistic non-uniform cohort counts across the 9 (CGPA, Skill) groups (total = 60 unseen students)
        cohort_specs = [
            (0, 0, 6),   # G1: Low CGPA, Low Skill (6 students)
            (0, 1, 4),   # G2: Low CGPA, Mid Skill (4 students)
            (0, 2, 2),   # G3: Low CGPA, Good Skill (2 students)
            (1, 0, 8),   # G4: Mid CGPA, Low Skill (8 students)
            (1, 1, 14),  # G5: Mid CGPA, Mid Skill (14 students - largest typical campus group)
            (1, 2, 8),   # G6: Mid CGPA, Good Skill (8 students)
            (2, 0, 3),   # G7: Top CGPA, Low Skill (3 students)
            (2, 1, 7),   # G8: Top CGPA, Mid Skill (7 students)
            (2, 2, 8),   # G9: Top CGPA, Good Skill (8 students)
        ]
        check_rows = []
        s_counter = 1
        for b, st, count in cohort_specs:
            for _ in range(count):
                if b == 0:
                    cgpa = float(np.round(rng.uniform(5.05, 5.92), 2))
                elif b == 1:
                    cgpa = float(np.round(rng.uniform(6.15, 7.88), 2))
                else:
                    cgpa = float(np.round(rng.uniform(8.10, 9.65), 2))

                if st == 0 and b == 0:
                    skills, lang, solved, proj, exp, cert, comm = "html, css", "C", rng.randint(6, 22), 0, 0, 0, "low"
                elif st == 0:
                    skills, lang, solved, proj, exp, cert, comm = "html, css, sql", "Python", rng.randint(15, 42), int(rng.choice([0, 1])), 0, int(rng.choice([0, 1])), "mid"
                elif st == 1:
                    skills, lang, solved, proj, exp, cert, comm = "python, java, sql, javascript, react, oop, git", "Python", rng.randint(85, 175), int(rng.choice([2, 3, 4])), int(rng.choice([2, 3, 4])), int(rng.choice([2, 3])), "mid"
                else:
                    skills, lang, solved, proj, exp, cert, comm = "c++, python, java, sql, ml, dl, sklearn, docker, aws, system design, oop", "Python", rng.randint(260, 490), int(rng.choice([5, 6, 7])), int(rng.choice([6, 9, 12])), int(rng.choice([4, 5, 6])), "good"
                check_rows.append([
                    f"S-{4460 + s_counter}", rng.choice(["B.Tech", "B.E", "MCA"], p=[0.70, 0.20, 0.10]),
                    rng.choice(["CSE", "IT", "ECE"], p=[0.55, 0.30, 0.15]),
                    cgpa, lang, solved, skills, proj, exp, cert, comm
                ])
                s_counter += 1
        pd.DataFrame(check_rows, columns=[
            "student_id", "degree", "branch", "cgpa", "coding_language",
            "coding_questions_solved", "technical_skills", "projects_count",
            "experience_months", "certifications_count", "communication",
        ]).to_csv(check_student_file_name, index=False)

    return datasets_status


# ==============================================================================
# SECTION 3: DATA CLEANING, NORMALIZATION & VALIDATION FUNCTIONS
# ==============================================================================
def normalize_header(text):
    """Strips punctuation/spaces and lowercases column headers for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def rename_with_aliases(raw_df, alias_dict):
    """
    Maps raw dataset columns to canonical names using alias_dict.
    Returns (renamed_df, mapping_dict, unmapped_original_cols).
    """
    rename_map = {}
    used_canonical = set()
    norm_to_orig = {normalize_header(col): col for col in raw_df.columns}

    for canonical, spellings in alias_dict.items():
        for spelling in spellings:
            norm_sp = normalize_header(spelling)
            if norm_sp in norm_to_orig:
                orig_col = norm_to_orig[norm_sp]
                if orig_col not in rename_map and canonical not in used_canonical:
                    rename_map[orig_col] = canonical
                    used_canonical.add(canonical)
                    break

    unmapped = [c for c in raw_df.columns if c not in rename_map]
    return raw_df.rename(columns=rename_map), rename_map, unmapped


def rename_to_known_columns(raw_df):
    """Backward-compatible wrapper mapping student headers to canonical names."""
    renamed, _, _ = rename_with_aliases(raw_df, student_column_aliases)
    return renamed


def clean_skill(text):
    """Normalizes a skill token to its canonical lowercase form via skill_aliases."""
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return skill_aliases.get(text, text)


def split_skills(text):
    """Parses comma/semicolon/pipe-separated skills into a deduplicated, sorted list."""
    if pd.isna(text) or str(text).strip() == "":
        return []
    raw_parts = re.split(r"[,;|/]+", str(text))
    parts = [clean_skill(part) for part in raw_parts]
    return sorted(set(part for part in parts if part and part not in empty_words))


def clean_degree(text):
    """
    Normalizes degree strings across formats:
      'B.Tech', 'BTech', 'Bachelor of Technology' -> 'btech'
      'B.E.', 'BE', 'Bachelor of Engineering' -> 'be'
      'M.Tech', 'MTech' -> 'mtech'
      'MCA' -> 'mca', 'BCA' -> 'bca', 'B.Sc' -> 'bsc', 'B.Com' -> 'bcom'
    """
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    raw = str(text).strip().lower().replace(".", "").replace(" ", "").replace("-", "")
    degree_map = {
        "bacheloroftechnology": "btech",
        "bachelorofengineering": "be",
        "masteroftechnology": "mtech",
        "masterofcomputerapplications": "mca",
        "bachelorofcomputerapplications": "bca",
        "bachelorofscience": "bsc",
        "bachelorofcommerce": "bcom",
        "btech/be": "btech",
    }
    return degree_map.get(raw, raw)


def parse_salary_to_lpa(value):
    """
    Normalizes salary/package representations into Lakhs Per Annum (LPA):
      - Numeric LPA (e.g. 6.5 -> 6.5)
      - Strings with 'lpa', 'lakh', 'lakhs' (e.g. '7.5 LPA' -> 7.5)
      - Annual INR values > 100,000 (e.g. 650000 -> 6.5 LPA)
      - Monthly INR values in [5000, 100000] if marked '/month' or 'pm'
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "").replace("₹", "").replace("inr", "").replace("rs", "").strip()
    if s in empty_words:
        return np.nan
    is_monthly = ("month" in s) or ("/mo" in s) or ("pm" in s)
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return np.nan
    vals = [float(x) for x in nums]
    val = float(np.mean(vals[:2])) if len(vals) >= 2 and "-" in s else vals[0]
    if is_monthly:
        val = (val * 12.0) / 100000.0 if val > 100 else val * 0.12
    elif val >= 100000.0:
        val = val / 100000.0
    elif val > 100.0:
        val = val / 100.0
    return float(val)


def parse_job_experience_months(value, is_binary_flag=False):
    """
    Normalizes job experience requirement into months without fabricating values:
      - 'fresher', '0', 'no', 'none', 'entry level' -> 0.0 months
      - If is_binary_flag (as in Kaggle Dataset 1 where experience_required in {0, 1}):
        0 -> 0.0 months, 1 -> 3.0 months (requires prior internship/project experience;
        flagged as derived from binary indicator).
      - Explicit months (e.g. '6 months') -> 6.0
      - Explicit years (e.g. '0.5', '1 year') -> years * 12.0
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"fresher", "freshers", "entry", "entry level", "none", "no", "false", "0", "0.0"}:
        return 0.0
    if s in {"yes", "true"}:
        return 3.0
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return np.nan
    val = float(nums[0])
    if val < 0:
        return np.nan
    if is_binary_flag and val in (0.0, 1.0):
        return 0.0 if val == 0.0 else 3.0
    if "month" in s or "mo" in s:
        return val
    return val * 12.0


def cgpa_band(cgpa):
    """
    Hierarchical Level-1 CGPA classification (Phase 3A):
      - Low CGPA : below 6.0 (< 6.0)
      - Mid CGPA : 6.0 to below 8.0 (6.0 <= cgpa < 8.0)
      - Top CGPA : 8.0 and above (>= 8.0)
    """
    if cgpa >= 8.0:
        return "top"
    if cgpa >= 6.0:
        return "mid"
    return "low"


cgpa_band_rank = {"low": 0, "mid": 1, "top": 2}
canonical_skill_features = [
    "skill_count", "coding_questions_solved", "projects_count",
    "experience_months", "certifications_count",
]
skill_feature_columns = list(canonical_skill_features)


def parse_placement_label(val):
    """
    Normalizes placement_status labels to binary {0, 1} or np.nan if unavailable:
      'Placed', 'Yes', '1', 'True', 'Offered', 'Selected' -> 1
      'Not Placed', 'Unplaced', 'No', '0', 'False', 'Rejected' -> 0
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"placed", "yes", "1", "1.0", "true", "offered", "selected", "hired"}:
        return 1
    if s in {"not placed", "unplaced", "not_placed", "no", "0", "0.0", "false", "rejected"}:
        return 0
    return np.nan


def make_skill_list(skills_text, language_text, solved, ai_ml_projects=0.0):
    """
    Constructs a student's canonical technical skill list from genuinely available attributes:
      1. Explicitly listed technical_skills (if column is present)
      2. Primary coding_language (if present)
      3. Algorithmic foundations ('data structures', 'algorithms') when
         coding_questions_solved >= coding_skill_threshold (50)
      4. 'machine learning' when ai_ml_projects >= 1 (in Kaggle Dataset 3)
      5. Filters against known_skills demanded by job postings (when known_skills is non-empty)
    """
    skills = split_skills(skills_text)
    if not pd.isna(language_text) and str(language_text).strip().lower() not in empty_words:
        skills = sorted(set(skills + [clean_skill(language_text)]))
    if not pd.isna(solved) and float(solved) >= coding_skill_threshold:
        skills = sorted(set(skills + skills_from_coding))
    if not pd.isna(ai_ml_projects) and float(ai_ml_projects) >= 1.0:
        skills = sorted(set(skills + ["machine learning"]))
    if known_skills:
        return [skill for skill in skills if skill in known_skills]
    return skills


def row_is_valid(row, active_limits=None):
    """Checks numeric parseability and domain bounds for active quantitative student fields."""
    limits_to_check = active_limits if active_limits is not None else field_limits
    for field, (low, high) in limits_to_check.items():
        if field not in row or pd.isna(row.get(field)):
            continue
        value = row.get(field)
        try:
            value = float(value)
        except (ValueError, TypeError):
            return f"{field} is not a number ({row.get(field)!r})"
        if not low <= value <= high:
            return f"{field} out of range ({value}), expected {low} to {high}"
    return None


# ==============================================================================
# SECTION 4: LOAD & PREPROCESS JOB DATASET (PHASE 2A & PHASE 3C)
# ==============================================================================
def load_and_cluster_jobs(file_path):
    """
    Loads brands_data.csv, maps headers, distinguishes actual/missing/optional/derived
    job requirements without fabricating missing cutoffs, and performs salary-weighted
    K-Means (K=3) job-tier clustering.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Job dataset not found: {file_path}")

    raw_jobs = pd.read_csv(file_path)
    orig_cols = list(raw_jobs.columns)
    n_raw = len(raw_jobs)
    dup_count = int(raw_jobs.duplicated().sum())
    if dup_count > 0:
        raw_jobs = raw_jobs.drop_duplicates().reset_index(drop=True)

    df, mapped_cols, unmapped_cols = rename_with_aliases(raw_jobs, job_column_aliases)

    required_base = ["company_name", "job_role", "salary_avg_lpa"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Job dataset '{file_path}' is missing critical columns: {missing_base}. "
            f"Found columns: {orig_cols}"
        )

    req_categories = {
        "actual_source_provided": [],
        "missing_requirements": [],
        "optional_requirements": [],
        "derived_features": [],
    }

    for col in ["company_name", "job_role", "required_skills", "degrees_accepted",
                "salary_avg_lpa", "min_cgpa", "experience_required", "coding_test_rounds",
                "coding_test", "selection_rounds", "interview_count", "company_type", "city", "state"]:
        if col in df.columns and df[col].notna().any():
            req_categories["actual_source_provided"].append(col)

    for opt_col in ["preferred_skills", "primary_skill", "internship_experience", "stipend_if_intern", "remote", "work_type"]:
        if opt_col in df.columns and df[opt_col].notna().any():
            req_categories["optional_requirements"].append(opt_col)

    df["salary_avg_lpa"] = df["salary_avg_lpa"].apply(parse_salary_to_lpa)
    dropped_reasons = []
    valid_mask = pd.Series(True, index=df.index)
    for idx, row in df.iterrows():
        sal = row["salary_avg_lpa"]
        if pd.isna(sal) or sal <= 0 or sal > 250:
            valid_mask.loc[idx] = False
            dropped_reasons.append(f"Row {idx}: invalid salary_avg_lpa ({row.get('salary_avg_lpa')!r})")
        elif pd.isna(row["company_name"]) or str(row["company_name"]).strip() == "":
            valid_mask.loc[idx] = False
            dropped_reasons.append(f"Row {idx}: empty company_name")
        elif pd.isna(row["job_role"]) or str(row["job_role"]).strip() == "":
            valid_mask.loc[idx] = False
            dropped_reasons.append(f"Row {idx}: empty job_role")

    df = df[valid_mask].copy().reset_index(drop=True)

    if "required_skills" in df.columns:
        req_lists = df["required_skills"].apply(split_skills)
    else:
        req_lists = pd.Series([[] for _ in range(len(df))], index=df.index)
        req_categories["missing_requirements"].append("required_skills")

    if "primary_skill" in df.columns:
        prim_lists = df["primary_skill"].apply(split_skills)
        df["required_list"] = [
            sorted(set(r + p)) for r, p in zip(req_lists, prim_lists)
        ]
        req_categories["derived_features"].append("required_list (merged skills + primary_skill)")
    else:
        df["required_list"] = req_lists

    has_preferred_skills = "preferred_skills" in df.columns and df["preferred_skills"].notna().any()
    if has_preferred_skills:
        df["preferred_list"] = df["preferred_skills"].apply(split_skills)
    else:
        df["preferred_list"] = [[] for _ in range(len(df))]
        req_categories["missing_requirements"].append("preferred_skills")

    has_degrees = "degrees_accepted" in df.columns and df["degrees_accepted"].notna().any()
    if has_degrees:
        df["degree_list"] = df["degrees_accepted"].apply(
            lambda text: [clean_degree(part) for part in re.split(r"[,;/|]+", str(text)) if clean_degree(part) != "unknown"]
            if pd.notna(text) and str(text).strip() != "" else []
        )
    else:
        df["degree_list"] = [[] for _ in range(len(df))]
        req_categories["missing_requirements"].append("degrees_accepted")

    has_min_cgpa = "min_cgpa" in df.columns and df["min_cgpa"].notna().any()
    if has_min_cgpa:
        df["min_cgpa"] = pd.to_numeric(df["min_cgpa"], errors="coerce")
        df["min_cgpa"] = np.where(df["min_cgpa"] > 10.0, df["min_cgpa"] / 10.0, df["min_cgpa"])
    else:
        df["min_cgpa"] = np.nan
        req_categories["missing_requirements"].append("min_cgpa")

    has_exp_req = "experience_required" in df.columns and df["experience_required"].notna().any()
    if has_exp_req:
        raw_exp_vals = set(df["experience_required"].dropna().astype(str).str.strip().str.lower().unique())
        is_binary_exp = raw_exp_vals.issubset({"0", "1", "0.0", "1.0", "yes", "no", "true", "false"})
        df["exp_months"] = df["experience_required"].apply(
            lambda v: parse_job_experience_months(v, is_binary_flag=is_binary_exp)
        )
        if is_binary_exp:
            req_categories["derived_features"].append("exp_months (from binary experience_required flag)")
    else:
        df["exp_months"] = np.nan
        req_categories["missing_requirements"].append("experience_required")

    has_coding_rounds = "coding_test_rounds" in df.columns and df["coding_test_rounds"].notna().any()
    if has_coding_rounds:
        df["coding_test_rounds"] = pd.to_numeric(df["coding_test_rounds"], errors="coerce")
    elif "coding_test" in df.columns and df["coding_test"].notna().any():
        coding_bin = pd.to_numeric(df["coding_test"], errors="coerce").fillna(0.0)
        if "selection_rounds" in df.columns and df["selection_rounds"].notna().any():
            sel_rounds = pd.to_numeric(df["selection_rounds"], errors="coerce").fillna(1.0)
            df["coding_test_rounds"] = np.where(coding_bin > 0, np.maximum(1.0, np.round(sel_rounds * 0.5)), 0.0)
            req_categories["derived_features"].append("coding_test_rounds (derived from coding_test & selection_rounds)")
        else:
            df["coding_test_rounds"] = coding_bin
            req_categories["derived_features"].append("coding_test_rounds (from binary coding_test indicator)")
        req_categories["missing_requirements"].append("coding_test_rounds (exact count not in source; derived indicator used)")
        has_coding_rounds = True
    else:
        df["coding_test_rounds"] = np.nan
        req_categories["missing_requirements"].append("coding_test_rounds")

    df["req_skill_count"] = df["required_list"].apply(lambda lst: len([s for s in lst if s not in soft_skills]))
    req_categories["derived_features"].append("req_skill_count")
    df["needs_comm"] = df["required_list"].apply(lambda skills: "communication" in skills)

    active_job_features = ["salary_avg_lpa"]
    active_job_weights = [3.0]
    for cand_col in ["min_cgpa", "exp_months", "coding_test_rounds"]:
        if cand_col in df.columns and df[cand_col].notna().sum() >= max(3, int(0.5 * len(df))):
            active_job_features.append(cand_col)
            active_job_weights.append(1.0)
    if len(active_job_features) == 1 and df["req_skill_count"].nunique() > 1:
        active_job_features.append("req_skill_count")
        active_job_weights.append(1.0)

    job_weights_arr = np.array(active_job_weights, dtype=float)
    job_feat_matrix = df[active_job_features].copy()
    for col in active_job_features:
        med = float(job_feat_matrix[col].median()) if job_feat_matrix[col].notna().any() else 0.0
        job_feat_matrix[col] = job_feat_matrix[col].fillna(med)

    j_scaler = StandardScaler()
    j_scaled = j_scaler.fit_transform(job_feat_matrix.values) * job_weights_arr

    k_eff = min(job_tiers, len(df))
    j_model = KMeans(n_clusters=k_eff, n_init=10, random_state=random_seed)
    df["job_cluster"] = j_model.fit_predict(j_scaled)

    j_centers = j_scaler.inverse_transform(j_model.cluster_centers_ / job_weights_arr)
    j_order = np.argsort(j_centers[:, 0])
    j_cluster_to_rank = {int(cid): rank for rank, cid in enumerate(j_order)}
    df["tier_rank"] = df["job_cluster"].map(j_cluster_to_rank)

    is_kaggle_ds1 = ("job_id" in mapped_cols.values() and "company_type" in mapped_cols.values())
    provenance_note = (
        "Kaggle Dataset 1 (ankushnarwade/indian-fresher-salary-and-skill-trends-2025) detected. "
        "Provenance verification: Although titled 'Indian Job Market Dataset 2024-25', its Kaggle "
        "documentation describes '500 realistic job records' and inspection reveals synthetic "
        "combinations (e.g., mismatched city-state pairs and uniform random skill triplets) without "
        "verified employer source disclosures. It is treated as a synthetic/simulated job-market "
        "benchmark rather than verified real-world employer records."
        if is_kaggle_ds1 else
        "Structured benchmark company/job dataset (brands_data.csv). Treated as a research benchmark "
        "dataset; verify employer-level provenance before representing as live recruiting records."
    )

    job_prep_meta = {
        "file_path": file_path,
        "raw_rows": n_raw,
        "duplicates_removed": dup_count,
        "valid_rows": len(df),
        "dropped_rows": n_raw - dup_count - len(df),
        "dropped_reasons": dropped_reasons,
        "original_columns": orig_cols,
        "mapped_columns": mapped_cols,
        "unmapped_columns": unmapped_cols,
        "missing_values_per_col": raw_jobs.isna().sum().to_dict(),
        "requirement_categories": req_categories,
        "active_job_cluster_features": active_job_features,
        "active_job_cluster_weights": active_job_weights,
        "has_min_cgpa": bool(has_min_cgpa),
        "has_preferred_skills": bool(has_preferred_skills),
        "has_degrees": bool(has_degrees),
        "has_exp_req": bool(has_exp_req),
        "has_coding_rounds": bool(has_coding_rounds),
        "provenance_note": provenance_note,
    }
    return df, j_scaler, j_scaled, j_model, active_job_features, job_weights_arr, job_prep_meta


# ==============================================================================
# SECTION 5: LOAD & PREPROCESS STUDENT TRAINING AND TESTING DATASETS (PHASE 2B-2D)
# ==============================================================================
def preprocess_student_dataframe(
    raw_df,
    dataset_label,
    known_job_skills,
    fitted_stats=None,
    is_training=True,
    train_student_ids=None,
):
    """
    Normalizes, validates, and preprocesses a student dataset (training or testing).
    When is_training=True, fits data-dependent statistics (imputation medians,
    communication quantiles, experience unit scaling) strictly on the training set
    and returns them in fitted_stats.
    When is_training=False, applies the frozen fitted_stats from training data without
    peeking at test distributions or test labels, and verifies zero student_id overlap.
    """
    orig_cols = list(raw_df.columns)
    n_raw = len(raw_df)
    missing_counts = raw_df.replace(list(empty_words), np.nan).isna().sum().to_dict()

    df, mapped_cols, unmapped_cols = rename_with_aliases(raw_df, student_column_aliases)

    if fitted_stats is None:
        fitted_stats = {}

    if "student_id" not in df.columns:
        prefix = "TR" if is_training else "TS"
        df["student_id"] = [f"{prefix}-{i+1:05d}" for i in range(len(df))]

    df["student_id"] = df["student_id"].astype(str).str.strip()

    dropped_reasons = []
    keep_mask = pd.Series(True, index=df.index)
    seen_ids = set()
    dup_id_count = 0
    overlap_id_count = 0

    for idx, row in df.iterrows():
        sid = str(row["student_id"]).strip()
        if not sid or sid.lower() in empty_words:
            keep_mask.loc[idx] = False
            dropped_reasons.append(f"Row {idx}: empty student_id")
            continue
        if sid in seen_ids:
            keep_mask.loc[idx] = False
            dup_id_count += 1
            dropped_reasons.append(f"Row {idx} ({sid}): duplicate student_id within {dataset_label}")
            continue
        if not is_training and train_student_ids is not None and sid in train_student_ids:
            keep_mask.loc[idx] = False
            overlap_id_count += 1
            dropped_reasons.append(f"Row {idx} ({sid}): student_id overlaps with training set (removed to prevent leakage)")
            continue
        seen_ids.add(sid)

    df = df[keep_mask].copy().reset_index(drop=True)

    if "cgpa" not in df.columns:
        raise ValueError(f"{dataset_label} is missing required 'cgpa' column. Columns found: {orig_cols}")

    cgpa_vals = pd.to_numeric(df["cgpa"], errors="coerce")
    pct_mask = (cgpa_vals > 10.0) & (cgpa_vals <= 100.0)
    if pct_mask.any():
        cgpa_vals = np.where(pct_mask, cgpa_vals / 10.0, cgpa_vals)
    df["cgpa"] = cgpa_vals

    feature_availability = {}
    derived_notes = []

    if "projects_count" in df.columns and df["projects_count"].notna().any():
        df["projects_count"] = pd.to_numeric(df["projects_count"], errors="coerce")
        feature_availability["projects_count"] = "actual"
    elif "development_projects_count" in df.columns or "ai_ml_projects" in df.columns:
        dev_p = pd.to_numeric(df.get("development_projects_count", 0), errors="coerce").fillna(0.0)
        aiml_p = pd.to_numeric(df.get("ai_ml_projects", 0), errors="coerce").fillna(0.0)
        df["projects_count"] = dev_p + aiml_p
        feature_availability["projects_count"] = "derived (development_projects_count + ai_ml_projects)"
        derived_notes.append("projects_count derived as sum of development_projects_count and ai_ml_projects")
    else:
        df["projects_count"] = np.nan
        feature_availability["projects_count"] = "unavailable"

    if "experience_months" in df.columns and df["experience_months"].notna().any():
        df["experience_months"] = pd.to_numeric(df["experience_months"], errors="coerce")
        feature_availability["experience_months"] = "actual (months)"
    elif "internships_completed" in df.columns and df["internships_completed"].notna().any():
        intern_cnt = pd.to_numeric(df["internships_completed"], errors="coerce")
        df["internships_completed"] = intern_cnt
        df["experience_months"] = intern_cnt * 2.0
        feature_availability["experience_months"] = "derived_unit_normalized (internships_completed * 2.0 months)"
        derived_notes.append("experience_months normalized from internships_completed count (1 internship = 2 months)")
    else:
        df["experience_months"] = np.nan
        feature_availability["experience_months"] = "unavailable"

    if "coding_questions_solved" in df.columns and df["coding_questions_solved"].notna().any():
        df["coding_questions_solved"] = pd.to_numeric(df["coding_questions_solved"], errors="coerce")
        feature_availability["coding_questions_solved"] = "actual"
    else:
        df["coding_questions_solved"] = np.nan
        feature_availability["coding_questions_solved"] = "unavailable"

    if "certifications_count" in df.columns and df["certifications_count"].notna().any():
        df["certifications_count"] = pd.to_numeric(df["certifications_count"], errors="coerce")
        feature_availability["certifications_count"] = "actual"
    else:
        df["certifications_count"] = np.nan
        feature_availability["certifications_count"] = "unavailable"

    has_free_text_skills = "technical_skills" in df.columns and df["technical_skills"].replace(list(empty_words), np.nan).notna().any()
    has_coding_lang = "coding_language" in df.columns and df["coding_language"].replace(list(empty_words), np.nan).notna().any()
    if not has_coding_lang:
        df["coding_language"] = ""
    if not has_free_text_skills:
        df["technical_skills"] = ""
        feature_availability["technical_skills"] = (
            "derived_from_structured_fields (primary_language, DSA volume, AI/ML projects)"
            if has_coding_lang else "unavailable"
        )
    else:
        feature_availability["technical_skills"] = "actual"

    valid_rows_mask = pd.Series(True, index=df.index)
    for idx, row in df.iterrows():
        sid = row["student_id"]
        cgpa_v = row["cgpa"]
        if pd.isna(cgpa_v) or not (0.0 <= float(cgpa_v) <= 10.0):
            valid_rows_mask.loc[idx] = False
            dropped_reasons.append(f"Student {sid}: cgpa invalid or out of [0, 10] ({row.get('cgpa')!r})")
            continue
        for fld, (lo, hi) in field_limits.items():
            if fld == "cgpa":
                continue
            if feature_availability.get(fld, "unavailable") != "unavailable":
                val = row.get(fld)
                if pd.isna(val):
                    raw_val = row.get(fld)
                    if raw_val is not None and str(raw_val).strip().lower() not in empty_words and not pd.isna(raw_val):
                        valid_rows_mask.loc[idx] = False
                        dropped_reasons.append(f"Student {sid}: {fld} is non-numeric ({raw_val!r})")
                        break
                elif not (lo <= float(val) <= hi):
                    valid_rows_mask.loc[idx] = False
                    dropped_reasons.append(f"Student {sid}: {fld} out of range ({val}), expected {lo} to {hi}")
                    break

    df = df[valid_rows_mask].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid student records remained in {dataset_label} after validation.")

    numeric_cols_to_impute = [
        "coding_questions_solved", "projects_count", "experience_months", "certifications_count",
        "github_repos", "hackathons_participated", "aptitude_score", "resume_score",
        "mock_interview_score", "backlog_history", "attendance_percentage", "college_tier",
        "ai_tool_usage_frequency", "prompt_engineering_skill", "adaptability_score", "leadership_score"
    ]
    if is_training:
        fitted_stats["imputation_medians"] = {}
        for col in numeric_cols_to_impute:
            if col in df.columns:
                s_num = pd.to_numeric(df[col], errors="coerce")
                med = float(s_num.median()) if s_num.notna().any() else np.nan
                fitted_stats["imputation_medians"][col] = med
                if not np.isnan(med):
                    df[col] = s_num.fillna(med)
    else:
        medians = fitted_stats.get("imputation_medians", {})
        for col in numeric_cols_to_impute:
            if col in df.columns:
                s_num = pd.to_numeric(df[col], errors="coerce")
                if s_num.isna().any() and not s_num.isna().all():
                    fill_val = medians.get(col, np.nan)
                    if np.isnan(fill_val):
                        fill_val = 0.0
                    df[col] = s_num.fillna(fill_val)
                else:
                    df[col] = s_num

    aiml_col = pd.to_numeric(df["ai_ml_projects"], errors="coerce").fillna(0.0) if "ai_ml_projects" in df.columns else pd.Series(0.0, index=df.index)
    df["skill_list"] = [
        make_skill_list(sk_txt, lang_txt, solved_val, aiml_val)
        for sk_txt, lang_txt, solved_val, aiml_val in zip(
            df["technical_skills"],
            df["coding_language"],
            df["coding_questions_solved"].fillna(0.0),
            aiml_col,
        )
    ]
    df["skill_count"] = df["skill_list"].apply(len).astype(float)
    feature_availability["skill_count"] = (
        "actual" if has_free_text_skills else
        ("derived_from_structured_fields" if has_coding_lang else "unavailable")
    )

    df["degree"] = df["degree"].apply(clean_degree) if "degree" in df.columns else "btech"
    df["branch"] = df["branch"].astype(str).str.strip().str.lower() if "branch" in df.columns else "cse"

    if "communication" in df.columns and df["communication"].notna().any():
        raw_comm = df["communication"].astype(str).str.strip().str.lower()
        mapped_cat = raw_comm.map(communication_scores)
        num_comm = pd.to_numeric(df["communication"], errors="coerce")

        if mapped_cat.notna().sum() >= 0.5 * len(df):
            df["comm_score"] = mapped_cat.fillna(2).astype(int)
            df["comm_continuous"] = df["comm_score"].astype(float)
            if is_training:
                fitted_stats["comm_mode"] = "categorical"
        elif num_comm.notna().any():
            if is_training:
                q33, q66 = np.nanquantile(num_comm, [0.33, 0.67])
                fitted_stats["comm_mode"] = "numeric"
                fitted_stats["comm_quantiles"] = (float(q33), float(q66))
                fitted_stats["comm_median"] = float(np.nanmedian(num_comm))
            q33, q66 = fitted_stats.get("comm_quantiles", (4.0, 7.0))
            c_med = fitted_stats.get("comm_median", 5.0)
            filled_num = num_comm.fillna(c_med)
            df["comm_continuous"] = filled_num.astype(float)
            df["comm_score"] = np.where(filled_num < q33, 1, np.where(filled_num < q66, 2, 3)).astype(int)
        else:
            df["comm_score"] = 2
            df["comm_continuous"] = 2.0
        feature_availability["communication"] = "actual"
    else:
        df["comm_score"] = np.nan
        df["comm_continuous"] = np.nan
        feature_availability["communication"] = "unavailable"

    if "placement_status" in df.columns and df["placement_status"].notna().any():
        df["placement_label"] = df["placement_status"].apply(parse_placement_label)
        has_placement_labels = bool(df["placement_label"].notna().any())
    else:
        df["placement_label"] = np.nan
        has_placement_labels = False

    if "salary_lpa" in df.columns and df["salary_lpa"].notna().any():
        df["observed_salary_lpa"] = df["salary_lpa"].apply(parse_salary_to_lpa)
    else:
        df["observed_salary_lpa"] = np.nan

    df["cgpa_band"] = df["cgpa"].apply(cgpa_band)

    prep_meta = {
        "dataset_label": dataset_label,
        "raw_rows": n_raw,
        "valid_rows": len(df),
        "dropped_rows": n_raw - len(df),
        "duplicate_ids_removed": dup_id_count,
        "train_overlap_ids_removed": overlap_id_count,
        "dropped_reasons": dropped_reasons,
        "original_columns": orig_cols,
        "mapped_columns": mapped_cols,
        "unmapped_columns": unmapped_cols,
        "missing_values_per_col": missing_counts,
        "feature_availability": feature_availability,
        "derived_notes": derived_notes,
        "has_placement_labels": has_placement_labels,
        "has_free_text_skills": bool(has_free_text_skills),
        "provenance_note": (
            f"{dataset_label} is documented as a SYNTHETIC student dataset (Kaggle 2026 placement "
            f"benchmark). Records represent simulated engineering profiles and must NOT be "
            f"represented as real individual students."
        ),
    }
    return df, fitted_stats, prep_meta


def load_students_to_check(file_path, fitted_stats=None, train_ids=None):
    """Loads and validates unseen student testing dataset (check_student.csv)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Testing student dataset not found: {file_path}")
    raw = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    df, _, _ = preprocess_student_dataframe(
        raw,
        dataset_label=f"Student Testing Dataset ({file_path})",
        known_job_skills=known_skills,
        fitted_stats=fitted_stats,
        is_training=False,
        train_student_ids=train_ids,
    )
    return df


# ==============================================================================
# SECTION 6: HIERARCHICAL K-MEANS STUDENT CLUSTERING & MARGIN SCORE (PHASE 3B & 4C)
# ==============================================================================
def determine_common_clustering_features(train_df, check_df, train_meta, check_meta):
    """
    Determines the scientifically justified common feature set available in BOTH
    student_data.csv and check_student.csv for Level-2 K-Means skill clustering.
    Never fabricates missing testing columns.
    """
    common_features = []
    unavailable_in_train = []
    unavailable_in_check = []

    for feat in canonical_skill_features:
        tr_avail = train_meta["feature_availability"].get(feat, "unavailable") != "unavailable"
        ch_avail = check_meta["feature_availability"].get(feat, "unavailable") != "unavailable"
        if feat == "skill_count" and (train_meta["has_free_text_skills"] != check_meta["has_free_text_skills"]):
            ch_avail = False
        if tr_avail and ch_avail and train_df[feat].notna().all() and check_df[feat].notna().all():
            common_features.append(feat)
        else:
            if not tr_avail:
                unavailable_in_train.append(feat)
            if not ch_avail:
                unavailable_in_check.append(feat)

    if len(common_features) < 3:
        for extra_feat in ["github_repos", "hackathons_participated", "aptitude_score", "resume_score"]:
            if (extra_feat in train_df.columns and extra_feat in check_df.columns
                    and train_df[extra_feat].notna().all() and check_df[extra_feat].notna().all()):
                common_features.append(extra_feat)

    if not common_features:
        raise ValueError(
            "No common numerical skill-readiness features found between training and testing datasets."
        )
    return common_features, unavailable_in_train, unavailable_in_check


def fit_hierarchical_student_models(train_df, active_features):
    """
    Fits Level-2 K-Means (K=3) inside each Level-1 CGPA band ('low', 'mid', 'top')
    strictly on training data, ordering cluster IDs by ascending centroid strength.
    Handles insufficient records safely via standardized composite-score tertiles.
    """
    models = {}
    for band in ["low", "mid", "top"]:
        part = train_df[train_df["cgpa_band"] == band]
        if len(part) == 0:
            models[band] = None
            continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(part[active_features].values)
        if len(part) < min_students_to_cluster or len(part) < skill_subclusters:
            comp_scores = scaled.mean(axis=1)
            q33, q66 = np.quantile(comp_scores, [1 / 3, 2 / 3]) if len(part) >= 3 else (-0.5, 0.5)
            models[band] = {
                "mode": "small_sample_fallback",
                "scaler": scaler,
                "model": None,
                "quantiles": (float(q33), float(q66)),
                "features": list(active_features),
                "n_train": len(part),
            }
            continue

        model = KMeans(n_clusters=skill_subclusters, n_init=10, random_state=random_seed)
        model.fit(scaled)
        strength = model.cluster_centers_.mean(axis=1)
        order = np.argsort(strength)
        cluster_to_rank = {int(cid): rank for rank, cid in enumerate(order)}
        models[band] = {
            "mode": "kmeans",
            "scaler": scaler,
            "model": model,
            "cluster_to_rank": cluster_to_rank,
            "features": list(active_features),
            "n_train": len(part),
        }
    return models


def compute_relative_cluster_margin(distances):
    """
    Computes the Relative Cluster-Margin Score (Phase 4C):
        M(x) = 1 - d_1(x) / d_2(x)
    where d_1(x) <= d_2(x) are the Euclidean distances from standardized point x
    to the nearest and second-nearest K-Means centroids.
    """
    dists = np.sort(np.asarray(distances, dtype=float))
    if len(dists) < 2:
        return 0.0
    d1, d2 = float(dists[0]), float(dists[1])
    if d2 <= 1e-12:
        return 0.0
    if abs(d1 - d2) <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - (d1 / d2), 0.0, 1.0))


def predict_skill_band(row, band):
    """
    Projects a student record into its trained Level-2 skill cluster ('low', 'mid', 'good')
    and returns (skill_band_name, relative_cluster_margin_score).
    """
    trained = band_models.get(band)
    if trained is None:
        return "mid", np.nan
    feats = trained.get("features", skill_feature_columns)
    x_vec = [[float(row[f]) for f in feats]]
    scaled = trained["scaler"].transform(x_vec)

    if trained.get("mode") == "small_sample_fallback" or trained.get("model") is None:
        score = float(scaled.mean())
        q33, q66 = trained["quantiles"]
        rank = 0 if score < q33 else (1 if score < q66 else 2)
        return skill_names[rank], 0.50

    distances = trained["model"].transform(scaled)[0]
    cluster_id = int(np.argmin(distances))
    margin_score = compute_relative_cluster_margin(distances)
    rank = trained["cluster_to_rank"][cluster_id]
    return skill_names[rank], margin_score


# ==============================================================================
# SECTION 7: INITIALIZE PIPELINE DATA & SAVE PREPROCESSING REPORT
# ==============================================================================
def write_preprocessing_report(datasets_status, job_meta, train_meta, check_meta, common_feats, unavail_tr, unavail_ch):
    """Writes output/reports/dataset_preprocessing_report.txt."""
    lines = [
        "=" * 88,
        "DATASET SCHEMA MAPPING, PROVENANCE & PREPROCESSING REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 88,
        "",
        "1. DATASET SOURCES & PROVENANCE VERIFICATION",
        "-" * 88,
        f"  - Job Dataset (brands_data.csv)     : {datasets_status.get('jobs_source')}",
        f"    Provenance Note: {job_meta['provenance_note']}",
        f"  - Training Dataset (student_data.csv): {datasets_status.get('train_source')}",
        f"    Provenance Note: {train_meta['provenance_note']}",
        f"  - Testing Dataset (check_student.csv): {datasets_status.get('check_source')}",
        f"    Provenance Note: {check_meta['provenance_note']}",
        "",
        "2. DATASET 1 — COMPANY / JOB DATASET (brands_data.csv)",
        "-" * 88,
        f"  - Raw Rows Loaded         : {job_meta['raw_rows']}",
        f"  - Duplicate Rows Removed  : {job_meta['duplicates_removed']}",
        f"  - Valid Rows Retained     : {job_meta['valid_rows']}",
        f"  - Dropped Invalid Rows    : {job_meta['dropped_rows']}",
        f"  - Original Columns ({len(job_meta['original_columns'])}): {', '.join(job_meta['original_columns'])}",
        "  - Mapped Canonical Columns:",
    ]
    for orig_c, can_c in job_meta["mapped_columns"].items():
        lines.append(f"      * {orig_c:<28} -> {can_c}")
    if job_meta["unmapped_columns"]:
        lines.append(f"  - Unmapped/Preserved Extra Columns: {', '.join(job_meta['unmapped_columns'])}")
    lines.extend([
        "  - Missing Values per Original Column:",
        "      " + ", ".join(f"{k}: {v}" for k, v in job_meta["missing_values_per_col"].items()),
        "  - Requirement Categorization (No Fabricated Cutoffs):",
        f"      * Actual Source-Provided : {', '.join(job_meta['requirement_categories']['actual_source_provided']) or 'None'}",
        f"      * Missing Requirements   : {', '.join(job_meta['requirement_categories']['missing_requirements']) or 'None'}",
        f"      * Optional Requirements  : {', '.join(job_meta['requirement_categories']['optional_requirements']) or 'None'}",
        f"      * Derived Features       : {', '.join(job_meta['requirement_categories']['derived_features']) or 'None'}",
        f"  - Active Job-Tier Clustering Features: {job_meta['active_job_cluster_features']} (weights: {job_meta['active_job_cluster_weights']})",
    ])
    if job_meta["dropped_reasons"]:
        lines.append("  - Dropped Row Reasons:")
        for r in job_meta["dropped_reasons"][:15]:
            lines.append(f"      * {r}")

    for sec_num, meta in [("3. DATASET 2 — HISTORICAL STUDENT TRAINING DATASET (student_data.csv)", train_meta),
                          ("4. DATASET 3 — INDEPENDENT STUDENT TESTING DATASET (check_student.csv)", check_meta)]:
        lines.extend([
            "",
            sec_num,
            "-" * 88,
            f"  - Raw Rows Loaded                : {meta['raw_rows']}",
            f"  - Duplicate IDs Removed          : {meta['duplicate_ids_removed']}",
            f"  - Train-Test Overlap IDs Removed : {meta['train_overlap_ids_removed']}",
            f"  - Valid Rows Retained            : {meta['valid_rows']}",
            f"  - Total Dropped Rows             : {meta['dropped_rows']}",
            f"  - Original Columns ({len(meta['original_columns'])}): {', '.join(meta['original_columns'])}",
            "  - Mapped Canonical Columns:",
        ])
        for orig_c, can_c in meta["mapped_columns"].items():
            lines.append(f"      * {orig_c:<30} -> {can_c}")
        if meta["unmapped_columns"]:
            lines.append(f"  - Unmapped/Preserved Columns     : {', '.join(meta['unmapped_columns'])}")
        lines.append("  - Missing Values per Original Column:")
        lines.append("      " + ", ".join(f"{k}: {v}" for k, v in meta["missing_values_per_col"].items()))
        lines.append("  - Feature Availability Status:")
        for fk, fv in meta["feature_availability"].items():
            lines.append(f"      * {fk:<26} : {fv}")
        if meta["derived_notes"]:
            lines.append("  - Derived Feature Notes:")
            for dn in meta["derived_notes"]:
                lines.append(f"      * {dn}")
        if meta["dropped_reasons"]:
            lines.append(f"  - Dropped Row Reasons (showing up to 15 of {len(meta['dropped_reasons'])}):")
            for r in meta["dropped_reasons"][:15]:
                lines.append(f"      * {r}")

    lines.extend([
        "",
        "5. COMMON-FEATURE MODEL & LEAKAGE PREVENTION SUMMARY",
        "-" * 88,
        f"  - Canonical Candidate Skill Features : {canonical_skill_features}",
        f"  - Active Common Clustering Features  : {common_feats}",
        f"  - Unavailable in Training Dataset    : {unavail_tr or 'None'}",
        f"  - Unavailable in Testing Dataset     : {unavail_ch or 'None'}",
        "  - Leakage Prevention Verification    :",
        "      * StandardScaler, K-Means centroids, communication quantiles, and imputation medians",
        "        were fitted strictly on student_data.csv (training data) and applied to check_student.csv.",
        "      * Post-placement outcome variables (placement_status, salary_lpa) are strictly excluded",
        "        from unsupervised clustering and placement prediction inputs.",
        "      * Training and testing student_id sets verified to have 0 overlap.",
        "=" * 88,
    ])
    report_text = "\n".join(lines) + "\n"
    with open(output_preprocessing_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    log_info(f"Saved preprocessing report to {output_preprocessing_report}")
    return report_text


def initialize_pipeline():
    """Loads or reloads all three datasets, fits training models, and updates module globals."""
    global datasets_status_info, jobs, job_scaler, job_scaled, job_model
    global job_feature_columns, job_feature_weights, job_prep_metadata
    global known_skills, job_min_cgpa, job_exp_months, job_needs_comm, all_degrees, degree_accepts_job
    global train_students, fitted_preprocessing_stats, train_prep_metadata
    global checked_students, check_prep_metadata
    global skill_feature_columns, unavailable_train_features, unavailable_check_features
    global band_models, group_records, student_to_group, preprocessing_report_text

    ensure_output_directories()
    datasets_status_info = ensure_default_datasets()
    (
        jobs,
        job_scaler,
        job_scaled,
        job_model,
        job_feature_columns,
        job_feature_weights,
        job_prep_metadata,
    ) = load_and_cluster_jobs(job_file_name)

    known_skills = set()
    for skill_list in jobs["required_list"] + jobs["preferred_list"]:
        known_skills.update(s for s in skill_list if s not in soft_skills)

    job_min_cgpa = jobs["min_cgpa"].to_numpy()
    job_exp_months = jobs["exp_months"].to_numpy()
    job_needs_comm = jobs["needs_comm"].to_numpy()

    all_degrees = {clean_degree(d) for row in jobs["degree_list"] for d in row if d}
    degree_accepts_job = {
        degree: jobs["degree_list"].apply(lambda row: (degree in row) if len(row) > 0 else True).to_numpy()
        for degree in all_degrees
    }

    raw_train_df = pd.read_csv(train_student_file_name, dtype=str, keep_default_na=False)
    train_students, fitted_preprocessing_stats, train_prep_metadata = preprocess_student_dataframe(
        raw_train_df,
        dataset_label=f"Student Training Dataset ({train_student_file_name})",
        known_job_skills=known_skills,
        fitted_stats=None,
        is_training=True,
    )

    raw_check_df = pd.read_csv(check_student_file_name, dtype=str, keep_default_na=False)
    checked_students, _, check_prep_metadata = preprocess_student_dataframe(
        raw_check_df,
        dataset_label=f"Student Testing Dataset ({check_student_file_name})",
        known_job_skills=known_skills,
        fitted_stats=fitted_preprocessing_stats,
        is_training=False,
        train_student_ids=set(train_students["student_id"]),
    )

    (
        skill_feature_columns,
        unavailable_train_features,
        unavailable_check_features,
    ) = determine_common_clustering_features(
        train_students, checked_students, train_prep_metadata, check_prep_metadata
    )

    band_models = fit_hierarchical_student_models(train_students, skill_feature_columns)

    train_preds = [predict_skill_band(row, row["cgpa_band"]) for _, row in train_students.iterrows()]
    train_students["skill_band"] = [b_name for b_name, _ in train_preds]
    train_students["margin_score"] = [m_val for _, m_val in train_preds]
    train_students["confidence"] = train_students["margin_score"]

    check_preds = [predict_skill_band(row, row["cgpa_band"]) for _, row in checked_students.iterrows()]
    checked_students["skill_band"] = [b_name for b_name, _ in check_preds]
    checked_students["margin_score"] = [m_val for _, m_val in check_preds]
    checked_students["confidence"] = checked_students["margin_score"]

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

    train_group_numbers = []
    for _, row in train_students.iterrows():
        b_idx = cgpa_band_rank[row["cgpa_band"]]
        s_idx = skill_names.index(row["skill_band"])
        train_group_numbers.append(b_idx * 3 + s_idx + 1)
    train_students["group_number"] = train_group_numbers

    preprocessing_report_text = write_preprocessing_report(
        datasets_status_info,
        job_prep_metadata,
        train_prep_metadata,
        check_prep_metadata,
        skill_feature_columns,
        unavailable_train_features,
        unavailable_check_features,
    )


initialize_pipeline()


# ==============================================================================
# SECTION 8: STUDENT-JOB MATCHING, SKILL-GAP & UPGRADE RECOMMENDATIONS (PHASE 3D)
# ==============================================================================
def group_skill_frequency(member_students):
    """Computes the fraction of students in a group possessing each canonical technical skill."""
    counts = Counter(skill for skills in member_students["skill_list"] for skill in skills)
    total = max(len(member_students), 1)
    return {skill: count / total for skill, count in counts.items()}


def score_jobs_for_group(member_students, required_only=False):
    """
    Evaluates all job postings for a student cohort (or a 1-row individual DataFrame):
      - Checks only genuinely available eligibility requirements (CGPA, Degree, Experience,
        Communication).
      - Computes skill_match:
          * 0.8 * req_cov + 0.2 * pref_cov when preferred_skills is available for that job
          * 1.0 * req_cov when preferred_skills is empty/unavailable in the job dataset.
    """
    skill_freq = group_skill_frequency(member_students)
    active_gates = []

    if "cgpa" in member_students.columns and np.any(~np.isnan(job_min_cgpa)):
        cgpa_arr = member_students["cgpa"].to_numpy()[:, None]
        raw_cgpa_ok = (cgpa_arr >= job_min_cgpa[None, :]).mean(axis=0)
        cgpa_ok = np.where(np.isnan(job_min_cgpa), 1.0, raw_cgpa_ok)
        active_gates.append(cgpa_ok)

    if ("experience_months" in member_students.columns
            and member_students["experience_months"].notna().any()
            and np.any(~np.isnan(job_exp_months))):
        exp_arr = member_students["experience_months"].to_numpy()[:, None]
        raw_exp_ok = (exp_arr >= job_exp_months[None, :]).mean(axis=0)
        exp_ok = np.where(np.isnan(job_exp_months), 1.0, raw_exp_ok)
        active_gates.append(exp_ok)

    if "comm_score" in member_students.columns and member_students["comm_score"].notna().any():
        comm_arr = member_students["comm_score"].to_numpy()[:, None]
        comm_ok_if_needed = (comm_arr >= 2).mean(axis=0)
        comm_ok = np.where(job_needs_comm, comm_ok_if_needed, 1.0)
        active_gates.append(comm_ok)

    if "degree" in member_students.columns and job_prep_metadata.get("has_degrees", True):
        degree_ok_rows = np.vstack([
            degree_accepts_job.get(
                d,
                jobs["degree_list"].apply(lambda r: len(r) == 0).to_numpy()
            )
            for d in member_students["degree"]
        ])
        degree_ok = degree_ok_rows.mean(axis=0)
        active_gates.append(degree_ok)

    if active_gates:
        eligible_fraction = np.mean(np.vstack(active_gates), axis=0)
    else:
        eligible_fraction = np.ones(len(jobs), dtype=float)

    skill_match, matched_list, missing_list, required_out = [], [], [], []
    for required, preferred in zip(jobs["required_list"], jobs["preferred_list"]):
        required = [s for s in required if s not in soft_skills]
        preferred = [s for s in preferred if s not in soft_skills]
        required_out.append(required)
        req_cov = float(np.mean([skill_freq.get(s, 0.0) for s in required])) if required else 1.0
        if len(preferred) > 0 and not required_only:
            pref_cov = float(np.mean([skill_freq.get(s, 0.0) for s in preferred]))
            match_val = 0.8 * req_cov + 0.2 * pref_cov
        else:
            match_val = req_cov
        skill_match.append(match_val)
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


def roles_table(group_number, chosen, recommendation_type="apply_now"):
    columns = [
        "group", "recommendation_type", "rank", "role", "tier", "companies",
        "salary_lpa", "skill_match_pct", "required_skills", "missing_skills",
    ]
    if len(chosen) == 0:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "group": group_number,
        "recommendation_type": recommendation_type,
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
    Produces the cohort outputs for a (cgpa_band, skill_band) group:
      1. Group summary row (with relative cluster-margin score)
      2. Apply Now roles at or below the group's tier passing skill & eligibility gates
      3. Upgrade Path target roles in the next achievable tier
      4. Prioritized missing technical skills to bridge the tier gap
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
        "avg_margin_score_pct": (
            f"{member_students['margin_score'].mean() * 100:.1f}"
            if member_students["margin_score"].notna().any() else "-"
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

    if len(ready) > 0:
        achieved_tier = int(ready["tier_rank"].max())
        next_rank = min(achieved_tier + 1, job_tiers - 1)
    else:
        next_rank = 0

    targets = collapse_roles(scored[scored["tier_rank"] == next_rank]).sort_values(
        ["skill_match", "salary_avg_lpa"], ascending=False
    ).head(roles_to_show)

    apply_table = roles_table(group_number, ready, recommendation_type="apply_now")
    upgrade_table = roles_table(group_number, targets, recommendation_type="upgrade_path")

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
# SECTION 9: SPRINGER MULTI-CRITERIA CAPABILITY MATRIX & MATHEMATICALLY VALID
#            CAPACITY-CONSTRAINED ALLOCATION (PHASE 3D)
# ==============================================================================
def compute_springer_tier_mapping(entity_df, entity_labels, is_group=False):
    """
    Computes the Springer Table 1 & Table 2 multi-criteria evaluation across the
    3 clustered Job Tiers (J1: Low Tier, J2: Mid Tier, J3: Top Tier).
    """
    tier_benchmarks = {}
    for t_rank in range(job_tiers):
        t_jobs = jobs[jobs["tier_rank"] == t_rank].sort_values("salary_avg_lpa", ascending=False)
        if len(t_jobs) == 0:
            t_jobs = jobs.sort_values("salary_avg_lpa", ascending=False)
        rep_job = t_jobs.iloc[0]
        med_cgpa = float(t_jobs["min_cgpa"].median()) if t_jobs["min_cgpa"].notna().any() else np.nan
        tier_benchmarks[t_rank] = {
            "job_code": f"Job{t_rank + 1}(J{t_rank + 1})",
            "role": rep_job["job_role"],
            "company": rep_job["company_name"],
            "cutoff": tier_cutoffs[t_rank],
            "salary_lpa": float(t_jobs["salary_avg_lpa"].median()),
            "max_salary_lpa": float(t_jobs["salary_avg_lpa"].max()),
            "min_cgpa": med_cgpa,
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

        comm_mean = float(member_df["comm_score"].mean()) if member_df["comm_score"].notna().any() else 2.0
        code_mean = float(member_df["coding_questions_solved"].mean()) if member_df["coding_questions_solved"].notna().any() else 50.0
        cert_mean = float(member_df["certifications_count"].mean()) if member_df["certifications_count"].notna().any() else 1.0
        proj_mean = float(member_df["projects_count"].mean()) if member_df["projects_count"].notna().any() else 1.0
        exp_mean = float(member_df["experience_months"].mean()) if member_df["experience_months"].notna().any() else 0.0

        f1_comm = float(np.clip(comm_mean / 3.0 * 100.0, 0, 100))
        f3_code = float(np.clip(code_mean / 250.0 * 100.0, 15, 100))
        f5_cert = float(np.clip(cert_mean / 4.0 * 100.0, 10, 100))
        f6_proj = float(np.clip((0.65 * (proj_mean / 5.0) + 0.35 * (exp_mean / 6.0)) * 100.0, 10, 100))

        tier_scores, tier_qualified, tier_best_company = {}, {}, {}

        for t_rank in range(job_tiers):
            t_scored = scored[scored["tier_rank"] == t_rank].sort_values(
                ["skill_match", "salary_avg_lpa"], ascending=False
            )
            best_row = t_scored.iloc[0]
            f4_skill = float(best_row["skill_match"] * 100.0)
            req_cgpa = tier_benchmarks[t_rank]["min_cgpa"]
            if not np.isnan(req_cgpa):
                cgpa_ratio = float(np.clip(member_df["cgpa"].mean() / max(req_cgpa, 5.0), 0.5, 1.15))
            else:
                cgpa_ratio = float(np.clip(member_df["cgpa"].mean() / 10.0 * 1.15, 0.5, 1.15))
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
                and comp_score >= cutoff
            )
            comp_score = int(np.clip(comp_score, 15, 99))
            tier_scores[t_rank] = comp_score
            tier_qualified[t_rank] = bool(qual)
            tier_best_company[t_rank] = f"{best_row['company_name']} ({best_row['job_role']})"

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
            need_skills = ", ".join(low_best["missing"][:3]) or "python, sql, data structures"
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


def build_eligibility_and_utility_matrices(students_df, company_Capacities=None, j_offer=default_j_offer):
    """
    Constructs:
      - company_rows: deduplicated company/role offerings ordered by descending salary
      - qual_matrix (n x m bool): True iff student i clears hard eligibility gates,
        minimum skill match, and hierarchical tier restriction for company j
      - utility_matrix (n x m float): matching utility w_{i,j} = salary_j * (0.5 + 0.5 * skill_match_{i,j})
      - salary_vector (m float): salary_avg_lpa of each company slot
      - capacities (m int): slot capacity C_j per company
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
    skill_matrix = np.zeros((n_students, m_companies), dtype=float)
    utility_matrix = np.zeros((n_students, m_companies), dtype=float)
    salary_vector = company_rows["salary_avg_lpa"].to_numpy(dtype=float)

    for i in range(n_students):
        one_student = students_df.iloc[[i]]
        scored = score_jobs_for_group(one_student)
        comb_rank = (
            cgpa_band_rank[one_student["cgpa_band"].iloc[0]]
            + skill_names.index(one_student["skill_band"].iloc[0])
        )
        s_tier = 0 if comb_rank <= 1 else (1 if comb_rank == 2 else 2)
        for j, comp_row in company_rows.iterrows():
            comp_name = comp_row["company_name"]
            sub = scored[scored["company_name"] == comp_name].sort_values(
                ["skill_match", "salary_avg_lpa"], ascending=False
            ).iloc[0]
            is_qual = (
                int(sub["tier_rank"]) <= s_tier
                and float(sub["skill_match"]) >= min_skill_match
                and float(sub["eligible_fraction"]) >= min_eligible_fraction
            )
            qual_matrix[i, j] = is_qual
            sm = float(sub["skill_match"])
            skill_matrix[i, j] = sm
            utility_matrix[i, j] = float(comp_row["salary_avg_lpa"]) * (0.5 + 0.5 * sm) if is_qual else 0.0

    if company_Capacities is not None:
        capacities = np.asarray(company_Capacities, dtype=int).copy()
    elif "job_capacity" in company_rows.columns and company_rows["job_capacity"].notna().any():
        capacities = company_rows["job_capacity"].fillna(2).astype(int).to_numpy()
    else:
        default_cap = max(1, int(np.ceil((n_students * j_offer * 0.85) / max(m_companies, 1))))
        capacities = np.full(m_companies, default_cap, dtype=int)

    return company_rows, qual_matrix, utility_matrix, skill_matrix, salary_vector, capacities


def solve_capacity_constrained_allocation(qual_matrix, utility_matrix, salary_vector, capacities, j_offer=default_j_offer):
    """
    Mathematically Valid Capacity-Constrained Bipartite Assignment (Phase 3D):
    Formulates student-company allocation as an exact Mixed-Integer Linear Program (MILP).
    """
    n_students, m_companies = qual_matrix.shape
    assigned = np.zeros((n_students, m_companies), dtype=bool)

    eligible_pairs = [(i, j) for i in range(n_students) for j in range(m_companies) if qual_matrix[i, j]]
    if not eligible_pairs:
        return assigned, {"status": "no_eligible_pairs", "optimal": True}

    n_x = len(eligible_pairs)
    n_y = n_students
    total_vars = n_x + n_y

    max_sal = float(np.max(salary_vector)) if len(salary_vector) > 0 else 20.0
    m_cov = 1000.0 * max_sal

    c = np.zeros(total_vars, dtype=float)
    for k, (i, j) in enumerate(eligible_pairs):
        c[k] = -float(utility_matrix[i, j])
    for i in range(n_students):
        c[n_x + i] = -m_cov

    rows_A = []
    rhs_ub = []

    student_to_x = {i: [] for i in range(n_students)}
    company_to_x = {j: [] for j in range(m_companies)}
    for k, (i, j) in enumerate(eligible_pairs):
        student_to_x[i].append(k)
        company_to_x[j].append(k)

    for i in range(n_students):
        if student_to_x[i]:
            row = np.zeros(total_vars, dtype=float)
            row[student_to_x[i]] = 1.0
            rows_A.append(row)
            rhs_ub.append(float(j_offer))

    for j in range(m_companies):
        if company_to_x[j]:
            row = np.zeros(total_vars, dtype=float)
            row[company_to_x[j]] = 1.0
            rows_A.append(row)
            rhs_ub.append(float(max(0, int(capacities[j]))))

    for i in range(n_students):
        row = np.zeros(total_vars, dtype=float)
        row[n_x + i] = 1.0
        if student_to_x[i]:
            row[student_to_x[i]] = -1.0
        rows_A.append(row)
        rhs_ub.append(0.0)

    A_ub = np.vstack(rows_A)
    b_ub = np.array(rhs_ub, dtype=float)

    try:
        constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))
        integrality = np.ones(total_vars)
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if res.success and res.x is not None:
            sol = res.x
            for k, (i, j) in enumerate(eligible_pairs):
                if sol[k] >= 0.5:
                    assigned[i, j] = True
            return assigned, {"status": "optimal_milp", "optimal": True, "fun": float(-res.fun)}
    except Exception as exc:
        log_info(f"MILP solver fallback triggered: {exc}")

    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs")
    if res_lp.success and res_lp.x is not None:
        sol = res_lp.x
        for k, (i, j) in enumerate(eligible_pairs):
            if sol[k] >= 0.5:
                assigned[i, j] = True
        return assigned, {"status": "optimal_lp", "optimal": True, "fun": float(-res_lp.fun)}

    slot_cols = []
    for j in range(m_companies):
        for _ in range(max(0, int(capacities[j]))):
            slot_cols.append(j)
    if slot_cols:
        cost_mat = np.full((n_students * j_offer, len(slot_cols)), 1e6, dtype=float)
        for i in range(n_students):
            for q in range(j_offer):
                r_idx = i * j_offer + q
                bonus = m_cov if q == 0 else 0.0
                for c_idx, j in enumerate(slot_cols):
                    if qual_matrix[i, j]:
                        cost_mat[r_idx, c_idx] = -(bonus + utility_matrix[i, j])
        r_ind, c_ind = linear_sum_assignment(cost_mat)
        for r_idx, c_idx in zip(r_ind, c_ind):
            if cost_mat[r_idx, c_idx] < 0:
                i = r_idx // j_offer
                j = slot_cols[c_idx]
                assigned[i, j] = True
    return assigned, {"status": "hungarian_slot_expansion", "optimal": True}


def summarize_allocation_metrics(students_df, company_rows, assigned, qual_matrix, salary_vector, capacities, j_offer=default_j_offer):
    """Computes comprehensive, non-fabricated allocation metrics."""
    n_students, m_companies = assigned.shape
    offers_per_student = assigned.sum(axis=1)
    elig_counts_per_student = qual_matrix.sum(axis=1)

    eligible_mask = elig_counts_per_student > 0
    placed_mask = offers_per_student > 0

    n_eligible = int(eligible_mask.sum())
    n_placed = int(placed_mask.sum())
    n_unassigned = int((~placed_mask).sum())
    n_unassigned_eligible = int((eligible_mask & (~placed_mask)).sum())
    n_ineligible = int((~eligible_mask).sum())

    unmet_quota_mask = eligible_mask & (offers_per_student < j_offer)
    n_unmet_quota = int(unmet_quota_mask.sum())

    assigned_salaries = []
    best_student_salaries = []
    for i in range(n_students):
        cols = np.where(assigned[i])[0]
        if len(cols) > 0:
            sals = salary_vector[cols]
            assigned_salaries.extend(list(sals))
            best_student_salaries.append(float(np.max(sals)))

    total_salary_obj = float(np.sum(assigned_salaries)) if assigned_salaries else 0.0
    mean_offer_salary = float(np.mean(assigned_salaries)) if assigned_salaries else 0.0
    mean_best_salary = float(np.mean(best_student_salaries)) if best_student_salaries else 0.0

    used_per_company = assigned.sum(axis=0)
    over_cap = np.maximum(0, used_per_company - capacities)
    capacity_violations = int(over_cap.sum())

    return {
        "total_students": int(n_students),
        "eligible_students": n_eligible,
        "ineligible_students": n_ineligible,
        "placed_students": n_placed,
        "placement_coverage_all_pct": float(n_placed / max(n_students, 1) * 100.0),
        "placement_coverage_eligible_pct": float(n_placed / max(n_eligible, 1) * 100.0),
        "unassigned_students": n_unassigned,
        "unassigned_eligible_students": n_unassigned_eligible,
        "unmet_quota_students": n_unmet_quota,
        "total_offers_assigned": int(offers_per_student.sum()),
        "mean_assigned_salary_lpa": mean_offer_salary,
        "mean_best_offer_salary_lpa": mean_best_salary,
        "total_salary_objective_lpa": total_salary_obj,
        "capacity_violations": capacity_violations,
    }


def student_company_mapping(students_df, j_offer=default_j_offer, capacities=None, method="optimal"):
    """
    Capacity-constrained student-to-company allocation (Phase 3D).
    Supports 'optimal', 'greedy_heuristic', 'fcfs', and 'random'.
    """
    (
        company_rows,
        qual_matrix,
        utility_matrix,
        skill_matrix,
        salary_vector,
        cap_vec,
    ) = build_eligibility_and_utility_matrices(students_df, company_Capacities=capacities, j_offer=j_offer)

    n_students, m_companies = qual_matrix.shape
    assigned = np.zeros((n_students, m_companies), dtype=bool)
    rem_cap = cap_vec.copy()

    if method == "optimal":
        assigned, solver_info = solve_capacity_constrained_allocation(
            qual_matrix, utility_matrix, salary_vector, cap_vec, j_offer=j_offer
        )
    elif method == "greedy_heuristic":
        rem_qual = qual_matrix.copy()
        for j in range(m_companies):
            if rem_cap[j] <= 0:
                rem_qual[:, j] = False
        processed = np.zeros(n_students, dtype=bool)
        for _ in range(n_students):
            qual_counts = rem_qual.sum(axis=1)
            pos_cands = [i for i in range(n_students) if not processed[i] and qual_counts[i] > 0]
            if not pos_cands:
                break
            z_idx = min(pos_cands, key=lambda idx: (qual_counts[idx], -float(students_df["cgpa"].iloc[idx])))
            processed[z_idx] = True
            avail_cols = sorted(
                [j for j in range(m_companies) if rem_qual[z_idx, j] and rem_cap[j] > 0],
                key=lambda j: utility_matrix[z_idx, j],
                reverse=True,
            )
            for j in avail_cols[:j_offer]:
                assigned[z_idx, j] = True
                rem_cap[j] -= 1
                if rem_cap[j] <= 0:
                    rem_qual[:, j] = False
        solver_info = {"status": "greedy_heuristic", "optimal": False}
    elif method == "fcfs":
        comp_order = np.argsort(salary_vector)
        offers_cnt = np.zeros(n_students, dtype=int)
        for j in comp_order:
            for i in range(n_students):
                if rem_cap[j] <= 0:
                    break
                if offers_cnt[i] < j_offer and qual_matrix[i, j]:
                    assigned[i, j] = True
                    offers_cnt[i] += 1
                    rem_cap[j] -= 1
        solver_info = {"status": "fcfs_baseline", "optimal": False}
    elif method == "random":
        rng = np.random.RandomState(random_seed)
        student_order = rng.permutation(n_students)
        for i in student_order:
            avail_cols = [j for j in range(m_companies) if qual_matrix[i, j] and rem_cap[j] > 0]
            if avail_cols:
                chosen = rng.choice(avail_cols, size=min(j_offer, len(avail_cols)), replace=False)
                for j in chosen:
                    assigned[i, j] = True
                    rem_cap[j] -= 1
        solver_info = {"status": "random_baseline", "optimal": False}
    else:
        raise ValueError(f"Unknown allocation method: {method}")

    alloc_metrics = summarize_allocation_metrics(
        students_df, company_rows, assigned, qual_matrix, salary_vector, cap_vec, j_offer=j_offer
    )
    alloc_metrics["solver_info"] = solver_info
    alloc_metrics["capacities"] = cap_vec.tolist()

    matrix_rows = []
    for i in range(n_students):
        cols = np.where(assigned[i])[0]
        assigned_names = [company_rows["company_name"].iloc[j] for j in cols]
        assigned_sals = [float(salary_vector[j]) for j in cols]
        elig_cnt = int(qual_matrix[i].sum())
        status_str = (
            "Placed (Quota Met)" if len(cols) >= j_offer else
            ("Placed (Partial Quota)" if len(cols) > 0 else
             ("Unassigned (Capacity Starved)" if elig_cnt > 0 else "Unassigned (Ineligible)"))
        )
        row_dict = {
            "student_id": students_df["student_id"].iloc[i],
            "group": f"G{students_df['group_number'].iloc[i]}",
            "eligible_companies": elig_cnt,
            "offers_assigned": len(cols),
            "assigned_companies": ", ".join(assigned_names) if assigned_names else "-",
            "mean_offer_lpa": f"{np.mean(assigned_sals):.2f}" if assigned_sals else "0.00",
            "allocation_status": status_str,
        }
        for j, comp_row in company_rows.iterrows():
            col_label = comp_row["company_name"].upper()[:12]
            row_dict[col_label] = "*" if assigned[i, j] else "-"
        matrix_rows.append(row_dict)

    return pd.DataFrame(matrix_rows), company_rows, assigned, qual_matrix, alloc_metrics


# ==============================================================================
# SECTION 10: CONSOLE TABLE PRINTERS & CLUSTERING DISTRIBUTION GRAPHS (PHASE 5 & 6)
# ==============================================================================
def print_table(title, table, wrap_ids=False):
    """Renders any DataFrame inside a clean Unicode box-drawing grid."""
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
    if is_qual and is_opt:
        return f"✓ {score:>2d} ✔"
    if is_qual:
        return f"✓ {score:>2d}  "
    return f"  {score:>2d}  "


def print_springer_mapping_box(title, mapping_df, tier_benchmarks):
    print("\n" + title)
    if len(mapping_df) == 0:
        return

    w_id = max(12, int(mapping_df["entity_id"].astype(str).str.len().max()) + 2)
    w_job = 21
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


def save_clustering_distribution_figures():
    """
    Generates the three independent 300-DPI clustering figures inside output/clustering/:
      1. output/clustering/cgpa_distribution.png
      2. output/clustering/skill_cluster_distribution.png
      3. output/clustering/job_tier_distribution.png
    """
    if not GENERATE_CLUSTERING_DISTRIBUTION_PLOTS:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_output_directories()

    # 1. CGPA-Band Distribution (cgpa_distribution.png)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    band_labels = ["Low CGPA\n(< 6.0)", "Mid CGPA\n(6.0 to < 8.0)", "Top CGPA\n(>= 8.0)"]
    tr_counts = [int((train_students["cgpa_band"] == b).sum()) for b in ["low", "mid", "top"]]
    ch_counts = [int((checked_students["cgpa_band"] == b).sum()) for b in ["low", "mid", "top"]]
    tr_pcts = [c / max(len(train_students), 1) * 100.0 for c in tr_counts]
    ch_pcts = [c / max(len(checked_students), 1) * 100.0 for c in ch_counts]

    x = np.arange(len(band_labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, tr_pcts, w, label=f"Training Set (n={len(train_students)}, μ={train_students['cgpa'].mean():.2f})", color="#1d4ed8", edgecolor="#1e3a8a")
    b2 = ax.bar(x + w / 2, ch_pcts, w, label=f"Testing Set (n={len(checked_students)}, μ={checked_students['cgpa'].mean():.2f})", color="#0f766e", edgecolor="#115e59")
    ax.set_xticks(x, band_labels, fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Percentage of Students (%)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Level-1 Academic CGPA Category", fontweight="bold", fontsize=10)
    ax.set_title("Student Academic CGPA-Band Distribution (Training vs. Testing)", fontweight="bold", fontsize=11, pad=12)
    ax.set_ylim(0, max(tr_pcts + ch_pcts + [40.0]) * 1.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, cnt in zip(b1, tr_counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    for bar, cnt in zip(b2, ch_counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#115e59")
    fig.tight_layout()
    cgpa_fig_path = os.path.join(OUTPUT_CLUSTER_DIR, "cgpa_distribution.png")
    fig.savefig(cgpa_fig_path, dpi=300)
    plt.close(fig)

    # 2. Student Skill-Cluster Distribution across CGPA Bands (skill_cluster_distribution.png)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(3)
    w = 0.25
    colors = {"low": "#dc2626", "mid": "#d97706", "good": "#15803d"}
    max_bar_h = 10
    for s_idx, s_name in enumerate(skill_names):
        counts = [
            int(((train_students["cgpa_band"] == b) & (train_students["skill_band"] == s_name)).sum())
            for b in ["low", "mid", "top"]
        ]
        max_bar_h = max(max_bar_h, max(counts + [1]))
        bars = ax.bar(
            x + (s_idx - 1) * w, counts, w,
            label=f"Skill Sub-Cluster: {s_name.capitalize()}",
            color=colors[s_name], alpha=0.9, edgecolor="#0f172a"
        )
        for bar, b_name in zip(bars, ["low", "mid", "top"]):
            h = bar.get_height()
            b_tot = max(int((train_students["cgpa_band"] == b_name).sum()), 1)
            pct = h / b_tot * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{int(h)}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_xticks(x, band_labels, fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Level-1 Academic CGPA Category", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of Students (Training Cohort)", fontweight="bold", fontsize=10)
    ax.set_title("Hierarchical Level-2 K-Means Student Skill-Cluster Distribution", fontweight="bold", fontsize=11, pad=12)
    ax.set_ylim(0, max_bar_h * 1.28)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    skill_fig_path = os.path.join(OUTPUT_CLUSTER_DIR, "skill_cluster_distribution.png")
    fig.savefig(skill_fig_path, dpi=300)
    plt.close(fig)

    # 3. Job-Tier Distribution (job_tier_distribution.png)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    tier_labels = ["Low Tier\n(Entry / Mass Service)", "Mid Tier\n(Specialist / Digital)", "Top Tier\n(Product / High CTC)"]
    t_counts = [int((jobs["tier_rank"] == r).sum()) for r in range(job_tiers)]
    t_means = [float(jobs[jobs["tier_rank"] == r]["salary_avg_lpa"].mean()) if (jobs["tier_rank"] == r).any() else 0.0 for r in range(job_tiers)]
    t_mins = [float(jobs[jobs["tier_rank"] == r]["salary_avg_lpa"].min()) if (jobs["tier_rank"] == r).any() else 0.0 for r in range(job_tiers)]
    t_maxs = [float(jobs[jobs["tier_rank"] == r]["salary_avg_lpa"].max()) if (jobs["tier_rank"] == r).any() else 0.0 for r in range(job_tiers)]

    bar_colors = ["#64748b", "#0284c7", "#15803d"]
    bars = ax.bar(tier_labels, t_counts, color=bar_colors, width=0.50, edgecolor="#0f172a")
    ax.set_xlabel("Unsupervised Salary-Weighted Job Tier (K=3)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of Job Postings", fontweight="bold", fontsize=10)
    ax.set_title("Company / Job-Tier Distribution & Package Ranges (LPA)", fontweight="bold", fontsize=11, pad=12)
    ax.set_ylim(0, max(t_counts + [5]) * 1.30)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, cnt, mn, mx, avg in zip(bars, t_counts, t_mins, t_maxs, t_means):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + max(t_counts + [1]) * 0.03,
            f"{cnt} roles\n({mn:.1f}–{mx:.1f} LPA, μ={avg:.1f})",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a"
        )
    fig.tight_layout()
    job_fig_path = os.path.join(OUTPUT_CLUSTER_DIR, "job_tier_distribution.png")
    fig.savefig(job_fig_path, dpi=300)
    plt.close(fig)
    log_info("Saved clustering distribution figures in output/clustering/.")


# ==============================================================================
# SECTION 11: MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    ensure_output_directories()
    log_info("Starting career_matcher.main() execution.")

    print(f"\n[Dataset Integration & Provenance Summary]")
    print(f"  - Job dataset      : {len(jobs)} valid roles from {job_file_name}")
    print(f"  - Training students: {len(train_students)} valid records from {train_student_file_name} (Synthetic dataset)")
    print(f"  - Testing students : {len(checked_students)} valid unseen records from {check_student_file_name} (Synthetic dataset)")
    print(f"  - Common Level-2 clustering features: {skill_feature_columns}")
    if unavailable_check_features:
        print(f"  - Features unavailable in testing dataset (excluded, not fabricated): {unavailable_check_features}")

    print("\nJob tiers found by salary-weighted clustering (K=3):")
    for rank, tier in enumerate(tier_names):
        part = jobs[jobs["tier_rank"] == rank]
        if len(part) > 0:
            print(f"  {tier:<4} tier: {len(part):>3} roles, salary {part['salary_avg_lpa'].min():.1f} "
                  f"to {part['salary_avg_lpa'].max():.1f} LPA (avg {part['salary_avg_lpa'].mean():.1f} LPA)")
    print(f"Job tier silhouette score: {silhouette_score(job_scaled, jobs['job_cluster']):.3f}")

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
    combined_matching_results = pd.concat([apply_all, upgrade_all], ignore_index=True)

    filled = int((summary_table["students"] > 0).sum())
    print_table(
        f"Table 1: Student Cohort Groups ({filled} of {len(group_records)} groups populated; margin score = 1 - d1/d2)",
        summary_table, wrap_ids=True,
    )
    print_table(
        f"Table 2: 'Apply Now' Eligible Roles per Cohort (top {roles_to_show} roles at or below cohort tier)",
        apply_all,
    )
    print_table("Table 3: 'Upgrade Path' Target Roles in the Next Achievable Tier", upgrade_all)
    print_table("Table 4: Prioritized Missing Technical Skills to Learn per Cohort", learn_table)

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
        "Table 5a: Cohort-to-Job Capability & Max-Salary Mapping Matrix (Springer Table 1 & 2 Format)",
        group_map_df, tier_benchmarks,
    )
    print_springer_mapping_box(
        "Table 5b: Individual Student-to-Job Mapping Matrix (Representative Student per Populated Cohort)",
        student_map_df, tier_benchmarks,
    )

    mapping_df, _, _, _, alloc_metrics = student_company_mapping(
        checked_students, j_offer=default_j_offer, method="optimal"
    )
    print_table(
        f"Table 6: Capacity-Constrained Optimal Campus Allocation Matrix (first 12 students; j_offer = {default_j_offer})",
        mapping_df.head(12),
    )
    print(
        f"\n[Allocation Summary] Eligible Placed: {alloc_metrics['placed_students']}/{alloc_metrics['eligible_students']} "
        f"(Coverage: {alloc_metrics['placement_coverage_all_pct']:.1f}% of all, "
        f"{alloc_metrics['placement_coverage_eligible_pct']:.1f}% of eligible) | "
        f"Mean Offer Salary: {alloc_metrics['mean_assigned_salary_lpa']:.2f} LPA | "
        f"Total Salary Objective: {alloc_metrics['total_salary_objective_lpa']:.2f} LPA | "
        f"Capacity Violations: {alloc_metrics['capacity_violations']} | "
        f"Unassigned Students: {alloc_metrics['unassigned_students']} | "
        f"Unmet Offer Quotas: {alloc_metrics['unmet_quota_students']}"
    )

    save_clustering_distribution_figures()

    if save_tables:
        summary_table.to_csv(output_summary_file, index=False)
        combined_matching_results.to_csv(output_matching_file, index=False)
        mapping_df.to_csv(output_mapping_file, index=False)
        apply_all.to_csv(os.path.join(OUTPUT_TABLE_DIR, "apply_now_recommendations.csv"), index=False)
        upgrade_all.to_csv(os.path.join(OUTPUT_TABLE_DIR, "upgrade_path_recommendations.csv"), index=False)
        learn_table.to_csv(os.path.join(OUTPUT_TABLE_DIR, "skills_to_learn_roadmap.csv"), index=False)
        group_map_df.to_csv(os.path.join(OUTPUT_TABLE_DIR, "springer_group_mapping_matrix.csv"), index=False)
        print(f"\nSaved all tables & reports inside '{OUTPUT_ROOT}/':")
        print(f"  - {output_summary_file}")
        print(f"  - {output_matching_file}")
        print(f"  - {output_mapping_file}")
        print(f"  - {output_preprocessing_report}")

    log_info("Completed career_matcher.main() successfully.")
    print("\ndone")


if __name__ == "__main__":
    main()