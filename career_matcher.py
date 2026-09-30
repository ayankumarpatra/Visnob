# career_matcher.py
# single model: trains on the job sheet and the historical student sheet, then
# checks a whole batch of new students at once and prints one result per group.
# steps: load jobs -> cluster jobs into tiers -> load student_data.csv (training)
#        -> cgpa bands -> train a skill sub-cluster model inside each band
#        -> load + validate check_student.csv (the batch to check)
#        -> put every checked student into a (cgpa band, skill band) group using
#           the trained models -> match each group with job roles -> print.
# needs: pip install pandas numpy scikit-learn

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# section 1: settings
# job_file_name and train_student_file_name are the training data, they are
# read once when the script starts and are not shown in the final printout.
# check_student_file_name is the batch of new students to check, it can have
# any number of rows, and every one of them appears in the printed groups.

job_file_name = "brands_data.csv"
train_student_file_name = "student_data.csv"
check_student_file_name = "check_student.csv"

random_seed = 42        # keeps every clustering result the same on every run
job_tiers = 3            # low, mid, top company tiers
skill_subclusters = 3    # low, mid, good skill groups inside one cgpa band
min_students_to_cluster = 10  # a cgpa band needs at least this many training
                               # students before a skill model is trained for it
tier_names = ["low", "mid", "top"]
skill_names = ["low", "mid", "good"]
group_tier_names = ["beginner", "mid", "top"]

min_skill_match = 0.5        # a role needs at least this average skill coverage
min_eligible_fraction = 0.5  # at least half the group must clear degree/cgpa/exp/comm
roles_to_show = 5            # "apply now" and "upgrade" roles printed per group

# soft skills are not matched like technical skills: "communication" is checked
# separately from the communication column, "problem solving" is assumed to be
# covered by coding practice, so both are removed from the skill matching.
soft_skills = {"communication", "problem solving"}

# common short forms students type, mapped to the names used in the job data
skill_aliases = {
    "ml": "machine learning", "dl": "deep learning", "ai": "machine learning",
    "oop": "object oriented programming", "oops": "object oriented programming",
    "dsa": "data structures", "ds": "data structures",
    "js": "javascript", "node": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "cpp": "c++", "ci cd": "ci/cd", "cicd": "ci/cd",
    "sklearn": "scikit-learn", "powerbi": "power bi", "reactjs": "react",
    "sql server": "sql", "mysql": "sql", "postgres": "sql", "postgresql": "sql",
}

# words typed when a student has nothing to list (treated as empty, not a skill)
empty_words = {"na", "n/a", "none", "no", "nil", "-"}

# solving many coding questions means the core skills below are practised too
coding_skill_threshold = 50
skills_from_coding = ["data structures", "algorithms"]

# communication text converted to a number (jobs asking for communication need 2+)
communication_scores = {"low": 1, "mid": 2, "good": 3}

# limits used to validate a student row, row is dropped if any of these is broken
field_limits = {
    "cgpa": (0, 10),
    "coding_questions_solved": (0, 100000),
    "projects_count": (0, 100),
    "experience_months": (0, 360),
    "certifications_count": (0, 50),
}

# many people may type a column header differently, this maps every accepted
# spelling to the one name used in the rest of the code
column_aliases = {
    "student_id": ["student_id", "studentid", "id"],
    "degree": ["degree"],
    "branch": ["branch"],
    "cgpa": ["cgpa"],
    "coding_language": ["coding_language", "knowncodinglanguage", "primarylanguage", "language"],
    "coding_questions_solved": ["coding_questions_solved", "approxcodingquestionssolvedacrossallsites",
                                 "questionssolved"],
    "technical_skills": ["technical_skills", "technicalskills", "skills"],
    "projects_count": ["projects_count", "numberofprojects", "projects"],
    "experience_months": ["experience_months", "internshipworkexperienceinmonths", "experience"],
    "certifications_count": ["certifications_count", "numberofcertifications", "certifications"],
    "communication": ["communication", "communicationskill"],
}


# section 2: small cleaning helpers
# everything is converted to lowercase so the job sheet, the training sheet
# and the checked sheet all use exactly the same spelling.

def normalize_header(text):
    # "Number of Projects" / "number_of_projects " -> "numberofprojects"
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def clean_skill(text):
    text = str(text).strip().lower()
    return skill_aliases.get(text, text)


def split_skills(text):
    # "python, SQL ,ml" -> ["machine learning", "python", "sql"]
    if pd.isna(text) or str(text).strip() == "":
        return []
    parts = [clean_skill(part) for part in str(text).split(",")]
    return sorted(set(part for part in parts if part and part not in empty_words))


def clean_degree(text):
    # "B.Tech" / "b tech" -> "btech", "B.E" -> "be"
    return str(text).strip().lower().replace(".", "").replace(" ", "")


def parse_experience(value):
    # job sheet stores "fresher" or a number of years (assumed years, not months)
    value = str(value).strip().lower()
    if value == "fresher":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def cgpa_band(cgpa):
    # fixed, explainable cutoffs, chosen by hand rather than learned from data
    if cgpa > 8:
        return "top"
    if cgpa >= 6:
        return "mid"
    return "low"


cgpa_band_rank = {"low": 0, "mid": 1, "top": 2}
skill_feature_columns = ["skill_count", "coding_questions_solved", "projects_count",
                          "experience_months", "certifications_count"]


def rename_to_known_columns(raw):
    # matches whatever headers the file has to the canonical names used below
    rename_map = {}
    for canonical, spellings in column_aliases.items():
        for column in raw.columns:
            if normalize_header(column) in spellings:
                rename_map[column] = canonical
                break
    return raw.rename(columns=rename_map)


def row_is_valid(row):
    # returns a text reason if the row is bad, or none if every value makes sense
    for field, (low, high) in field_limits.items():
        value = row.get(field)
        try:
            value = float(value)
        except (ValueError, TypeError):
            return f"{field} is not a number ({row.get(field)!r})"
        if not low <= value <= high:
            return f"{field} out of range ({value}), expected {low} to {high}"
    return None


# section 3: load the job data and cluster the roles into tiers (low, mid, top)
# each row is one company + role. features are scaled first so big numbers
# (salary) do not overpower small ones (cgpa), then salary is given 3x weight
# because a tier is mainly about pay, without that weight the tiers overlap.

jobs = pd.read_csv(job_file_name)
jobs["required_list"] = jobs["required_skills"].apply(split_skills)
jobs["preferred_list"] = jobs["preferred_skills"].apply(split_skills)
jobs["degree_list"] = jobs["degrees_accepted"].apply(
    lambda text: [clean_degree(part) for part in str(text).split(",")]
)
jobs["exp_months"] = jobs["experience_required"].apply(parse_experience) * 12
jobs["needs_comm"] = jobs["required_list"].apply(lambda skills: "communication" in skills)

job_feature_columns = ["salary_avg_lpa", "min_cgpa", "exp_months", "coding_test_rounds"]
job_feature_weights = np.array([3, 1, 1, 1])
job_scaler = StandardScaler()
job_scaled = job_scaler.fit_transform(jobs[job_feature_columns].values) * job_feature_weights

job_model = KMeans(n_clusters=job_tiers, n_init=10, random_state=random_seed)
jobs["job_cluster"] = job_model.fit_predict(job_scaled)

# name the clusters by average salary: lowest salary cluster = low, highest = top
job_centers = job_scaler.inverse_transform(job_model.cluster_centers_ / job_feature_weights)
job_order = np.argsort(job_centers[:, 0])  # column 0 is salary
job_cluster_to_rank = {int(cluster_id): rank for rank, cluster_id in enumerate(job_order)}
jobs["tier_rank"] = jobs["job_cluster"].map(job_cluster_to_rank)

# every skill in the job sheet (used to drop skills a student typed that we don't know)
known_skills = set()
for skill_list in jobs["required_list"] + jobs["preferred_list"]:
    known_skills.update(skill_list)

# job arrays used later for fast, whole-group eligibility checks
job_min_cgpa = jobs["min_cgpa"].to_numpy()
job_exp_months = jobs["exp_months"].to_numpy()
job_needs_comm = jobs["needs_comm"].to_numpy()

# for every degree spelling seen in the job sheet, which jobs accept it
all_degrees = {clean_degree(d) for row in jobs["degree_list"] for d in row}
degree_accepts_job = {
    degree: jobs["degree_list"].apply(lambda row: degree in row).to_numpy()
    for degree in all_degrees
}


# section 4: train the student skill model on the historical student sheet
# this is the "training" step: cgpa bands are fixed, but inside each band a
# k-means model is fit on the training students so it learns what a "low",
# "mid" or "good" skill profile looks like for students of that cgpa band.
# these trained models are reused later on the batch of new students, they
# are never re-fit on the students being checked.

train_raw = rename_to_known_columns(pd.read_csv(train_student_file_name, dtype=str))
train_raw["problem"] = train_raw.apply(row_is_valid, axis=1)
dropped = train_raw["problem"].notna().sum()
train_raw = train_raw[train_raw["problem"].isna()].copy()
if dropped:
    print(f"training data: dropped {dropped} rows that failed validation")

train_students = pd.DataFrame({
    "cgpa": train_raw["cgpa"].astype(float),
    "skill_list": train_raw["technical_skills"].apply(split_skills),
    "coding_questions_solved": train_raw["coding_questions_solved"].astype(float),
    "projects_count": train_raw["projects_count"].astype(float),
    "experience_months": train_raw["experience_months"].astype(float),
    "certifications_count": train_raw["certifications_count"].astype(float),
})
train_students["skill_count"] = train_students["skill_list"].apply(len)
train_students["cgpa_band"] = train_students["cgpa"].apply(cgpa_band)

# band_models[band] holds the scaler + trained k-means + cluster-to-skill-name
# map for that cgpa band, or none if there were too few training rows in it
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
    # rank clusters by average scaled strength: low, mid, good
    strength = model.cluster_centers_.mean(axis=1)
    order = np.argsort(strength)
    cluster_to_rank = {int(cid): rank for rank, cid in enumerate(order)}
    band_models[band] = {"scaler": scaler, "model": model, "cluster_to_rank": cluster_to_rank}

print(f"trained on {len(train_students)} students from {train_student_file_name} "
      f"({dropped} rows dropped)")


# section 5: load and validate the batch of students to check
# every row is checked before use, a bad row is skipped and the reason is
# printed instead of guessing at a value that was never really there.

def load_students_to_check(file_path):
    # keep_default_na=False stops pandas turning a blank cell into NaN (which
    # would print as the text "nan" and slip past the "is it empty" checks)
    raw = rename_to_known_columns(pd.read_csv(file_path, dtype=str, keep_default_na=False))

    missing_columns = [c for c in column_aliases if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {', '.join(missing_columns)}")

    good_rows = []
    for _, row in raw.iterrows():
        student_id = str(row["student_id"]).strip()
        problem = validate_check_row(row)
        if problem:
            print(f"skipped student {student_id or '(no id)'}: {problem}")
            continue
        good_rows.append(build_student_record(row))

    if not good_rows:
        raise ValueError(f"no valid student rows found in {file_path}")
    return pd.DataFrame(good_rows)


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
    skills = split_skills(row["technical_skills"])
    language = clean_skill(row["coding_language"])
    skills = sorted(set(skills + [language]))

    solved = float(row["coding_questions_solved"])
    if solved >= coding_skill_threshold:
        skills = sorted(set(skills + skills_from_coding))
    skills = [skill for skill in skills if skill in known_skills]

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


checked_students = load_students_to_check(check_student_file_name)
print(f"loaded {len(checked_students)} valid students from {check_student_file_name}")


# section 6: place every checked student into a (cgpa band, skill band) group
# using the models trained in section 4. no clustering is (re)fit here, each
# student is only passed through the model that was trained for their band.

def predict_skill_band(row, band):
    trained = band_models[band]
    if trained is None:
        return "mid"  # not enough training data for this band, use a safe default
    features = [[row["skill_count"], row["coding_questions_solved"], row["projects_count"],
                 row["experience_months"], row["certifications_count"]]]
    scaled = trained["scaler"].transform(features)
    cluster_id = int(trained["model"].predict(scaled)[0])
    rank = trained["cluster_to_rank"][cluster_id]
    return skill_names[rank]


checked_students["cgpa_band"] = checked_students["cgpa"].apply(cgpa_band)
checked_students["skill_band"] = checked_students.apply(
    lambda row: predict_skill_band(row, row["cgpa_band"]), axis=1)

# one group per (cgpa_band, skill_band) combination that actually appears
# in the checked batch, tier = cgpa band rank + skill band rank added together,
# 0-1 -> beginner, 2 -> mid, 3-4 -> top
group_records = []
for (band, skill_band), member_index in checked_students.groupby(
        ["cgpa_band", "skill_band"]).groups.items():
    combined = cgpa_band_rank[band] + skill_names.index(skill_band)
    overall_tier = 0 if combined <= 1 else (1 if combined == 2 else 2)
    group_records.append({
        "cgpa_band": band, "skill_band": skill_band,
        "member_index": member_index, "tier_rank": overall_tier,
    })

print("note: a group's tier comes from cgpa band rank + skill band rank added "
      "together, 0-1 -> beginner, 2 -> mid, 3-4 -> top")


# section 7: match one group of students with the job sheet
# matching is done at the group level (average skill coverage, fraction of the
# group that is eligible), since the output is one result per group.

def group_skill_frequency(member_students):
    # fraction of the group that has each skill
    counts = Counter(skill for skills in member_students["skill_list"] for skill in skills)
    total = len(member_students)
    return {skill: count / total for skill, count in counts.items()}


def score_jobs_for_group(member_students):
    skill_freq = group_skill_frequency(member_students)
    cgpa_arr = member_students["cgpa"].to_numpy()[:, None]        # shape (n, 1)
    exp_arr = member_students["experience_months"].to_numpy()[:, None]
    comm_arr = member_students["comm_score"].to_numpy()[:, None]

    cgpa_ok = (cgpa_arr >= job_min_cgpa[None, :]).mean(axis=0)     # shape (m,)
    exp_ok = (exp_arr >= job_exp_months[None, :]).mean(axis=0)
    comm_ok_if_needed = (comm_arr >= 2).mean(axis=0)
    comm_ok = np.where(job_needs_comm, comm_ok_if_needed, 1.0)

    degree_ok_rows = np.vstack([degree_accepts_job.get(d, np.zeros(len(jobs), dtype=bool))
                                for d in member_students["degree"]])
    degree_ok = degree_ok_rows.mean(axis=0)

    eligible_fraction = (cgpa_ok + exp_ok + comm_ok + degree_ok) / 4

    skill_match, matched_list, missing_list = [], [], []
    for required, preferred in zip(jobs["required_list"], jobs["preferred_list"]):
        required = [s for s in required if s not in soft_skills]
        preferred = [s for s in preferred if s not in soft_skills]
        req_cov = np.mean([skill_freq.get(s, 0.0) for s in required]) if required else 1.0
        pref_cov = np.mean([skill_freq.get(s, 0.0) for s in preferred]) if preferred else 0.0
        skill_match.append(0.8 * req_cov + 0.2 * pref_cov)
        matched_list.append([s for s in required if skill_freq.get(s, 0.0) >= 0.5])
        missing_list.append([s for s in required if skill_freq.get(s, 0.0) < 0.5])

    result = jobs[["company_name", "job_role", "salary_avg_lpa", "tier_rank"]].copy()
    result["skill_match"] = skill_match
    result["eligible_fraction"] = eligible_fraction
    result["matched"] = matched_list
    result["missing"] = missing_list
    return result


def print_role_line(item):
    print(f"    - {item['job_role']} at {item['company_name']} | {item['salary_avg_lpa']:.1f} lpa | "
          f"skill match {item['skill_match'] * 100:.0f}% (approx for the group)")
    print(f"        required skills covered by most of the group: {', '.join(item['matched']) or 'none'}")
    if item["missing"]:
        print(f"        missing for most of the group: {', '.join(item['missing'])}")


def print_group(group_number, record):
    member_index = record["member_index"]
    member_students = checked_students.loc[member_index]
    tier_rank = record["tier_rank"]

    print("\n" + "=" * 60)
    print(f"student group {group_number}  (cgpa: {record['cgpa_band']}, skills: {record['skill_band']})")
    print("=" * 60)
    print(f"student ids: {', '.join(member_students['student_id'])}")
    print(f"tier: {group_tier_names[tier_rank]}")

    scored = score_jobs_for_group(member_students)

    ready = scored[(scored["tier_rank"] <= tier_rank)
                   & (scored["skill_match"] >= min_skill_match)
                   & (scored["eligible_fraction"] >= min_eligible_fraction)]
    ready = ready.sort_values(["skill_match", "salary_avg_lpa"], ascending=False)

    print(f"\n  roles this group can apply for now (top {roles_to_show}):")
    if len(ready) == 0:
        print("    none yet, see the upgrade path below")
    for _, item in ready.head(roles_to_show).iterrows():
        print_role_line(item)

    next_rank = min(tier_rank + 1, job_tiers - 1)
    print(f"\n  upgrade path (target tier: {group_tier_names[next_rank]}):")
    targets = scored[scored["tier_rank"] == next_rank].sort_values(
        ["skill_match", "salary_avg_lpa"], ascending=False).head(roles_to_show)
    print(f"  roles to aim for (top {len(targets)}):")
    for _, item in targets.iterrows():
        print_role_line(item)

    missing_counter = Counter(skill for skills in targets["missing"] for skill in skills)
    if missing_counter:
        print("\n  skills to learn (most needed across the target roles first):")
        for skill, count in missing_counter.most_common(6):
            print(f"    - {skill} (missing for {count} of {len(targets)} target roles)")


# section 8: main program

def main():
    print(f"\njob roles loaded: {len(jobs)}")
    print("job tiers found by clustering:")
    for rank, tier in enumerate(tier_names):
        part = jobs[jobs["tier_rank"] == rank]
        print(f"  {tier:<4} tier: {len(part):>3} roles, salary {part['salary_avg_lpa'].min():.1f} "
              f"to {part['salary_avg_lpa'].max():.1f} lpa (avg {part['salary_avg_lpa'].mean():.1f})")
    print(f"job silhouette score: {silhouette_score(job_scaled, jobs['job_cluster']):.2f}")

    for group_number, record in enumerate(group_records, start=1):
        print_group(group_number, record)

    print("\ndone")


if __name__ == "__main__":
    main()
